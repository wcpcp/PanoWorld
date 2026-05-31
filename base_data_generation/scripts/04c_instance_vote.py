#!/usr/bin/env python
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _common import load_cfg

from erp_meta.erp_projection import backproject_mask_to_erp
from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.mask_ops import mask_centroid_lonlat, mask_iou, spherical_distance
from erp_meta.rle import encode_binary_mask
from erp_meta.view_sampling import ViewSpec, view_to_erp_maps


def _clip_box(bbox: list[float] | tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_iou(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _bbox_to_segments(bbox: list[int], erp_w: int) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    if x1 <= x2:
        return [[x1, y1, x2, y2]]
    return [[0, y1, x2, y2], [x1, y1, erp_w, y2]]


def _bbox_intersects(a: list[int], b: list[int], erp_w: int) -> bool:
    for ax1, ay1, ax2, ay2 in _bbox_to_segments(a, erp_w):
        for bx1, by1, bx2, by2 in _bbox_to_segments(b, erp_w):
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                return True
    return False


def _mask_iou_full(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    return mask_iou((mask_a > 0).astype(np.uint8), (mask_b > 0).astype(np.uint8))


def _seam_aware_bbox_from_mask(mask: np.ndarray) -> tuple[list[int], list[list[int]]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0], []
    h, w = mask.shape[:2]
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    cols = np.unique(xs.astype(np.int32))
    if cols.size == 1:
        bbox = [int(cols[0]), y1, int(cols[0]) + 1, y2]
        return bbox, [bbox]
    gap_best = -1
    gap_index = cols.size - 1
    for idx in range(cols.size - 1):
        gap = int(cols[idx + 1] - cols[idx] - 1)
        if gap > gap_best:
            gap_best = gap
            gap_index = idx
    wrap_gap = int(cols[0] + w - cols[-1] - 1)
    if wrap_gap > gap_best:
        gap_best = wrap_gap
        gap_index = cols.size - 1
    start = int(cols[(gap_index + 1) % cols.size])
    end = int(cols[gap_index]) + 1
    if gap_index == cols.size - 1:
        bbox = [start, y1, end, y2]
    else:
        bbox = [start, y1, end, y2]
    return bbox, _bbox_to_segments(bbox, w)


def _nms_view_detections(detections: list[dict], score_thr: float, nms_iou_thr: float, topk: int) -> tuple[list[dict], list[int]]:
    candidates: list[tuple[int, dict]] = []
    for raw_index, det in enumerate(detections):
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        score = float(det.get("score", 0.0))
        if not bbox or len(bbox) != 4 or score < score_thr:
            continue
        candidates.append((raw_index, det))
    candidates.sort(key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
    kept: list[dict] = []
    kept_indices: list[int] = []
    for raw_index, det in candidates:
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        if any(_box_iou(bbox, prev.get("bbox") or prev.get("bbox_xyxy")) >= nms_iou_thr for prev in kept):
            continue
        kept.append(det)
        kept_indices.append(raw_index)
        if topk > 0 and len(kept) >= topk:
            break
    return kept, kept_indices


def _project_detection(
    view: ViewSpec,
    det_index: int,
    raw_index: int,
    det: dict,
    map_x: np.ndarray,
    map_y: np.ndarray,
    pano_mask01: np.ndarray | None,
) -> dict | None:
    bbox = det.get("bbox") or det.get("bbox_xyxy")
    if not bbox:
        return None
    h, w = map_x.shape[:2]
    clipped = _clip_box(bbox, w, h)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    mask_view = np.zeros((h, w), dtype=np.uint8)
    mask_view[y1:y2, x1:x2] = 1
    mask_erp = backproject_mask_to_erp(mask_view, map_x, map_y, view.erp_w, view.erp_h).astype(np.uint8)
    if pano_mask01 is not None and pano_mask01.shape[:2] == mask_erp.shape[:2]:
        mask_erp = (mask_erp & pano_mask01).astype(np.uint8)
    area = int(mask_erp.sum())
    if area <= 0:
        return None
    bbox_erp, bbox_erp_segments = _seam_aware_bbox_from_mask(mask_erp)
    lonlat = mask_centroid_lonlat(mask_erp)
    return {
        "det_index": int(det_index),
        "raw_index": int(raw_index),
        "view_id": view.view_id,
        "view_type": view.view_type,
        "image_path": view.image_path,
        "label": str(det.get("label", "")),
        "score": float(det.get("score", 0.0)),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "bbox_erp": [int(v) for v in bbox_erp],
        "bbox_erp_segments": [[int(v) for v in seg] for seg in bbox_erp_segments],
        "area_erp": area,
        "lon_lat": [float(lonlat[0]), float(lonlat[1])],
        "mask_erp": mask_erp,
        "matched_views": set(),
        "support_count": 0,
        "support_score": 0.0,
        "conflict_count": 0,
        "conflict_score": 0.0,
        "overlap_checks": 0,
        "cluster_id": -1,
        "is_cluster_representative": False,
        "keep": False,
    }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _iter_grid_cells(bbox: list[int], erp_w: int, erp_h: int, cell_size: int) -> list[tuple[int, int]]:
    cells = []
    for x1, y1, x2, y2 in _bbox_to_segments(bbox, erp_w):
        gx1 = max(0, min((erp_w - 1) // cell_size, x1 // cell_size))
        gy1 = max(0, min((erp_h - 1) // cell_size, y1 // cell_size))
        gx2 = max(0, min((erp_w - 1) // cell_size, max(x2 - 1, x1) // cell_size))
        gy2 = max(0, min((erp_h - 1) // cell_size, max(y2 - 1, y1) // cell_size))
        for gy in range(gy1, gy2 + 1):
            for gx in range(gx1, gx2 + 1):
                cells.append((gx, gy))
    return sorted(set(cells))


def _draw_wrap_bbox(draw: ImageDraw.ImageDraw, bbox: list[int], erp_w: int, color: tuple[int, int, int], width: int = 2) -> None:
    for x1, y1, x2, y2 in _bbox_to_segments(bbox, erp_w):
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def _draw_perspective_detections(image_path: str, detections: list[dict], raw_indices: list[int] | None, out_path: Path, title: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for draw_idx, det in enumerate(detections):
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = (64, 255, 64)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        rid = raw_indices[draw_idx] if raw_indices is not None and draw_idx < len(raw_indices) else draw_idx
        draw.text((x1, max(0, y1 - 12)), f"{rid}:{float(det.get('score', 0.0)):.2f}", fill=color)
    draw.text((8, 8), title, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def _save_proj_viz(projected: list[dict], erp_h: int, erp_w: int, out_path: Path) -> None:
    mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
    for det in projected:
        det_mask = det.get("mask_erp")
        if det_mask is None:
            continue
        mask = np.maximum(mask, det_mask.astype(np.uint8))
    rgb = np.stack([np.zeros_like(mask), (mask * 255).astype(np.uint8), np.zeros_like(mask)], axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


def _draw_erp_overlay(
    erp_path: str,
    items: list[dict],
    out_path: Path,
    title: str,
    kept_ids: set[int] | None = None,
    suppressed_ids: set[int] | None = None,
) -> None:
    if not erp_path or not Path(erp_path).exists():
        return
    base = np.array(Image.open(erp_path).convert("RGB"), dtype=np.uint8)
    overlay = base.astype(np.float32).copy()
    for idx, item in enumerate(items):
        mask = item.get("mask_erp")
        if mask is None:
            continue
        gid = int(item.get("global_id", -1))
        if kept_ids is not None and gid in kept_ids:
            color = np.array([64, 255, 64], dtype=np.float32)
        elif suppressed_ids is not None and gid in suppressed_ids:
            color = np.array([255, 64, 64], dtype=np.float32)
        else:
            color = np.array([(idx * 53) % 255, (128 + idx * 71) % 255, (64 + idx * 37) % 255], dtype=np.float32)
        mask01 = mask.astype(bool)
        if np.any(mask01):
            overlay[mask01] = overlay[mask01] * 0.55 + color * 0.45
    img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    erp_w = img.size[0]
    for item in items:
        bbox = item.get("bbox_erp")
        if not bbox:
            continue
        gid = int(item.get("global_id", -1))
        if kept_ids is not None and gid in kept_ids:
            color = (64, 255, 64)
        elif suppressed_ids is not None and gid in suppressed_ids:
            color = (255, 64, 64)
        else:
            color = (255, 255, 0)
        _draw_wrap_bbox(draw, [int(v) for v in bbox], erp_w, color, width=2)
    draw.text((8, 8), title, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def _save_vote_viz(
    projected: list[dict],
    keep_set: set[tuple[str, int]],
    suppress_set: set[tuple[str, int]],
    erp_h: int,
    erp_w: int,
    out_path: Path,
) -> None:
    kept = np.zeros((erp_h, erp_w), dtype=np.uint8)
    suppressed = np.zeros((erp_h, erp_w), dtype=np.uint8)
    for det in projected:
        mask = det.get("mask_erp")
        if mask is None:
            continue
        view_id = str(det.get("view_id", ""))
        idx = int(det.get("det_index", -1))
        key = (view_id, idx)
        if key in keep_set:
            kept = np.maximum(kept, mask.astype(np.uint8))
        elif key in suppress_set:
            suppressed = np.maximum(suppressed, mask.astype(np.uint8))
    rgb = np.stack([
        (suppressed * 255).astype(np.uint8),
        (kept * 255).astype(np.uint8),
        np.zeros_like(kept, dtype=np.uint8),
    ], axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


def _draw_erp_cluster_viz(erp_path: str, clusters: list[dict], out_path: Path, title: str) -> None:
    img = Image.open(erp_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 255),
        (255, 128, 192),
        (192, 255, 128),
    ]
    for cluster in clusters:
        color = colors[int(cluster.get("cluster_id", 0)) % len(colors)]
        bbox = [int(v) for v in cluster.get("bbox_erp", [0, 0, 0, 0])]
        _draw_wrap_bbox(draw, bbox, img.size[0], color, width=3)
        text = f"C{cluster.get('cluster_id', -1)} v={cluster.get('support_views', 0)} s={float(cluster.get('best_score', 0.0)):.2f}"
        label_x = bbox[0] if bbox[0] <= bbox[2] else 0
        label_y = bbox[1]
        draw.text((max(0, label_x), max(0, label_y - 14)), text, fill=color)
    draw.text((8, 8), title, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def _make_cluster_summary(cluster_id: int, member_ids: list[int], all_dets: list[dict]) -> dict:
    members = [all_dets[idx] for idx in member_ids]
    rep = max(members, key=lambda item: float(item.get("score", 0.0)))
    union_mask = np.zeros_like(members[0]["mask_erp"], dtype=np.uint8)
    view_ids = set()
    for det in members:
        union_mask = np.maximum(union_mask, det["mask_erp"].astype(np.uint8))
        view_ids.add(str(det["view_id"]))
    bbox_erp, bbox_erp_segments = _seam_aware_bbox_from_mask(union_mask)
    lonlat = mask_centroid_lonlat(union_mask)
    mask_rle = encode_binary_mask(union_mask)
    area = int(union_mask.sum())
    h, w = union_mask.shape[:2]
    support_views = len(view_ids)
    support_count = max(0, support_views - 1)
    vote_score = float(rep.get("score", 0.0)) * (1.0 + float(support_count))
    return {
        "cluster_id": int(cluster_id),
        "entity_id": f"E{cluster_id:06d}",
        "representative_view_id": rep["view_id"],
        "representative_det_index": int(rep["det_index"]),
        "label_open": rep.get("label", ""),
        "confidence": float(rep.get("score", 0.0)),
        "bbox_erp": [int(v) for v in bbox_erp],
        "bbox_erp_segments": [[int(v) for v in seg] for seg in bbox_erp_segments],
        "bbox_xyxy": [int(v) for v in bbox_erp],
        "mask_rle": mask_rle,
        "lon_lat": [float(lonlat[0]), float(lonlat[1])],
        "area_ratio": float(area) / float(max(h * w, 1)),
        "source_views": sorted(view_ids),
        "best_score": float(rep.get("score", 0.0)),
        "support_views": int(support_views),
        "support_count": int(support_count),
        "vote_score": float(vote_score),
        "member_count": len(member_ids),
        "member_views": sorted(view_ids),
        "members": [
            {
                "view_id": det["view_id"],
                "det_index": int(det["det_index"]),
                "raw_index": int(det["raw_index"]),
                "score": float(det["score"]),
                "bbox": [int(v) for v in det["bbox"]],
                "bbox_erp": [int(v) for v in det["bbox_erp"]],
                "bbox_erp_segments": [[int(v) for v in seg] for seg in det.get("bbox_erp_segments", [])],
            }
            for det in members
        ],
    }


def _prune_cluster_fields(clusters: list[dict], args: argparse.Namespace) -> None:
    if not args.save_bbox_segments:
        for cluster in clusters:
            cluster.pop("bbox_erp_segments", None)
            members = cluster.get("members")
            if isinstance(members, list):
                for member in members:
                    member.pop("bbox_erp_segments", None)
    if not args.save_members:
        for cluster in clusters:
            cluster.pop("members", None)
            cluster.pop("member_views", None)


def _process_views_json(
    views_json: str,
    args: argparse.Namespace,
    det_root: Path,
    out_root: Path,
    viz_root: Path | None,
) -> str:
    views_obj = load_json(views_json)
    scene_id = views_obj["scene_id"]
    viewpoint_id = views_obj["viewpoint_id"]
    erp_path = views_obj.get("erp_path", "")
    views = [ViewSpec(**v) for v in views_obj["views"]]

    missing_det_jsons: list[Path] = []
    det_json_paths: list[Path] = []
    for view in views:
        det_json = det_root / scene_id / viewpoint_id / "detections" / f"{view.view_id}.json"
        det_json_paths.append(det_json)
        if not det_json.exists():
            missing_det_jsons.append(det_json)

    if missing_det_jsons:
        print(
            f"[wait] {scene_id}/{viewpoint_id} missing detections "
            f"{len(missing_det_jsons)}/{len(det_json_paths)}, skip for now",
            flush=True,
        )
        return "wait_missing"

    vp_out = out_root / scene_id / viewpoint_id
    ensure_dir(vp_out)
    out_json = vp_out / "instance_vote.json"
    if out_json.exists() and not args.overwrite:
        out_mtime = out_json.stat().st_mtime
        latest_det_mtime = max((p.stat().st_mtime for p in det_json_paths), default=0.0)
        if latest_det_mtime <= out_mtime:
            return "skip_exists"
        print(f"[refresh] {scene_id}/{viewpoint_id} detections updated after output, recomputing", flush=True)

    pano_mask01 = None
    pano_mask_path = views_obj.get("pano_mask_path")
    if pano_mask_path and Path(pano_mask_path).exists():
        try:
            pano_mask01 = (np.array(Image.open(pano_mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        except Exception:
            pano_mask01 = None

    projected_by_view: dict[str, list[dict]] = {}
    raw_dets_by_view: dict[str, list[dict]] = {}
    filtered_raw_indices_by_view: dict[str, list[int]] = {}
    erp_h = views[0].erp_h if views else 0
    erp_w = views[0].erp_w if views else 0

    for view in views:
        det_json = det_root / scene_id / viewpoint_id / "detections" / f"{view.view_id}.json"
        try:
            raw_dets = load_json(det_json)
        except Exception:
            print(f"[wait] {scene_id}/{viewpoint_id} detection file is not ready yet: {det_json}", flush=True)
            return "wait_missing"
        raw_dets_by_view[view.view_id] = raw_dets
        filtered_dets, kept_raw_indices = _nms_view_detections(
            raw_dets,
            score_thr=args.pre_score_thr,
            nms_iou_thr=args.view_nms_iou,
            topk=args.view_topk,
        )
        filtered_raw_indices_by_view[view.view_id] = kept_raw_indices

        if viz_root is not None and args.viz_persp:
            persp_dir = viz_root / scene_id / viewpoint_id / "perspective"
            _draw_perspective_detections(
                view.image_path,
                raw_dets,
                list(range(len(raw_dets))),
                persp_dir / f"{view.view_id}_raw.jpg",
                title=f"raw dets: {view.view_id}",
            )
            _draw_perspective_detections(
                view.image_path,
                filtered_dets,
                kept_raw_indices,
                persp_dir / f"{view.view_id}_filtered.jpg",
                title=f"filtered dets: {view.view_id}",
            )

        map_x, map_y = view_to_erp_maps(view)
        projected: list[dict] = []
        for det_index, (raw_index, det) in enumerate(zip(kept_raw_indices, filtered_dets)):
            proj = _project_detection(view, det_index, raw_index, det, map_x, map_y, pano_mask01)
            if proj is not None:
                projected.append(proj)
        projected_by_view[view.view_id] = projected

    all_dets: list[dict] = []
    for view_id in sorted(projected_by_view.keys()):
        for det in projected_by_view[view_id]:
            all_dets.append(det)

    cell_to_ids: dict[tuple[int, int], list[int]] = {}
    for det_id, det in enumerate(all_dets):
        det["global_id"] = det_id
        for cell in _iter_grid_cells(det["bbox_erp"], erp_w, erp_h, args.grid_size):
            cell_to_ids.setdefault(cell, []).append(det_id)

    candidate_pairs: set[tuple[int, int]] = set()
    for det_ids in cell_to_ids.values():
        if len(det_ids) < 2:
            continue
        det_ids = sorted(set(det_ids))
        for pos_a in range(len(det_ids)):
            for pos_b in range(pos_a + 1, len(det_ids)):
                a = det_ids[pos_a]
                b = det_ids[pos_b]
                if a != b:
                    candidate_pairs.add((a, b))

    uf = _UnionFind(len(all_dets))
    for a, b in candidate_pairs:
        da = all_dets[a]
        db = all_dets[b]
        if da["view_id"] == db["view_id"]:
            continue
        if not _bbox_intersects(da["bbox_erp"], db["bbox_erp"], erp_w):
            continue
        if args.iou_thr > 0:
            iou = _mask_iou_full(da["mask_erp"], db["mask_erp"])
            if iou >= args.iou_thr:
                uf.union(a, b)
                continue
        if args.dist_thr > 0:
            if spherical_distance(da["lon_lat"], db["lon_lat"]) <= args.dist_thr:
                uf.union(a, b)

    clusters: dict[int, list[int]] = {}
    for det_id in range(len(all_dets)):
        root = uf.find(det_id)
        clusters.setdefault(root, []).append(det_id)

    cluster_summaries: list[dict] = []
    cluster_member_ids: list[list[int]] = []
    det_cluster_by_view: dict[tuple[str, int], int] = {}
    rep_set: set[tuple[str, int]] = set()
    for cid, member_ids in enumerate(clusters.values()):
        cluster_member_ids.append(member_ids)
        cluster = _make_cluster_summary(cid, member_ids, all_dets)
        cluster_summaries.append(cluster)
        rep_set.add((str(cluster.get("representative_view_id", "")), int(cluster.get("representative_det_index", -1))))
        for det_id in member_ids:
            det = all_dets[det_id]
            key = (str(det.get("view_id", "")), int(det.get("det_index", -1)))
            det_cluster_by_view[key] = int(cid)

    keep_set: set[tuple[str, int]] = set()
    suppress_set: set[tuple[str, int]] = set()
    kept_clusters: list[dict] = []
    for cluster, member_ids in zip(cluster_summaries, cluster_member_ids):
        vote_score = float(cluster.get("vote_score", 0.0))
        support_matches = int(cluster.get("support_count", 0))
        if vote_score >= args.min_vote_score and support_matches >= args.min_support_matches:
            kept_clusters.append(cluster)
            if args.keep_mode == "all-supported":
                for det_id in member_ids:
                    det = all_dets[det_id]
                    keep_set.add((str(det.get("view_id", "")), int(det.get("det_index", -1))))
            else:
                keep_set.add(
                    (str(cluster.get("representative_view_id", "")), int(cluster.get("representative_det_index", -1)))
                )
        else:
            for det_id in member_ids:
                det = all_dets[det_id]
                suppress_set.add((str(det.get("view_id", "")), int(det.get("det_index", -1))))

    _prune_cluster_fields(cluster_summaries, args)
    _prune_cluster_fields(kept_clusters, args)

    views_output = []
    if args.save_views != "none":
        for view in views:
            view_dets = raw_dets_by_view.get(view.view_id, [])
            kept_raw_indices = filtered_raw_indices_by_view.get(view.view_id, [])
            view_rows = []
            for det_index, raw_index in enumerate(kept_raw_indices):
                det = view_dets[raw_index]
                view_key = (str(view.view_id), int(det_index))
                keep = view_key in keep_set
                representative = view_key in rep_set
                cluster_id = int(det_cluster_by_view.get(view_key, -1))
                if args.save_views == "full":
                    row = dict(det)
                else:
                    bbox = det.get("bbox") or det.get("bbox_xyxy") or [0, 0, 0, 0]
                    row = {
                        "label": str(det.get("label", "")),
                        "score": float(det.get("score", 0.0)),
                        "bbox": [int(v) for v in bbox],
                    }
                row.update(
                    {
                        "det_index": int(det_index),
                        "raw_index": int(raw_index),
                        "keep": bool(keep),
                        "representative": bool(representative),
                        "cluster_id": int(cluster_id),
                    }
                )
                view_rows.append(row)
            views_output.append({"view_id": view.view_id, "detections": view_rows})

    out = {
        "scene_id": scene_id,
        "viewpoint_id": viewpoint_id,
        "erp_path": erp_path,
        "views_json": views_json,
        "views": views_output,
        "entities": kept_clusters if args.keep_mode == "representative" else cluster_summaries,
        "summary": {
            "num_views": len(views),
            "num_raw_detections_total": int(sum(len(v) for v in raw_dets_by_view.values())),
            "num_projected_detections_total": int(len(all_dets)),
            "num_candidate_pairs": int(len(candidate_pairs)),
            "num_pairs_box_pruned": 0,
            "num_pairs_iou_evaluated": int(len(candidate_pairs)),
            "num_view_pairs_matched": int(sum(1 for _ in clusters.values())),
            "num_clusters": int(len(cluster_summaries)),
            "num_detections_kept": int(len(keep_set)),
            "num_detections_suppressed": int(len(suppress_set)),
            "keep_rate": float(len(keep_set)) / float(max(len(all_dets), 1)),
            "mean_vote_score": float(np.mean([c.get("vote_score", 0.0) for c in cluster_summaries])) if cluster_summaries else 0.0,
            "pre_score_thr": float(args.pre_score_thr),
            "view_nms_iou": float(args.view_nms_iou),
            "view_topk": int(args.view_topk),
            "grid_size": int(args.grid_size),
            "keep_mode": str(args.keep_mode),
            "save_views": str(args.save_views),
            "save_members": bool(args.save_members),
            "save_bbox_segments": bool(args.save_bbox_segments),
        },
    }

    dump_json(out_json, out)

    if viz_root is not None:
        erp_viz_dir = viz_root / scene_id / viewpoint_id
        if args.viz_proj:
            _save_proj_viz(all_dets, erp_h, erp_w, erp_viz_dir / "erp_proj.png")
        if args.viz_views:
            _save_vote_viz(all_dets, keep_set, suppress_set, erp_h, erp_w, erp_viz_dir / "erp_keep_suppress.png")
        if args.viz_overlay:
            _draw_erp_overlay(erp_path, all_dets, erp_viz_dir / "erp_overlay_all.jpg", title="ERP all clusters")
            _draw_erp_overlay(
                erp_path,
                all_dets,
                erp_viz_dir / "erp_overlay_filtered.jpg",
                title="ERP final filtered clusters",
                kept_ids={
                    int(det.get("global_id", -1))
                    for det in all_dets
                    if (str(det.get("view_id", "")), int(det.get("det_index", -1))) in keep_set
                },
                suppressed_ids={
                    int(det.get("global_id", -1))
                    for det in all_dets
                    if (str(det.get("view_id", "")), int(det.get("det_index", -1))) in suppress_set
                },
            )
        if args.viz_erp_final:
            _draw_erp_cluster_viz(erp_path, kept_clusters, erp_viz_dir / "erp_clusters_filtered.jpg", title="ERP final filtered clusters")

    return "done"


def _process_views_json_worker(payload: tuple[str, argparse.Namespace, Path, Path, Path | None]) -> str:
    views_json, args, det_root, out_root, viz_root = payload
    return _process_views_json(views_json, args, det_root, out_root, viz_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--index_views", help="index_views.json from 01_make_views.py")
    src_group.add_argument("--views_json", help="views.json from 01_make_views.py (single viewpoint)")
    ap.add_argument("--det_root", required=True, help="Output dir used in 02_detect.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pre_score_thr", type=float, default=0.3, help="Per-view score threshold before ERP projection")  #最初始的单个透视图检测的每个box的score过滤值
    ap.add_argument("--view_nms_iou", type=float, default=0.45, help="Per-view NMS IoU threshold")   #在每个透视图内部做 NMS，去掉同一视图中高度重叠的重复框。
    ap.add_argument("--view_topk", type=int, default=32, help="Max kept detections per view before ERP projection")
    ap.add_argument("--grid_size", type=int, default=256, help="ERP spatial hashing cell size for candidate generation") #把 ERP 按网格切块，只比较落入同一网格的检测，避免全对全 O(N^2)。   其实这个本质就是通过设置grid大小的网格，然后只在这个网格周围大小找相邻的detection
    ap.add_argument("--iou_thr", type=float, default=0.5, help="ERP projection IoU threshold for cross-view matching")
    ap.add_argument("--dist_thr", type=float, default=0.0, help="Optional spherical-distance fallback threshold in rad; 0 disables it") #球面距离兜底条件。即使 IoU 不够，只要两个投影中心在球面上足够近，也可以认为是同一实例
    ap.add_argument("--min_vote_score", type=float, default=0.8, help="Minimum vote score to keep a cluster")   #cluster 保留阶段    即使一个 cluster 的跨视图支持数不够，只要其代表框分数足够高，仍可保留
    ap.add_argument("--min_support_matches", type=int, default=1, help="Minimum number of supporting views to keep a cluster") #cluster 保留阶段    即使一个 cluster 的跨视图支持数不够，只要其代表框分数足够高，仍可保留
    ap.add_argument("--keep_mode", default="representative", choices=["representative", "all-supported"], help="Keep one representative per cluster or all members of kept clusters")
    ap.add_argument("--save_views", default="filtered", choices=["none", "filtered", "full"], help="Controls how much per-view detection data to store")
    ap.add_argument("--save_members", action="store_true", help="Store per-cluster member lists in output JSON")
    ap.add_argument("--save_bbox_segments", action="store_true", help="Store ERP seam bbox segments in output JSON")
    ap.add_argument("--viz_dir", default="", help="Optional visualization output dir")
    ap.add_argument("--viz_persp", action="store_true", help="Save raw/filtered perspective detection visualizations")
    ap.add_argument("--viz_proj", action="store_true", help="Save per-view ERP projected detection masks")
    ap.add_argument("--viz_views", action="store_true", help="Save per-view ERP keep/suppress masks")
    ap.add_argument("--viz_erp_final", action="store_true", help="Save final ERP cluster visualization")
    ap.add_argument("--viz_overlay", action="store_true", help="Save ERP overlay visualizations on top of panorama")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1, help="Number of CPU processes for index_views")
    args = ap.parse_args()

    _ = load_cfg(args.cfg)
    if args.views_json:
        index = {"items": [{"views_json": args.views_json}]}
    else:
        index = load_json(args.index_views)

    det_root = Path(args.det_root)
    out_root = Path(args.out_dir)
    viz_root = Path(args.viz_dir) if args.viz_dir else None
    ensure_dir(out_root)
    if viz_root is not None:
        ensure_dir(viz_root)
    if args.views_json:
        _process_views_json(args.views_json, args, det_root, out_root, viz_root)
        return

    tasks = [item["views_json"] for item in index["items"]]
    total_tasks = len(tasks)
    completed = 0
    done_count = 0
    wait_count = 0
    skip_count = 0
    if int(args.num_workers) > 1:
        payloads = [(vj, args, det_root, out_root, viz_root) for vj in tasks]
        with mp.Pool(processes=int(args.num_workers)) as pool:
            for status in pool.imap_unordered(_process_views_json_worker, payloads):
                completed += 1
                if status == "done":
                    done_count += 1
                elif status == "wait_missing":
                    wait_count += 1
                else:
                    skip_count += 1
                if completed % 1000 == 0 or completed == total_tasks:
                    print(
                        f"[progress] completed {completed}/{total_tasks} "
                        f"done={done_count} wait={wait_count} skip={skip_count}",
                        flush=True,
                    )
    else:
        for views_json in tasks:
            status = _process_views_json(views_json, args, det_root, out_root, viz_root)
            completed += 1
            if status == "done":
                done_count += 1
            elif status == "wait_missing":
                wait_count += 1
            else:
                skip_count += 1
            if completed % 1000 == 0 or completed == total_tasks:
                print(
                    f"[progress] completed {completed}/{total_tasks} "
                    f"done={done_count} wait={wait_count} skip={skip_count}",
                    flush=True,
                )


if __name__ == "__main__":
    main()