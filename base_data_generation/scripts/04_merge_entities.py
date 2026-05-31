#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _common import load_cfg

from erp_meta.erp_projection import backproject_mask_to_erp
from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.merge_entities import ProjectedInstance, merge_projected_instances
from erp_meta.rle import decode_binary_mask
from erp_meta.view_sampling import ViewSpec, view_to_erp_maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--index_views", help="index_views.json from 01_make_views.py")
    src_group.add_argument("--views_json", help="views.json from 01_make_views.py (single viewpoint)")
    ap.add_argument("--seg_root", default="", help="Output dir used in 03_segment.py")
    ap.add_argument("--det_root", default="", help="Output dir used in 02_detect.py")
    ap.add_argument("--overlap_verify_root", default="", help="Output dir used in 04b_overlap_verify.py")
    ap.add_argument("--instance_vote_root", default="", help="Output dir used in 04c_instance_vote.py")
    ap.add_argument("--min_view_consistency", type=float, default=0.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--viz_dir", default="", help="Optional ERP visualization output dir")
    ap.add_argument("--iou_thr", type=float, default=0.25)
    ap.add_argument("--dist_thr", type=float, default=0.25, help="rad")
    ap.add_argument("--sem_thr", type=float, default=0.2)
    ap.add_argument("--min_support_views", type=int, default=1)
    ap.add_argument("--min_entity_score", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _ = load_cfg(args.cfg)  # reserved for later
    if not args.seg_root and not args.det_root:
        raise SystemExit("Provide --seg_root or --det_root")

    if args.views_json:
        index = {"items": [{"views_json": args.views_json}]}
    else:
        index = load_json(args.index_views)
    seg_root = Path(args.seg_root) if args.seg_root else None
    det_root = Path(args.det_root) if args.det_root else None
    overlap_root = Path(args.overlap_verify_root) if args.overlap_verify_root else None
    instance_vote_root = Path(args.instance_vote_root) if args.instance_vote_root else None
    out_root = Path(args.out_dir)
    viz_root = Path(args.viz_dir) if args.viz_dir else None
    ensure_dir(out_root)
    if viz_root is not None:
        ensure_dir(viz_root)

    for item in index["items"]:
        views_json = item["views_json"]
        views_obj = load_json(views_json)
        scene_id = views_obj["scene_id"]
        viewpoint_id = views_obj["viewpoint_id"]
        erp_path = views_obj["erp_path"]

        vp_out = out_root / scene_id / viewpoint_id
        ensure_dir(vp_out)
        out_json = vp_out / "entities.json"
        if out_json.exists() and not args.overwrite:
            continue

        # load ERP size
        erp_img = Image.open(erp_path)
        erp_w, erp_h = erp_img.size

        pano_mask01 = None
        pano_mask_path = views_obj.get("pano_mask_path")
        if pano_mask_path and Path(pano_mask_path).exists():
            try:
                m = np.array(Image.open(pano_mask_path).convert("L"), dtype=np.uint8)
                pano_mask01 = (m > 0).astype(np.uint8)
            except Exception:
                pano_mask01 = None

        seg_dir = seg_root / scene_id / viewpoint_id / "segments" if seg_root else None
        det_dir = det_root / scene_id / viewpoint_id / "detections" if det_root else None
        instances = []
        bad_views = set()
        vote_keep_indices: dict[str, set[int]] = {}
        vote_summary = {}
        num_vote_kept = 0
        num_vote_total = 0
        if overlap_root is not None and args.min_view_consistency > 0:
            ov_path = overlap_root / scene_id / viewpoint_id / "overlap_verify.json"
            if ov_path.exists():
                try:
                    ov = load_json(ov_path)
                    scores = {}
                    for pair in ov.get("pairs", []):
                        a = pair.get("view_id_a")
                        b = pair.get("view_id_b")
                        s = float(pair.get("fg_iou", 0.0))
                        if a:
                            scores.setdefault(a, []).append(s)
                        if b:
                            scores.setdefault(b, []).append(s)
                    for view_id, vals in scores.items():
                        if vals and float(np.mean(vals)) < args.min_view_consistency:
                            bad_views.add(view_id)
                except Exception:
                    bad_views = set()
        if det_dir is not None and instance_vote_root is not None:
            vote_path = instance_vote_root / scene_id / viewpoint_id / "instance_vote.json"
            if vote_path.exists():
                try:
                    vote_obj = load_json(vote_path)
                    vote_summary = vote_obj.get("summary", {})
                    for view_row in vote_obj.get("views", []):
                        view_id = view_row.get("view_id")
                        keep_indices = {int(v) for v in view_row.get("keep_indices", [])}
                        if view_id is not None:
                            vote_keep_indices[str(view_id)] = keep_indices
                            num_vote_kept += len(keep_indices)
                            num_vote_total += int(view_row.get("num_detections", 0))
                except Exception:
                    vote_keep_indices = {}
                    vote_summary = {}
        if det_dir is None and seg_dir is not None:
            for v in views_obj["views"]:
                view_id = v["view_id"]
                if view_id in bad_views:
                    continue
                seg_json = seg_dir / f"{view_id}.json"
                if not seg_json.exists():
                    continue
                view = ViewSpec(**v)
                map_x, map_y = view_to_erp_maps(view)
                segs = json.loads(seg_json.read_text(encoding="utf-8"))
                for s in segs:
                    mask_view = decode_binary_mask(s["rle"]).astype(np.uint8)
                    mask_erp = backproject_mask_to_erp(mask_view, map_x, map_y, erp_w, erp_h)
                    if pano_mask01 is not None and pano_mask01.shape[:2] == mask_erp.shape[:2]:
                        mask_erp = (mask_erp.astype(np.uint8) & pano_mask01).astype(mask_erp.dtype)
                    instances.append(
                        ProjectedInstance(
                            view_id=view_id,
                            label=str(s.get("label", "")),
                            score=float(s.get("score", 0.0)),
                            mask_erp=mask_erp,
                        )
                    )
        if det_dir is not None:
            for v in views_obj["views"]:
                view_id = v["view_id"]
                if view_id in bad_views:
                    continue
                det_json = det_dir / f"{view_id}.json"
                if not det_json.exists():
                    continue
                view = ViewSpec(**v)
                map_x, map_y = view_to_erp_maps(view)
                h, w = map_x.shape[:2]
                dets = json.loads(det_json.read_text(encoding="utf-8"))
                keep_indices = vote_keep_indices.get(view_id)
                for det_index, d in enumerate(dets):
                    if keep_indices is not None and det_index not in keep_indices:
                        continue
                    bbox = d.get("bbox") or d.get("bbox_xyxy")
                    if not bbox or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    x1 = int(np.floor(x1))
                    y1 = int(np.floor(y1))
                    x2 = int(np.ceil(x2))
                    y2 = int(np.ceil(y2))
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    mask_view = np.zeros((h, w), dtype=np.uint8)
                    mask_view[y1:y2, x1:x2] = 1
                    mask_erp = backproject_mask_to_erp(mask_view, map_x, map_y, erp_w, erp_h)
                    if pano_mask01 is not None and pano_mask01.shape[:2] == mask_erp.shape[:2]:
                        mask_erp = (mask_erp.astype(np.uint8) & pano_mask01).astype(mask_erp.dtype)
                    instances.append(
                        ProjectedInstance(
                            view_id=view_id,
                            label=str(d.get("label", "")),
                            score=float(d.get("score", 0.0)),
                            mask_erp=mask_erp,
                        )
                    )

        entities = merge_projected_instances(instances, iou_thr=args.iou_thr, dist_thr_rad=args.dist_thr, sem_thr=args.sem_thr)
        entities = [
            entity
            for entity in entities
            if int(entity.get("source_view_count", 0)) >= args.min_support_views
            and float(entity.get("confidence", 0.0)) >= args.min_entity_score
        ]
        quality_stats = {
            "merge_mode": "det" if det_dir is not None else "seg",
            "num_instances_input": int(len(instances)),
            "num_entities_output": int(len(entities)),
            "num_filtered_views": int(len(bad_views)),
            "filtered_views": sorted(bad_views),
            "instance_vote_enabled": bool(det_dir is not None and instance_vote_root is not None and vote_keep_indices),
            "instance_vote_num_detections_total": int(num_vote_total),
            "instance_vote_num_detections_kept": int(num_vote_kept),
            "instance_vote_summary": vote_summary,
        }
        dump_json(
            out_json,
            {
                "scene_id": scene_id,
                "viewpoint_id": viewpoint_id,
                "erp_path": erp_path,
                "views_json": views_json,
                "entities": entities,
                "quality_stats": quality_stats,
            },
        )
        if viz_root is not None:
            _draw_entities_viz(
                erp_path=erp_path,
                entities=entities,
                out_path=viz_root / scene_id / viewpoint_id / "entities_viz.jpg",
            )
        print(f"[{scene_id}/{viewpoint_id}] entities={len(entities)} -> {out_json}")


def _draw_entities_viz(erp_path: str, entities: list[dict], out_path: Path) -> None:
    img = Image.open(erp_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 255),
    ]
    for idx, entity in enumerate(entities):
        color = colors[idx % len(colors)]
        x1, y1, x2, y2 = map(int, entity.get("bbox_xyxy", [0, 0, 0, 0]))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{entity.get('label_open', 'object')} | v={entity.get('source_view_count', 0)} | s={float(entity.get('confidence', 0.0)):.2f}"
        draw.text((max(0, x1), max(0, y1 - 14)), text, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


if __name__ == "__main__":
    main()
