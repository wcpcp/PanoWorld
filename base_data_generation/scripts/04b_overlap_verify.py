#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _common import load_cfg

from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.overlap_verify import det_fg_mask_erp, footprint_erp, overlap_pair_metrics, seg_fg_mask_erp
from erp_meta.view_sampling import ViewSpec


def _load_pano_mask01(pano_mask_path: str | None) -> np.ndarray | None:
    if not pano_mask_path:
        return None
    p = Path(pano_mask_path)
    if not p.exists():
        return None
    try:
        m = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        return (m > 0).astype(np.uint8)
    except Exception:
        return None


def _is_lateral_persp(v: dict) -> bool:
    if v.get("view_type") != "persp":
        return False
    pitch = v.get("pitch_deg")
    if pitch is None:
        return True
    try:
        return abs(float(pitch)) < 1e-3
    except Exception:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--index_views", help="index_views.json from 01_make_views.py")
    src_group.add_argument("--views_json", help="views.json from 01_make_views.py (single viewpoint)")
    ap.add_argument("--seg_root", default="", help="Output dir used in 03_segment.py")
    ap.add_argument("--det_root", default="", help="Output dir used in 02_detect.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--viz_dir", default="", help="Optional ERP overlap heatmap output dir")
    ap.add_argument("--min_overlap_area", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--viz_views", action="store_true", help="Save per-view footprint/fg masks")
    ap.add_argument("--viz_pairs", action="store_true", help="Save per-pair overlap visualization")
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

        vp_out = out_root / scene_id / viewpoint_id
        ensure_dir(vp_out)
        out_json = vp_out / "overlap_verify.json"
        if out_json.exists() and not args.overwrite:
            continue

        pano_mask01 = _load_pano_mask01(views_obj.get("pano_mask_path"))

        # only consider lateral perspective faces for adjacent-overlap verification
        persp = [v for v in views_obj.get("views", []) if _is_lateral_persp(v)]
        if len(persp) < 2:
            dump_json(
                out_json,
                {
                    "scene_id": scene_id,
                    "viewpoint_id": viewpoint_id,
                    "views_json": views_json,
                    "status": "skipped",
                    "reason": "need >=2 lateral persp views",
                    "pairs": [],
                },
            )
            continue

        persp.sort(key=lambda v: float(v.get("yaw_deg") or 0.0))
        views = [ViewSpec(**v) for v in persp]

        seg_dir = seg_root / scene_id / viewpoint_id / "segments" if seg_root else None
        det_dir = det_root / scene_id / viewpoint_id / "detections" if det_root else None

        footprints = {}
        fg_masks = {}
        for v in views:
            footprints[v.view_id] = footprint_erp(v)
            if det_dir is not None:
                det_json = det_dir / f"{v.view_id}.json"
                dets = []
                if det_json.exists():
                    try:
                        dets = json.loads(det_json.read_text(encoding="utf-8"))
                    except Exception:
                        dets = []
                fg_masks[v.view_id] = det_fg_mask_erp(v, dets, pano_mask01=pano_mask01)
            else:
                seg_json = seg_dir / f"{v.view_id}.json" if seg_dir else None
                segs = []
                if seg_json is not None and seg_json.exists():
                    try:
                        segs = json.loads(seg_json.read_text(encoding="utf-8"))
                    except Exception:
                        segs = []
                fg_masks[v.view_id] = seg_fg_mask_erp(v, segs, pano_mask01=pano_mask01)

            if viz_root is not None and args.viz_views:
                vdir = viz_root / scene_id / viewpoint_id / "views"
                _save_mask(footprints[v.view_id], vdir / f"{v.view_id}_footprint.png")
                _save_mask(fg_masks[v.view_id], vdir / f"{v.view_id}_fg.png")

        pairs = []
        score_sum = np.zeros((views[0].erp_h, views[0].erp_w), dtype=np.float32)
        score_cnt = np.zeros((views[0].erp_h, views[0].erp_w), dtype=np.float32)
        for i in range(len(views)):
            a = views[i]
            b = views[(i + 1) % len(views)]
            overlap01 = (footprints[a.view_id] & footprints[b.view_id]).astype(np.uint8)
            if pano_mask01 is not None and pano_mask01.shape[:2] == overlap01.shape[:2]:
                overlap01 = (overlap01 & pano_mask01).astype(np.uint8)
            if int(overlap01.sum()) < args.min_overlap_area:
                continue

            m = overlap_pair_metrics(
                fg_masks[a.view_id],
                fg_masks[b.view_id],
                overlap01,
                view_id_a=a.view_id,
                view_id_b=b.view_id,
            )
            pairs.append(m.__dict__)
            score_sum += overlap01.astype(np.float32) * float(m.fg_iou)
            score_cnt += overlap01.astype(np.float32)

            if viz_root is not None and args.viz_pairs:
                pdir = viz_root / scene_id / viewpoint_id / "pairs"
                _save_pair_viz(
                    fg_masks[a.view_id],
                    fg_masks[b.view_id],
                    overlap01,
                    pdir / f"{a.view_id}__{b.view_id}.png",
                )

        mean_pair_iou = float(np.mean([float(p["fg_iou"]) for p in pairs])) if pairs else 0.0
        min_pair_iou = float(np.min([float(p["fg_iou"]) for p in pairs])) if pairs else 0.0

        dump_json(
            out_json,
            {
                "scene_id": scene_id,
                "viewpoint_id": viewpoint_id,
                "views_json": views_json,
                "status": "ok",
                "pairs": pairs,
                "summary": {
                    "num_pairs": len(pairs),
                    "mean_pair_iou": mean_pair_iou,
                    "min_pair_iou": min_pair_iou,
                },
            },
        )
        if viz_root is not None and pairs:
            _save_overlap_heatmap(score_sum, score_cnt, viz_root / scene_id / viewpoint_id / "overlap_heatmap.png")
        print(f"[{scene_id}/{viewpoint_id}] pairs={len(pairs)} -> {out_json}")


def _save_overlap_heatmap(score_sum: np.ndarray, score_cnt: np.ndarray, out_path: Path) -> None:
    mean = np.divide(score_sum, np.maximum(score_cnt, 1.0))
    covered = (score_cnt > 0).astype(np.uint8)
    norm = np.clip(mean, 0.0, 1.0)
    r = ((1.0 - norm) * 255.0).astype(np.uint8)
    g = (norm * 255.0).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[covered == 0] = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


def _save_mask(mask01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask01.astype(np.uint8) * 255)
    Image.fromarray(img).save(out_path)


def _save_pair_viz(fg_a: np.ndarray, fg_b: np.ndarray, overlap01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = (fg_a.astype(np.uint8) * 255)
    g = (fg_b.astype(np.uint8) * 255)
    b = (overlap01.astype(np.uint8) * 255)
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray(rgb).save(out_path)


if __name__ == "__main__":
    main()
