#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load_cfg

from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.pipeline import build_segmenter
from erp_meta.types import DetBox


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--index_views", required=True)
    ap.add_argument("--det_root", required=True, help="Output dir used in 02_detect.py")
    ap.add_argument("--out_dir", required=True, help="Output root")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    segmenter = build_segmenter(cfg)

    index = load_json(args.index_views)
    det_root = Path(args.det_root)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    for item in index["items"]:
        views_json = item["views_json"]
        views = load_json(views_json)
        scene_id = views["scene_id"]
        viewpoint_id = views["viewpoint_id"]
        vp_out = out_root / scene_id / viewpoint_id
        ensure_dir(vp_out)

        seg_dir = vp_out / "segments"
        ensure_dir(seg_dir)

        det_dir = det_root / scene_id / viewpoint_id / "detections"
        for v in views["views"]:
            view_id = v["view_id"]
            det_json = det_dir / f"{view_id}.json"
            if not det_json.exists():
                continue
            out_json = seg_dir / f"{view_id}.json"
            if out_json.exists() and not args.overwrite:
                continue
            dets = json.loads(det_json.read_text(encoding="utf-8"))
            boxes = [DetBox(bbox_xyxy=tuple(d["bbox"]), label=d.get("label", ""), score=float(d.get("score", 0.0))) for d in dets]
            segmenter.segment_view(v["image_path"], boxes, str(out_json))

        dump_json(vp_out / "segments_index.json", {"views_json": views_json, "seg_dir": str(seg_dir), "det_dir": str(det_dir)})
        print(f"[{scene_id}/{viewpoint_id}] segments -> {seg_dir}")


if __name__ == "__main__":
    main()
