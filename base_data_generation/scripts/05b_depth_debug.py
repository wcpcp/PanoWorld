#!/usr/bin/env python
from __future__ import annotations

"""
05b_depth_debug.py

Quick depth alignment debug:
- Load 05b filtered entities.
- Load ERP depth image and scale.
- Compute per-entity depth stats within mask.
- Visualize depth map with entity bboxes and depth text.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_meta.io_utils import ensure_dir, load_json
from erp_meta.rle import decode_binary_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--depth_path", default="", help="Optional depth_image.png path. If empty, infer from erp_path directory.")
    ap.add_argument("--depth_scale_path", default="", help="Optional depth_scale.txt path. If empty, infer from erp_path directory.")
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--pmin", type=float, default=2.0, help="Lower percentile for depth visualization normalization")
    ap.add_argument("--pmax", type=float, default=98.0, help="Upper percentile for depth visualization normalization")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    obj = load_json(args.entities_json)
    entities = obj.get("entities", [])
    if args.max_entities:
        entities = entities[: int(args.max_entities)]

    erp_path = obj.get("erp_path", "")
    if not erp_path:
        raise SystemExit("entities_json has no erp_path; cannot infer depth paths.")

    base_dir = Path(erp_path).parent
    depth_path = Path(args.depth_path) if args.depth_path else base_dir / "depth_image.png"
    scale_path = Path(args.depth_scale_path) if args.depth_scale_path else base_dir / "depth_scale.txt"
    if not depth_path.exists():
        raise SystemExit(f"depth_image not found: {depth_path}")

    depth_scale = 1.0
    if scale_path.exists():
        try:
            depth_scale = float(scale_path.read_text(encoding="utf-8").strip())
        except Exception:
            depth_scale = 1.0

    depth_img = Image.open(depth_path)
    depth_raw = np.array(depth_img)
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    depth_raw = depth_raw.astype(np.float32)
    # Realsee3D depth_scale.txt is typically a large integer (e.g., 5000).
    # Convert raw depth to meters by dividing when scale > 1.
    if float(depth_scale) > 1.0:
        depth_m = depth_raw / float(depth_scale)
    else:
        depth_m = depth_raw * float(depth_scale)

    valid = depth_m > 0

    # 1. Provide a better visualization: Log scale + Heatmap
    # Using log scale helps distinguish structures in large rooms
    depth_vis_rgb = np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    if np.any(valid):
        # We use a log scale on valid depths
        log_depth = np.log1p(depth_m[valid])
        pmin = np.percentile(log_depth, float(args.pmin))
        pmax = np.percentile(log_depth, float(args.pmax))

        # Normalize valid points
        norm_valid = np.clip((log_depth - pmin) / max(1e-6, pmax - pmin), 0.0, 1.0)

        # Apply colormap (e.g., JET or INFERNO style) using matplotlib
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('plasma')
        colored = cmap(norm_valid)[:, :3] * 255.0 # RGBA -> RGB

        depth_vis_rgb[valid] = colored.astype(np.uint8)

    depth_vis_img = Image.fromarray(depth_vis_rgb, mode="RGB")
    depth_vis_img.save(out_dir / "depth_vis_heatmap.png", quality=95)

    erp_img = Image.open(erp_path).convert("RGB")
    erp_w, erp_h = erp_img.size
    erp_overlay = erp_img.copy().convert("RGBA")
    erp_draw = ImageDraw.Draw(erp_overlay)

    overlay = depth_vis_img.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    h, w = depth_m.shape
    stats: list[dict[str, Any]] = []
    for entity in entities:
        mask_rle = entity.get("mask_rle")
        if not mask_rle:
            continue
        mask = decode_binary_mask(mask_rle).astype(np.uint8)
        if mask.shape[:2] != depth_m.shape[:2]:
            # Skip mismatched size to avoid incorrect alignment.
            continue
        mask_bool = mask > 0
        if not np.any(mask_bool):
            continue
        values = depth_m[mask_bool]
        valid_values = values[values > 0]
        if valid_values.size == 0:
            continue

        d_med = float(np.median(valid_values))
        d_mean = float(np.mean(valid_values))
        d_p10 = float(np.percentile(valid_values, 10))
        d_p90 = float(np.percentile(valid_values, 90))
        valid_ratio = float(valid_values.size) / float(values.size)

        bbox = _mask_to_bbox_xyxy(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox

        # Calculate angular position within ERP
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        yaw_rad = (cx / w) * 2 * np.pi - np.pi          # [-pi, pi]
        pitch_rad = (cy / h) * np.pi - (np.pi / 2.0)      # [-pi/2, pi/2]

        # Calculate 3D position relative to camera using depth (distance to camera)
        # Using typical spherical to cartesian conversion:
        # X: Right, Y: Up, Z: Forward (or similar depending on convention, using typical computer vision here)
        # Assuming d_med is radial distance to camera center point:
        rx = d_med * np.cos(pitch_rad) * np.sin(yaw_rad)
        ry = d_med * np.sin(-pitch_rad) # Y is up
        rz = d_med * np.cos(pitch_rad) * np.cos(yaw_rad)

        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 255, 0), width=2)
        label = f"{entity.get('entity_id', '')[:3]} d={d_med:.1f}m"
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 255, 255), font=font)

        color = _color_from_id(entity.get("entity_id", ""))

        ex1, ey1, ex2, ey2 = _scale_bbox(x1, y1, x2, y2, w, h, erp_w, erp_h)
        erp_draw.rectangle((ex1, ey1, ex2 - 1, ey2 - 1), outline=color + (255,), width=2)
        erp_label = f"{entity.get('entity_id', '')[:3]} d={d_med:.1f}m"
        erp_draw.text((ex1 + 2, max(0, ey1 - 12)), erp_label, fill=(255, 255, 0, 255), font=font)

        stats.append(
            {
                "entity_id": entity.get("entity_id", ""),
                "label_open": entity.get("label_open", ""),
                "depth_median": d_med,
                "depth_mean": d_mean,
                "depth_p10": d_p10,
                "depth_p90": d_p90,
                "depth_valid_ratio": valid_ratio,
                "xyz_camera_m": [float(rx), float(ry), float(rz)],
                "yaw_deg": np.degrees(yaw_rad),
                "pitch_deg": np.degrees(pitch_rad),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

    overlay.save(out_dir / "depth_overlay_entities.png", quality=95)
    erp_overlay.convert("RGB").save(out_dir / "erp_overlay_entities.png", quality=95)
    (out_dir / "depth_entity_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"depth_vis_heatmap -> {out_dir / 'depth_vis_heatmap.png'}")
    print(f"depth_overlay -> {out_dir / 'depth_overlay_entities.png'}")
    print(f"erp_overlay -> {out_dir / 'erp_overlay_entities.png'}")
    print(f"stats -> {out_dir / 'depth_entity_stats.json'}")


def _mask_to_bbox_xyxy(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask01)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _scale_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    if src_w == 0 or src_h == 0:
        return x1, y1, x2, y2
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _color_from_id(text: str) -> tuple[int, int, int]:
    hval = 0
    for ch in text:
        hval = (hval * 131 + ord(ch)) & 0xFFFFFFFF
    r = 64 + (hval & 0x7F)
    g = 64 + ((hval >> 8) & 0x7F)
    b = 64 + ((hval >> 16) & 0x7F)
    return r, g, b


if __name__ == "__main__":
    main()
