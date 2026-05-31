#!/usr/bin/env python
from __future__ import annotations

"""
05c_depth_spatial.py

Attach depth-derived 3D spatial fields to ERP entities.
- Read entities JSON (from 04c/05/05b).
- Read depth_image.png + depth_scale.txt.
- Compute per-entity depth stats within mask.
- Derive yaw/pitch/xyz in camera coords from ERP position.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import multiprocessing as mp

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_meta.io_utils import ensure_dir, load_json
from erp_meta.rle import decode_binary_mask


def _run_one(entities_json: str, out_json: str, args: argparse.Namespace) -> None:
    obj = load_json(entities_json)
    entities = obj.get("entities", [])
    if args.max_entities:
        entities = entities[: int(args.max_entities)]

    erp_path = obj.get("erp_path", "")
    if not erp_path:
        raise SystemExit("entities_json has no erp_path; cannot infer depth paths.")

    base_dir = Path(erp_path).parent
    if args.depth_path:
        depth_path = Path(args.depth_path)
    else:
        depth_path = base_dir / "depth_image.png"
        manifest_style_depth = Path(erp_path.replace("/images/", "/images_dep/")).with_suffix(".png")
        if not depth_path.exists() and manifest_style_depth.exists():
            depth_path = manifest_style_depth

    if args.depth_scale_path:
        scale_path = Path(args.depth_scale_path)
    else:
        scale_path = base_dir / "depth_scale.txt"
        if not scale_path.exists():
            scale_path = depth_path.parent / "depth_scale.txt"

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
    if float(depth_scale) > 1.0:
        depth_m = depth_raw / float(depth_scale)
    else:
        depth_m = depth_raw * float(depth_scale)

    h, w = depth_m.shape

    out_entities: list[dict[str, Any]] = []
    for entity in entities:
        entity_out = dict(entity)
        mask_rle = entity.get("mask_rle")
        if not mask_rle:
            out_entities.append(entity_out)
            continue

        mask = decode_binary_mask(mask_rle).astype(np.uint8)
        if mask.shape[:2] != depth_m.shape[:2]:
            out_entities.append(entity_out)
            continue

        mask_bool = mask > 0
        if not np.any(mask_bool):
            out_entities.append(entity_out)
            continue

        values = depth_m[mask_bool]
        valid_values = values[values > 0]
        if valid_values.size < int(args.min_valid_points):
            entity_out["depth"] = {
                "status": "insufficient_valid",
                "valid_points": int(valid_values.size),
                "valid_ratio": float(valid_values.size) / float(values.size),
            }
            out_entities.append(entity_out)
            continue

        d_med = float(np.median(valid_values))
        d_mean = float(np.mean(valid_values))
        d_p10 = float(np.percentile(valid_values, 10))
        d_p90 = float(np.percentile(valid_values, 90))
        valid_ratio = float(valid_values.size) / float(values.size)

        # --- 修正：采用 seam-aware 的 yaw/pitch 计算方式 ---
        # 1. 取 mask 的 seam-aware 中心（球面重心）
        lon, lat = _mask_centroid_lonlat(mask)
        yaw_rad = float(lon)
        pitch_rad = float(lat)

        rx = d_med * np.cos(pitch_rad) * np.sin(yaw_rad)
        ry = d_med * np.sin(pitch_rad)
        rz = d_med * np.cos(pitch_rad) * np.cos(yaw_rad)

        entity_out["depth"] = {
            "status": "ok",
            "median_m": d_med,
            "mean_m": d_mean,
            "p10_m": d_p10,
            "p90_m": d_p90,
            "valid_ratio": valid_ratio,
            "valid_points": int(valid_values.size),
            "scale": float(depth_scale),
        }
        entity_out["spatial"] = {
            "yaw_deg": float(np.degrees(yaw_rad)),
            "pitch_deg": float(np.degrees(pitch_rad)),
            "xyz_camera_m": [float(rx), float(ry), float(rz)],
            "range_m": d_med,
        }

        # [修正位置] 将原本与函数体错位混在外部的 append 放回循环内
        out_entities.append(entity_out)

    out = dict(obj)
    out["entities"] = out_entities
    out_path = Path(out_json)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(getattr(args, "_do_viz", False)):
        viz_dir = Path(getattr(args, "_viz_dir", ""))
        if viz_dir:
            _save_erp_depth_viz(viz_dir, out)


def _mask_centroid_lonlat(mask: np.ndarray) -> tuple[float, float]:
    """
    计算二值 mask 在 ERP 球面上的 seam-aware 中心 (lon, lat)
    返回值单位：弧度
    """
    # 依赖 erp_meta.mask_ops.mask_centroid_lonlat，如果有则直接用
    try:
        from erp_meta.mask_ops import mask_centroid_lonlat
        return mask_centroid_lonlat(mask)
    except ImportError:
        # 兜底实现：简单平均法
        h, w = mask.shape[:2]
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0.0, 0.0
        lon = ((xs + 0.5) / w) * 2 * np.pi - np.pi
        lat = (np.pi / 2.0) - ((ys + 0.5) / h) * np.pi
        lon_c = float(np.mean(lon))
        lat_c = float(np.mean(lat))
        return lon_c, lat_c


def _run_one_task(task: tuple[str, str, argparse.Namespace]) -> None:
    entities_json, out_json, args = task
    # Create visualization directories inside workers so task generation stays lightweight.
    if getattr(args, "_do_viz", False) and getattr(args, "_viz_dir", ""):
        ensure_dir(Path(args._viz_dir))
    _run_one(entities_json, out_json, args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities_json", default="")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--index_views", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--entities_root", default="results_syn/05b_local_reground1")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--depth_path", default="", help="Optional depth_image.png path. If empty, infer from erp_path directory.")
    ap.add_argument("--depth_scale_path", default="", help="Optional depth_scale.txt path. If empty, infer from erp_path directory.")
    ap.add_argument("--min_valid_points", type=int, default=10, help="Min valid depth samples per entity")
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--viz_root", default="", help="Optional visualization root (ERP with depth labels)")
    ap.add_argument("--viz_limit", type=int, default=10, help="Only visualize first N views globally in index mode")
    args = ap.parse_args()

    if args.index_views:
        if not args.out_root:
            raise SystemExit("--out_root is required when using --index_views")
        index = load_json(args.index_views)
        items = index["items"]
        if args.shard_count > 1:
            items = [it for i, it in enumerate(items) if (i % args.shard_count) == args.shard_id]
        ent_root = Path(args.entities_root)
        out_root = Path(args.out_root)
        viz_root = Path(args.viz_root) if args.viz_root else None

        # Stream tasks while walking the index instead of materializing the full list first.
        def task_generator():
            task_count = 0
            for it in items:
                scene_id = it["scene_id"]
                vp = it["viewpoint_id"]
                entities_json = ent_root / scene_id / vp / "entities_reground.json"
                out_json = out_root / scene_id / vp / "entities_with_depth.json"

                if not entities_json.exists():
                    continue

                if args.skip_existing and out_json.exists():
                    continue

                local_args = argparse.Namespace(**vars(args))
                if viz_root is not None and (
                    int(args.viz_limit) <= 0 or task_count < int(args.viz_limit)
                ):
                    local_args._do_viz = True
                    local_args._viz_dir = str(viz_root / scene_id / vp)
                else:
                    local_args._do_viz = False
                    local_args._viz_dir = ""

                yield (str(entities_json), str(out_json), local_args)
                task_count += 1

        print(f'Running depth/spatial on views with num_workers={args.num_workers}...')

        if int(args.num_workers) > 1:
            with mp.Pool(processes=int(args.num_workers)) as pool:
                for _ in pool.imap_unordered(_run_one_task, task_generator(), chunksize=10):
                    pass
        else:
            for task in task_generator():
                _run_one_task(task)
        return

    if not args.entities_json or not args.out_json:
        raise SystemExit("Provide --entities_json and --out_json, or use --index_views")

    if args.viz_root:
        out_path = Path(args.out_json)
        scene_dir = out_path.parent.parent.name if len(out_path.parents) >= 2 else ""
        vp_dir = out_path.parent.name
        if scene_dir:
            args._viz_dir = str(Path(args.viz_root) / scene_dir / vp_dir)
        else:
            args._viz_dir = str(Path(args.viz_root) / vp_dir)
        ensure_dir(Path(args._viz_dir))
        args._do_viz = True
    else:
        args._do_viz = False
        args._viz_dir = ""

    _run_one(args.entities_json, args.out_json, args)
    print(f"depth_spatial -> {args.out_json}")


def _mask_to_bbox_xyxy(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask01)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _save_erp_depth_viz(viz_dir: Path, obj: dict[str, Any]) -> None:
    erp_path = obj.get("erp_path", "")
    if not erp_path:
        return
    try:
        erp_img = Image.open(str(erp_path)).convert("RGB")
    except Exception:
        return

    draw = ImageDraw.Draw(erp_img)
    width, height = erp_img.size
    for entity in obj.get("entities", []):
        depth = entity.get("depth", {})
        label = None
        if isinstance(depth, dict) and depth.get("status") == "ok":
            d = depth.get("median_m")
            if d is not None:
                label = f"{float(d):.2f}m"

        mask_rle = entity.get("mask_rle")
        if mask_rle:
            try:
                mask = decode_binary_mask(mask_rle).astype(np.uint8)
            except Exception:
                mask = None
            if mask is not None:
                if mask.shape[:2] != (height, width):
                    mask_img = Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)
                    mask = np.array(mask_img)
                _draw_seam_aware_mask_bbox(draw, mask, width, label)
                continue

        bbox = entity.get("bbox_xyxy")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        if label:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 255, 0))

    out_path = viz_dir / "erp_depth.jpg"
    ensure_dir(out_path.parent)
    erp_img.save(out_path, quality=95)


def _draw_seam_aware_mask_bbox(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    width: int,
    label: str | None,
) -> None:
    cols = np.any(mask > 0, axis=0)
    if not np.any(cols):
        return

    idx = np.where(cols)[0]
    if idx.size == 0:
        return
    idx = np.sort(idx)

    # Find largest empty gap on the circular axis to place the seam there.
    best_gap = -1
    best_start = 0
    for i in range(len(idx)):
        j = (i + 1) % len(idx)
        gap = (idx[j] - idx[i] - 1) % width
        if gap > best_gap:
            best_gap = gap
            best_start = idx[i]

    seam = (best_start + 1 + max(0, best_gap // 2)) % width
    left = mask[:, :seam]
    right = mask[:, seam:]

    drew_label = False
    if np.any(left):
        ys, xs = np.nonzero(left)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        if label:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 255, 0))
            drew_label = True

    if np.any(right):
        ys, xs = np.nonzero(right)
        x1, y1, x2, y2 = int(xs.min()) + seam, int(ys.min()), int(xs.max()) + 1 + seam, int(ys.max()) + 1
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        if label and not drew_label:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 255, 0))


if __name__ == "__main__":
    main()
