#!/usr/bin/env python
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from _common import load_cfg

from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.mask_ops import mask_centroid_lonlat
from erp_meta.rle import decode_binary_mask


def _export_one(entities_json: str, out_json: str, viz_dir: str | None = None) -> None:
    ent = load_json(entities_json)
    entities_out: list[dict] = []
    for entity in ent.get("entities", []):
        entity_out = dict(entity)
        if "bfov" not in entity_out:
            bfov = _compute_bfov_from_entity(entity_out)
            if bfov is not None:
                entity_out["bfov"] = bfov
        entities_out.append(entity_out)

    out = {
        "image_path": ent.get("erp_path", ""),
        "scene_id": ent.get("scene_id", ""),
        "viewpoint_id": ent.get("viewpoint_id", ""),
        "entities": entities_out,
        "quality_stats": ent.get("quality_stats", {}),
    }
    dump_json(out_json, out)
    if viz_dir:
        _save_erp_metadata_viz(Path(viz_dir), out)


def _compute_bfov_from_entity(entity: dict) -> dict | None:
    mask_rle = entity.get("mask_rle")
    if not mask_rle:
        return None
    try:
        mask = decode_binary_mask(mask_rle).astype(np.uint8)
    except Exception:
        return None
    if mask.ndim != 2:
        return None

    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return None

    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    lon, lat = mask_centroid_lonlat(mask)
    yaw_deg = float(np.degrees(lon))
    pitch_deg = float(np.degrees(lat))

    ys = np.where(rows)[0]
    lat_vals = (np.pi / 2.0) - (ys + 0.5) / float(h) * np.pi
    lat_min = float(lat_vals.min())
    lat_max = float(lat_vals.max())
    y_fov_rad = max(0.0, lat_max - lat_min)

    xs = np.where(cols)[0]
    x_fov_rad = _seam_aware_lon_span(xs, w)

    return {
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "x_fov_deg": float(np.degrees(x_fov_rad)),
        "y_fov_deg": float(np.degrees(y_fov_rad)),
    }


def _seam_aware_lon_span(xs: np.ndarray, width: int) -> float:
    if xs.size == 0 or width <= 0:
        return 0.0
    xs = np.sort(xs.astype(np.int32))
    if xs.size == 1:
        span_cols = 1
    else:
        gaps = xs[1:] - xs[:-1] - 1
        wrap_gap = int(xs[0] + width - xs[-1] - 1)
        max_gap = int(max(wrap_gap, int(gaps.max()) if gaps.size else 0))
        span_cols = width - max_gap
    span_cols = max(1, min(width, int(span_cols)))
    return float(span_cols) / float(width) * (2.0 * np.pi)


def _export_one_task(task: tuple[str, str, str]) -> None:
    """包装一下供 worker 进程调用，同时将原本主进程的 ensure_dir 移到此处"""
    ent_path, out_path, viz_dir = task
    if viz_dir:
        ensure_dir(Path(viz_dir))
    _export_one(ent_path, out_path, viz_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--entities_json", default="")
    ap.add_argument("--relations_json", default="")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--index_views", default="")
    ap.add_argument("--entities_root", default="results_test/05c_depth_spatial")
    ap.add_argument("--out_root", default="results_test/metadata")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--viz_root", default="", help="Optional visualization root (ERP with metadata labels)")
    ap.add_argument("--viz_limit", type=int, default=10, help="Only visualize first N views globally in index mode")
    args = ap.parse_args()

    _ = load_cfg(args.cfg)
    if args.index_views:
        index = load_json(args.index_views)
        items = index["items"]
        if args.shard_count > 1:
            items = [it for i, it in enumerate(items) if (i % args.shard_count) == args.shard_id]
        ent_root = Path(args.entities_root)
        out_root = Path(args.out_root)
        ensure_dir(out_root)

        viz_root = Path(args.viz_root) if args.viz_root else None

        # 优化点：使用生成器按需喂送任务，彻底消除启动时的遍历卡顿
        def task_generator():
            viz_count = 0
            for it in items:
                scene_id = it["scene_id"]
                vp = it["viewpoint_id"]
                ent_path = ent_root / scene_id / vp / "entities_with_depth.json"

                # --- 修改点：同时检查文件是否存在且不为空 ---
                if not ent_path.exists() or ent_path.stat().st_size == 0:
                    print("not exist, skip",ent_path)
                    continue

                out_path = out_root / scene_id / vp / "metadata.json"
                if args.skip_existing and out_path.exists():
                    continue

                viz_dir = ""
                if viz_root is not None and (int(args.viz_limit) <= 0 or viz_count < int(args.viz_limit)):
                    viz_dir = str(viz_root / scene_id / vp)
                    viz_count += 1

                yield (str(ent_path), str(out_path), viz_dir)

        print(f"Exporting metadata on views with num_workers={args.num_workers}...")

        if int(args.num_workers) > 1:
            with mp.Pool(processes=int(args.num_workers)) as pool:
                # 优化点：使用 imap_unordered 实现来一个跑一个
                for _ in pool.imap_unordered(_export_one_task, task_generator(), chunksize=10):
                    pass
        else:
            for task in task_generator():
                _export_one_task(task)
        return

    if not args.entities_json or not args.out_json:
        raise SystemExit("Provide --entities_json and --out_json, or use --index_views")

    viz_dir = ""
    if args.viz_root:
        out_path = Path(args.out_json)
        scene_dir = out_path.parent.parent.name if len(out_path.parents) >= 2 else ""
        vp_dir = out_path.parent.name
        if scene_dir:
            viz_dir = str(Path(args.viz_root) / scene_dir / vp_dir)
        else:
            viz_dir = str(Path(args.viz_root) / vp_dir)
        ensure_dir(Path(viz_dir))
    _export_one(args.entities_json, args.out_json, viz_dir)
    print(f"metadata -> {args.out_json}")


def _save_erp_metadata_viz(viz_dir: Path, obj: dict[str, Any]) -> None:
    image_path = obj.get("image_path", "")
    if not image_path:
        return
    try:
        erp_img = Image.open(str(image_path)).convert("RGB")
    except Exception:
        return

    draw = ImageDraw.Draw(erp_img)
    width, height = erp_img.size
    for entity in obj.get("entities", []):
        bfov = entity.get("bfov")
        if not isinstance(bfov, dict):
            continue
        anchor = _draw_bfov_box(draw, bfov, width, height)
        if anchor is not None:
            label_lines = _build_entity_label(entity)
            _draw_label(draw, anchor, label_lines)

    out_path = viz_dir / "erp_metadata.jpg"
    ensure_dir(out_path.parent)
    erp_img.save(out_path, quality=95)


def _build_entity_label(entity: dict) -> list[str]:
    entity_id = str(entity.get("entity_id", ""))
    semantic = entity.get("semantic", {})
    identify = ""
    if isinstance(semantic, dict):
        identify = str(semantic.get("identify", ""))
    if not identify:
        identify = str(entity.get("label_open", ""))

    depth = entity.get("depth", {})
    depth_str = ""
    if isinstance(depth, dict) and depth.get("status") == "ok":
        d = depth.get("median_m")
        if d is not None:
            depth_str = f"d={float(d):.2f}m"

    reg = entity.get("local_reground", {})
    reg_str = ""
    if isinstance(reg, dict):
        passed = reg.get("passed")
        iou = reg.get("consistency_iou")
        if iou is not None:
            reg_str = f"iou={float(iou):.2f}"
        if passed is not None:
            reg_str = f"{reg_str} pass={int(bool(passed))}".strip()

    line1 = ":".join([item for item in [entity_id, identify] if item])
    line2 = " ".join([item for item in [depth_str, reg_str] if item])
    lines = [line for line in [line1, line2] if line]
    return lines


def _draw_label(draw: ImageDraw.ImageDraw, origin: tuple[int, int], lines: list[str]) -> None:
    if not lines:
        return
    x1, y1 = origin
    y = max(0, y1 - 12 * len(lines))
    for line in lines:
        draw.text((x1 + 2, y), line, fill=(0, 255, 0))
        y += 12


def _draw_seam_aware_mask_bbox(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    width: int,
    label_lines: list[str],
) -> None:
    cols = np.any(mask > 0, axis=0)
    if not np.any(cols):
        return

    idx = np.where(cols)[0]
    if idx.size == 0:
        return
    idx = np.sort(idx)

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
        _draw_label(draw, (x1, y1), label_lines)
        drew_label = True

    if np.any(right):
        ys, xs = np.nonzero(right)
        x1, y1, x2, y2 = int(xs.min()) + seam, int(ys.min()), int(xs.max()) + 1 + seam, int(ys.max()) + 1
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        if not drew_label:
            _draw_label(draw, (x1, y1), label_lines)


def _draw_bfov_box(draw: ImageDraw.ImageDraw, bfov: dict, width: int, height: int) -> tuple[int, int] | None:
    yaw_deg = bfov.get("yaw_deg")
    pitch_deg = bfov.get("pitch_deg")
    x_fov_deg = bfov.get("x_fov_deg")
    y_fov_deg = bfov.get("y_fov_deg")
    if any(v is None for v in [yaw_deg, pitch_deg, x_fov_deg, y_fov_deg]):
        return None

    yaw = float(np.radians(float(yaw_deg)))
    pitch = float(np.radians(float(pitch_deg)))
    x_fov = float(np.radians(float(x_fov_deg)))
    y_fov = float(np.radians(float(y_fov_deg)))

    lon_c = yaw
    lat_c = pitch
    lon_min = lon_c - x_fov / 2.0
    lon_max = lon_c + x_fov / 2.0
    lat_min = max(-np.pi / 2.0, lat_c - y_fov / 2.0)
    lat_max = min(np.pi / 2.0, lat_c + y_fov / 2.0)

    def lon_to_x(lon_val: float) -> float:
        return (lon_val + np.pi) / (2.0 * np.pi) * float(width)

    def lat_to_y(lat_val: float) -> float:
        return (np.pi / 2.0 - lat_val) / np.pi * float(height)

    x1 = lon_to_x(lon_min)
    x2 = lon_to_x(lon_max)
    y1 = lat_to_y(lat_max)
    y2 = lat_to_y(lat_min)

    if x_fov >= 2.0 * np.pi:
        x1 = 0.0
        x2 = float(width)

    if x1 <= x2:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 128, 0), width=2)
        return int(round(x1)), int(round(y1))

    draw.rectangle((0, y1, x2, y2), outline=(255, 128, 0), width=2)
    draw.rectangle((x1, y1, float(width), y2), outline=(255, 128, 0), width=2)
    return int(round(x1)), int(round(y1))


if __name__ == "__main__":
    main()