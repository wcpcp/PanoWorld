#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from _common import load_cfg

from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.pipeline import build_detector


def _parse_queries(raw: str):
    return [q.strip() for q in raw.split(",") if q.strip()] or None


def _resolve_index_views(index_views_arg: str) -> Path:
    index_path = Path(index_views_arg)
    if index_path.exists():
        return index_path

    candidates = [
        Path("results/01_make_views_output/index_views.json"),
        Path("results_syn/01_make_views_output/index_views.json"),
        Path("results_test/01_make_views_output_outdoor/index_views.json"),
        Path("results_testv2/01_make_views_output_outdoor/index_views.json"),
        Path("results_outdoor/01_make_views_output_outdoor/index_views.json"),
        Path("results_panox/01_make_views_output_outdoor/index_views.json")
    ]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        print(f"[warn] index_views not found: {index_path}. Using {existing[0]}")
        return existing[0]
    if existing:
        existing_str = ", ".join(str(p) for p in existing)
        raise SystemExit(f"index_views not found: {index_path}. Multiple candidates exist: {existing_str}")

    raise SystemExit(f"index_views not found: {index_path}")


def _iter_views_from_scan(
    scan_json: str,
    views_root: str,
    skip_missing: bool,
    shard_id: int,
    shard_count: int,
) -> list[str]:
    scan = load_json(scan_json)
    items = scan.get("viewpoints", []) if isinstance(scan, dict) else []
    if not items:
        raise SystemExit(f"scan_json has no viewpoints: {scan_json}")
    root = Path(views_root)
    views_jsons: list[str] = []
    for idx, item in enumerate(items):
        if shard_count > 1 and (idx % shard_count) != shard_id:
            continue
        scene_id = item.get("scene_id", "")
        viewpoint_id = item.get("viewpoint_id", "")
        if not scene_id or not viewpoint_id:
            continue
        vj = root / scene_id / viewpoint_id / "views.json"
        if vj.exists():
            views_jsons.append(str(vj))
        elif not skip_missing:
            raise SystemExit(f"views.json not found yet: {vj}")
    return views_jsons


def _flush_batch(detector, items: list[tuple[str, str]], queries) -> None:
    image_paths = [it[0] for it in items]
    out_jsons = [it[1] for it in items]
    detector.detect_batch(image_paths, out_jsons, queries=queries)


def _run_on_views_json(detector, views_json: str, out_dir: Path, queries, overwrite: bool, batch_size: int) -> None:
    views = load_json(views_json)
    scene_id = views["scene_id"]
    viewpoint_id = views["viewpoint_id"]
    vp_out = out_dir / scene_id / viewpoint_id
    ensure_dir(vp_out)

    det_dir = vp_out / "detections"
    ensure_dir(det_dir)

    batch_items: list[tuple[str, str]] = []
    for v in views["views"]:
        view_id = v["view_id"]
        out_json = det_dir / f"{view_id}.json"
        if out_json.exists() and not overwrite:
            continue
        batch_items.append((v["image_path"], str(out_json)))
        if len(batch_items) >= batch_size:
            _flush_batch(detector, batch_items, queries)
            batch_items = []

    if batch_items:
        _flush_batch(detector, batch_items, queries)

    dump_json(vp_out / "detections_index.json", {"views_json": views_json, "det_dir": str(det_dir)})
    print(f"[{scene_id}/{viewpoint_id}] detections -> {det_dir}")


def _run_on_single_image(detector, image_path: str, out_json: Path, queries) -> None:
    ensure_dir(out_json.parent)
    detector.detect_view(image_path, str(out_json), queries=queries)
    print(f"detections -> {out_json}")


def _run_worker(local_rank: int, args: argparse.Namespace) -> None:
    if args.num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(0)
        args.shard_id = local_rank
        args.shard_count = args.num_gpus

    cfg = load_cfg(args.cfg)
    detector = build_detector(cfg)
    queries = _parse_queries(args.queries)

    if args.image:
        if not args.out_json:
            raise SystemExit("--out_json is required when using --image")
        _run_on_single_image(detector, args.image, Path(args.out_json), queries)
        return

    if not args.out_dir:
        raise SystemExit("--out_dir is required when using --index_views/--views_json/--scan_json")

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    if args.views_json:
        _run_on_views_json(detector, args.views_json, out_root, queries, args.overwrite, args.batch_size)
        return

    if args.scan_json:
        if not args.views_root:
            raise SystemExit("--views_root is required when using --scan_json")
        views_jsons = _iter_views_from_scan(
            args.scan_json,
            args.views_root,
            args.skip_missing_views,
            int(args.shard_id),
            int(args.shard_count),
        )
        for views_json in views_jsons:
            _run_on_views_json(detector, views_json, out_root, queries, args.overwrite, args.batch_size)
        return

    index_views_path = _resolve_index_views(args.index_views)
    index = load_json(index_views_path)
    items = index["items"]
    if args.shard_count > 1:
        if args.shard_id < 0 or args.shard_id >= args.shard_count:
            raise SystemExit("--shard_id must be in [0, shard_count)")
        items = [it for i, it in enumerate(items) if (i % args.shard_count) == args.shard_id]

    for item in items:
        _run_on_views_json(detector, item["views_json"], out_root, queries, args.overwrite, args.batch_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="configs/default.json")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--index_views", help="index_views.json from 01_make_views.py")
    src_group.add_argument("--views_json", help="views.json from 01_make_views.py")
    src_group.add_argument("--image", help="Single image path")
    src_group.add_argument("--scan_json", help="00_scan_output.json to discover views.json under --views_root")
    ap.add_argument("--out_dir", help="Output root (required for index_views/views_json)")
    ap.add_argument("--out_json", help="Output json (required for --image)")
    ap.add_argument("--queries", default="", help="Optional comma-separated open-vocab queries")
    ap.add_argument("--views_root", default="", help="Root directory containing scene/viewpoint/views.json (required for --scan_json)")
    ap.add_argument("--skip_missing_views", action="store_true", help="Skip views without views.json when using --scan_json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size per process for detector backend")
    ap.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use via multiprocessing")
    ap.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-process inference")
    ap.add_argument("--shard_count", type=int, default=1, help="Total shard count for multi-process inference")
    args = ap.parse_args()

    if args.num_gpus > 1:
        if args.image:
            raise SystemExit("--num_gpus > 1 is not supported with --image")
        mp.spawn(_run_worker, nprocs=args.num_gpus, args=(args,))
    else:
        _run_worker(0, args)


if __name__ == "__main__":
    main()
