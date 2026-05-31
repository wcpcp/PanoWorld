#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import multiprocessing as mp
from pathlib import Path

# Allow running as a script without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.view_sampling import make_cubemap_views, make_persp4_views, make_persp_views, make_ring_views


def _resolve_scan_json(scan_json_arg: str) -> Path:
    scan_path = Path(scan_json_arg)
    if scan_path.exists():
        return scan_path

    candidates = [
        Path("results/00_scan_output.json"),
        Path("results_syn/00_scan_output.json"),
        Path("results_test/00_scan_output_outdoor.json"),
        Path("results_outdoor/00_scan_output_outdoor.json"),
    ]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        print(f"[warn] scan_json not found: {scan_path}. Using {existing[0]}")
        return existing[0]
    if existing:
        existing_str = ", ".join(str(p) for p in existing)
        raise SystemExit(f"scan_json not found: {scan_path}. Multiple candidates exist: {existing_str}")

    raise SystemExit(f"scan_json not found: {scan_path}")


def _process_one_view(task: tuple[dict, dict]) -> dict | None:
    item, args_dict = task

    scene_id = item["scene_id"]
    viewpoint_id = item["viewpoint_id"]
    erp_path = item["pano_path"]
    pano_mask_path = item.get("pano_mask_path")

    # 检查图像是否损坏
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(erp_path) as img:
            img.verify()
    except Exception as e:
        print(f"[损坏] 跳过: {erp_path} ({e})")
        return None

    out_root = Path(args_dict["out_dir"])
    vp_out = out_root / scene_id / viewpoint_id
    views_dir = vp_out / "views"
    if args_dict["skip_existing"] and views_dir.exists():
        print(f"[{scene_id}/{viewpoint_id}] views_dir already exists, skipping...")
        return {
            "scene_id": scene_id,
            "viewpoint_id": viewpoint_id,
            "erp_path": erp_path,
            "views_json": str(vp_out / "views.json"),
        }
    ensure_dir(views_dir)

    views = []
    mode = args_dict["mode"]
    if mode in ("persp4", "persp4+tb"):
        views.extend(
            make_persp4_views(
                erp_path,
                str(views_dir / "persp"),
                pano_mask_path=None if args_dict["disable_pano_mask"] else pano_mask_path,
                out_size=args_dict["persp_size"],
                fov_deg=args_dict["persp_fov"],
                add_top_bottom=(mode == "persp4+tb") or bool(args_dict["persp_add_tb"]),
            )
        )
    elif mode in ("persp6", "persp6+pair", "persp8", "persp8+pair"):
        default_n = 6 if mode in ("persp6", "persp6+pair") else 8
        n_yaw = args_dict["persp_n"] if args_dict["persp_n"] > 0 else default_n
        pair = (mode in ("persp6+pair", "persp8+pair")) or bool(args_dict["persp_pair_adjacent"])
        views.extend(
            make_persp_views(
                erp_path=erp_path,
                out_dir=str(views_dir / "persp"),
                n_yaw=n_yaw,
                pano_mask_path=None if args_dict["disable_pano_mask"] else pano_mask_path,
                out_size=args_dict["persp_size"],
                fov_deg=args_dict["persp_fov"],
                add_top_bottom=bool(args_dict["persp_add_tb"]),
                pair_adjacent=pair,
            )
        )
    else:
        if mode in ("ring+cubemap", "ring") and not args_dict["no_ring"]:
            views.extend(
                make_ring_views(
                    erp_path,
                    str(views_dir / "ring"),
                    tile_w=args_dict["ring_tile_w"],
                    overlap=args_dict["ring_overlap"],
                    include_seam=args_dict["ring_include_seam"],
                )
            )
        if mode in ("ring+cubemap", "cubemap") and not args_dict["no_cubemap"]:
            views.extend(make_cubemap_views(erp_path, str(views_dir / "cubemap"), face_size=args_dict["cubemap_size"]))

    views_json = vp_out / "views.json"
    dump_json(
        views_json,
        {
            "scene_id": scene_id,
            "viewpoint_id": viewpoint_id,
            "erp_path": erp_path,
            "pano_mask_path": pano_mask_path,
            "mode": mode,
            "views": [v.__dict__ for v in views],
        },
    )
    print(f"[{scene_id}/{viewpoint_id}] views={len(views)}")
    return {"scene_id": scene_id, "viewpoint_id": viewpoint_id, "erp_path": erp_path, "views_json": str(views_json)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_json", default="results/00_scan_output.json", help="Output of 00_scan_realsee.py")
    ap.add_argument("--out_dir", default="results/01_make_views_output", help="Output root dir")
    ap.add_argument(
        "--mode",
        default="persp6",
        choices=["persp4", "persp4+tb", "persp6", "persp6+pair", "persp8", "persp8+pair", "ring+cubemap", "ring", "cubemap"],
        help="View sampling strategy.",
    )

    ap.add_argument("--persp_size", type=int, default=0, help="Perspective view size (square). 0 => ERP height")
    ap.add_argument("--persp_fov", type=float, default=120.0, help="Perspective FOV in degrees; >90 gives overlap. Default `persp6 + 120` gives ~60° overlap for adjacent views.")
    ap.add_argument("--persp_n", type=int, default=0, help="Number of yaw faces (0 => infer from mode)")
    ap.add_argument("--persp_pair_adjacent", action="store_true", help="Also generate stitched adjacent-pair views")
    ap.add_argument("--persp_add_tb", action="store_true", help="Also generate top/bottom views when mask supports it")
    ap.add_argument(
        "--disable_pano_mask",
        action="store_true",
        help="Do not apply pano_mask.png when generating perspective views",
    )

    ap.add_argument("--ring_tile_w", type=int, default=800)
    ap.add_argument("--ring_overlap", type=float, default=0.5)
    ap.add_argument("--ring_include_seam", action="store_true")
    ap.add_argument("--no_ring", action="store_true")
    ap.add_argument("--cubemap_size", type=int, default=768)
    ap.add_argument("--no_cubemap", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start_scene", type=str, default="", help="Skip processing until this scene_id is reached")
    ap.add_argument("--rebuild_index", action="store_true", help="Rebuild index_views.json from existing views.json files")
    ap.add_argument(
        "--rebuild_index_from_scan",
        action="store_true",
        help="When used with --rebuild_index, build index directly from scan_json viewpoints (no view files required)",
    )
    ap.add_argument("--skip_existing", action="store_true", help="Skip viewpoints if views/ already exists")
    ap.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for view generation")
    args = ap.parse_args()

    scan_json_path = _resolve_scan_json(args.scan_json)
    scan = load_json(scan_json_path)
    vps = scan["viewpoints"]
    if args.limit:
        vps = vps[: args.limit]
    if not vps:
        root_info = scan.get("root", "") if isinstance(scan, dict) else ""
        print(f"[warn] 0 viewpoints in {scan_json_path}. root={root_info}")

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    if args.rebuild_index:
        index_items = []
        if args.rebuild_index_from_scan:
            for item in vps:
                scene_id = item.get("scene_id")
                viewpoint_id = item.get("viewpoint_id")
                erp_path = item.get("pano_path")
                if not scene_id or not viewpoint_id or not erp_path:
                    continue
                index_items.append(
                    {
                        "scene_id": scene_id,
                        "viewpoint_id": viewpoint_id,
                        "erp_path": erp_path,
                        "views_json": "",
                    }
                )
        else:
            for views_json in sorted(out_root.glob("scene_*/*/views.json")):
                try:
                    data = load_json(str(views_json))
                except Exception:
                    continue
                scene_id = data.get("scene_id") or views_json.parent.parent.name
                viewpoint_id = data.get("viewpoint_id") or views_json.parent.name
                erp_path = data.get("erp_path")
                if not erp_path:
                    continue
                index_items.append(
                    {
                        "scene_id": scene_id,
                        "viewpoint_id": viewpoint_id,
                        "erp_path": erp_path,
                        "views_json": str(views_json),
                    }
                )

        dump_json(out_root / "index_views.json", {"scan_json": args.scan_json, "items": index_items})
        print(f"Done -> {out_root / 'index_views.json'} (items={len(index_items)})")
        return

    index = []
    started = False if args.start_scene else True
    tasks = []
    for item in vps:
        scene_id = item["scene_id"]
        if not started:
            if scene_id == args.start_scene:
                started = True
            else:
                continue
        tasks.append(item)

    args_dict = vars(args).copy()
    if int(args.num_workers) > 1 and len(tasks) > 1:
        with mp.Pool(processes=int(args.num_workers)) as pool:
            for result in pool.imap_unordered(
                _process_one_view, [(item, args_dict) for item in tasks]
            ):
                if result:
                    index.append(result)
    else:
        for item in tasks:
            result = _process_one_view(item, args_dict)
            if result:
                index.append(result)

    dump_json(out_root / "index_views.json", {"scan_json": str(scan_json_path), "items": index})
    print(f"Done -> {out_root / 'index_views.json'}")


if __name__ == "__main__":
    main()
