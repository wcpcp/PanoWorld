#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--queries", default="")
    ap.add_argument("--det_only", action="store_true", help="Skip segmentation and use detection-only ERP alignment.")
    ap.add_argument("--run_semantic", action="store_true")
    ap.add_argument("--run_relations", action="store_true")
    ap.add_argument("--run_export", action="store_true")
    ap.add_argument("--semantic_skip_reground", action="store_true")
    ap.add_argument("--semantic_max_entities", type=int, default=0)
    ap.add_argument("--min_view_consistency", type=float, default=0.0)
    ap.add_argument("--min_support_views", type=int, default=1)
    ap.add_argument("--min_entity_score", type=float, default=0.0)
    ap.add_argument("--skip_instance_vote", action="store_true")
    ap.add_argument("--instance_vote_iou", type=float, default=0.15)
    ap.add_argument("--instance_vote_dist", type=float, default=0.35)
    ap.add_argument("--instance_vote_sem", type=float, default=0.2)
    ap.add_argument("--instance_overlap_ratio", type=float, default=0.05)
    ap.add_argument("--min_instance_vote_score", type=float, default=0.45)
    ap.add_argument("--min_instance_support", type=int, default=1)
    ap.add_argument("--viz_entities", action="store_true")
    ap.add_argument("--viz_overlap", action="store_true")

    # view sampling controls (passed through to 01_make_views.py)
    ap.add_argument(
        "--view_mode",
        default="persp8",
        choices=["persp4", "persp4+tb", "persp8", "persp8+pair", "ring+cubemap", "ring", "cubemap"],
        help="View sampling strategy.",
    )
    ap.add_argument("--persp_fov", type=float, default=90.0, help="Perspective FOV in degrees")
    ap.add_argument("--persp_size", type=int, default=0, help="Perspective view size (square). 0 => ERP height")
    ap.add_argument("--persp_n", type=int, default=0, help="Number of yaw faces (0 => infer from mode)")
    ap.add_argument("--persp_pair_adjacent", action="store_true")
    ap.add_argument("--persp_add_tb", action="store_true")
    ap.add_argument("--disable_pano_mask", action="store_true")

    # overlap verification
    ap.add_argument("--skip_overlap_verify", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)

    scan_json = root / "scan.json"

    # 0) scan
    import subprocess

    subprocess.run(
        [
            "python",
            "scripts/00_scan_realsee.py",
            "--root",
            cfg["dataset"]["root"],
            "--out",
            str(scan_json),
            "--limit",
            str(args.limit),
        ],
        check=True,
    )

    # 1) views
    views_cmd = [
        "python",
        "scripts/01_make_views.py",
        "--scan_json",
        str(scan_json),
        "--out_dir",
        str(root / "views"),
        "--limit",
        str(args.limit),
        "--mode",
        args.view_mode,
        "--persp_fov",
        str(args.persp_fov),
        "--persp_size",
        str(args.persp_size),
    ]
    if args.persp_n:
        views_cmd += ["--persp_n", str(args.persp_n)]
    if args.persp_pair_adjacent:
        views_cmd += ["--persp_pair_adjacent"]
    if args.persp_add_tb:
        views_cmd += ["--persp_add_tb"]
    if args.disable_pano_mask:
        views_cmd += ["--disable_pano_mask"]
    if args.view_mode in ("ring+cubemap", "ring"):
        views_cmd += ["--ring_include_seam"]

    subprocess.run(views_cmd, check=True)

    # 2) detect
    subprocess.run(
        [
            "python",
            "scripts/02_detect.py",
            "--cfg",
            args.cfg,
            "--index_views",
            str(root / "views" / "index_views.json"),
            "--out_dir",
            str(root / "det"),
            "--queries",
            args.queries,
        ],
        check=True,
    )

    if not args.det_only:
        subprocess.run(
            [
                "python",
                "scripts/03_segment.py",
                "--cfg",
                args.cfg,
                "--index_views",
                str(root / "views" / "index_views.json"),
                "--det_root",
                str(root / "det"),
                "--out_dir",
                str(root / "seg"),
            ],
            check=True,
        )

    # 3b) overlap verify (optional)
    if not args.skip_overlap_verify:
        overlap_cmd = [
            "python",
            "scripts/04b_overlap_verify.py",
            "--cfg",
            args.cfg,
            "--index_views",
            str(root / "views" / "index_views.json"),
            "--out_dir",
            str(root / "overlap"),
        ]
        if args.det_only:
            overlap_cmd += ["--det_root", str(root / "det")]
        else:
            overlap_cmd += ["--seg_root", str(root / "seg")]
        if args.viz_overlap:
            overlap_cmd += ["--viz_dir", str(root / "overlap_viz")]
        subprocess.run(
            overlap_cmd,
            check=True,
        )

    run_instance_vote = args.det_only and not args.skip_instance_vote
    if run_instance_vote:
        subprocess.run(
            [
                "python",
                "scripts/04c_instance_vote.py",
                "--cfg",
                args.cfg,
                "--index_views",
                str(root / "views" / "index_views.json"),
                "--det_root",
                str(root / "det"),
                "--out_dir",
                str(root / "instance_vote"),
                "--iou_thr",
                str(args.instance_vote_iou),
                "--dist_thr",
                str(args.instance_vote_dist),
                "--sem_thr",
                str(args.instance_vote_sem),
                "--min_overlap_ratio",
                str(args.instance_overlap_ratio),
                "--min_vote_score",
                str(args.min_instance_vote_score),
                "--min_support_matches",
                str(args.min_instance_support),
            ],
            check=True,
        )

    # 4) merge entities
    merge_cmd = [
        "python",
        "scripts/04_merge_entities.py",
        "--cfg",
        args.cfg,
        "--index_views",
        str(root / "views" / "index_views.json"),
        "--out_dir",
        str(root / "merged"),
        "--min_view_consistency",
        str(args.min_view_consistency),
        "--min_support_views",
        str(args.min_support_views),
        "--min_entity_score",
        str(args.min_entity_score),
    ]
    if args.det_only:
        merge_cmd += ["--det_root", str(root / "det")]
    else:
        merge_cmd += ["--seg_root", str(root / "seg")]
    if run_instance_vote:
        merge_cmd += ["--instance_vote_root", str(root / "instance_vote")]
    if not args.skip_overlap_verify:
        merge_cmd += ["--overlap_verify_root", str(root / "overlap")]
    if args.viz_entities:
        merge_cmd += ["--viz_dir", str(root / "merged_viz")]
    subprocess.run(
        merge_cmd,
        check=True,
    )

    if args.run_semantic or args.run_relations or args.run_export:
        views_index_path = root / "views" / "index_views.json"
        import json

        views_index = json.loads(views_index_path.read_text(encoding="utf-8"))
        for item in views_index["items"]:
            views_json = json.loads(Path(item["views_json"]).read_text(encoding="utf-8"))
            scene_id = views_json["scene_id"]
            viewpoint_id = views_json["viewpoint_id"]
            entities_json = root / "merged" / scene_id / viewpoint_id / "entities.json"
            enriched_json = root / "semantic" / scene_id / viewpoint_id / "entities_enriched.json"
            relations_json = root / "relations" / scene_id / viewpoint_id / "relations.json"
            final_json = root / "metadata" / scene_id / viewpoint_id / "metadata.json"

            if args.run_semantic:
                cmd = [
                    "python",
                    "scripts/05_semantic_enrich.py",
                    "--cfg",
                    args.cfg,
                    "--entities_json",
                    str(entities_json),
                    "--out_json",
                    str(enriched_json),
                ]
                if args.semantic_skip_reground:
                    cmd += ["--skip_reground"]
                if args.semantic_max_entities:
                    cmd += ["--max_entities", str(args.semantic_max_entities)]
                subprocess.run(cmd, check=True)

            src_entities = enriched_json if args.run_semantic else entities_json
            if args.run_relations:
                subprocess.run(
                    [
                        "python",
                        "scripts/06_build_relations.py",
                        "--cfg",
                        args.cfg,
                        "--entities_json",
                        str(src_entities),
                        "--out_json",
                        str(relations_json),
                    ],
                    check=True,
                )

            if args.run_export:
                rel_src = relations_json if args.run_relations else root / "relations" / scene_id / viewpoint_id / "relations.json"
                if not rel_src.exists():
                    rel_src.parent.mkdir(parents=True, exist_ok=True)
                    rel_src.write_text('{"relations": []}', encoding='utf-8')
                subprocess.run(
                    [
                        "python",
                        "scripts/07_export_metadata.py",
                        "--cfg",
                        args.cfg,
                        "--entities_json",
                        str(src_entities),
                        "--relations_json",
                        str(rel_src),
                        "--out_json",
                        str(final_json),
                    ],
                    check=True,
                )

    print(f"Done. See: {root}")


if __name__ == "__main__":
    main()
