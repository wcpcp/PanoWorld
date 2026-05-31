#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_meta.io_utils import dump_json, ensure_dir, load_json


DEFAULT_MODES = ["caption_dense", "caption_brief", "referring_full", "hybrid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--entities_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--views_json", default="")
    ap.add_argument("--det_root", default="")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--min_reground_iou", type=float, default=0.0)
    ap.add_argument("--ref_batch_size", type=int, default=8)
    ap.add_argument("--ref_workers", type=int, default=1)
    ap.add_argument("--drop_failed", action="store_true")
    ap.add_argument("--viz_root", default="")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    input_obj = load_json(args.entities_json)
    input_entities = input_obj.get("entities", [])
    effective_input_count = min(len(input_entities), int(args.max_entities)) if args.max_entities else len(input_entities)

    script_path = Path(__file__).with_name("05b_local_reground.py")
    summary_rows: list[dict[str, Any]] = []
    generated_outputs: dict[str, str] = {}

    for mode in args.modes:
        mode_out = out_dir / f"reground_{mode}.json"
        cmd = [
            sys.executable,
            str(script_path),
            "--cfg",
            args.cfg,
            "--entities_json",
            args.entities_json,
            "--out_json",
            str(mode_out),
            "--query_mode",
            str(mode),
            "--min_reground_iou",
            str(float(args.min_reground_iou)),
            "--ref_batch_size",
            str(max(1, int(args.ref_batch_size))),
            "--ref_workers",
            str(max(1, int(args.ref_workers))),
        ]
        if args.views_json:
            cmd.extend(["--views_json", args.views_json])
        if args.det_root:
            cmd.extend(["--det_root", args.det_root])
        if args.max_entities:
            cmd.extend(["--max_entities", str(int(args.max_entities))])
        if args.drop_failed:
            cmd.append("--drop_failed")
        if args.viz_root:
            cmd.extend(["--viz_dir", str(Path(args.viz_root) / mode)])

        subprocess.run(cmd, check=True)
        generated_outputs[mode] = str(mode_out)
        result_obj = load_json(mode_out)
        summary_rows.append(_summarize_mode(mode, result_obj, input_entities_count=effective_input_count))

    summary = {
        "entities_json": args.entities_json,
        "views_json": args.views_json or input_obj.get("views_json", ""),
        "det_root": args.det_root,
        "min_reground_iou": float(args.min_reground_iou),
        "drop_failed": bool(args.drop_failed),
        "modes": list(args.modes),
        "outputs": generated_outputs,
        "summary": summary_rows,
    }
    dump_json(str(out_dir / "query_mode_summary.json"), summary)
    print(_format_table(summary_rows))


def _summarize_mode(mode: str, obj: dict[str, Any], *, input_entities_count: int) -> dict[str, Any]:
    entities = obj.get("entities", [])
    regrounds = [entity.get("local_reground", {}) for entity in entities]
    consistency_ious = [float(item.get("consistency_iou", 0.0)) for item in regrounds]
    yellow_hits = [item for item in regrounds if len(item.get("proposal_consistency", {}).get("reference_bbox_xyxy", [])) == 4]
    yellow_from_04c = [
        item for item in regrounds if item.get("proposal_consistency", {}).get("reference_source", "") == "04c_filtered_view"
    ]
    yellow_from_02 = [
        item for item in regrounds if item.get("proposal_consistency", {}).get("reference_source", "") == "02_detect_fallback"
    ]
    passed_count = sum(1 for item in regrounds if bool(item.get("passed", False)))
    no_view_count = sum(1 for item in regrounds if item.get("status", "") == "no_view")
    empty_projection_count = sum(1 for item in regrounds if item.get("status", "") == "empty_projection")
    no_prediction_count = sum(1 for item in regrounds if item.get("status", "") == "no_prediction")
    request_count = int(obj.get("quality_stats", {}).get("local_reground_request_count", 0))
    dropped_count = max(0, int(input_entities_count) - int(len(entities)))
    failed_count = max(0, int(input_entities_count) - int(passed_count))
    yellow_rate_on_requests = float(len(yellow_hits)) / float(max(request_count, 1))
    return {
        "mode": mode,
        "input_entities": int(input_entities_count),
        "output_entities": int(len(entities)),
        "request_count": int(request_count),
        "passed_count": int(passed_count),
        "failed_or_unverified_count": int(failed_count),
        "dropped_count": int(dropped_count),
        "yellow_box_count": int(len(yellow_hits)),
        "yellow_box_rate_on_requests": float(yellow_rate_on_requests),
        "yellow_from_04c_count": int(len(yellow_from_04c)),
        "yellow_from_02_count": int(len(yellow_from_02)),
        "no_view_count": int(no_view_count),
        "empty_projection_count": int(empty_projection_count),
        "no_prediction_count": int(no_prediction_count),
        "consistency_iou_mean": float(statistics.fmean(consistency_ious)) if consistency_ious else 0.0,
        "consistency_iou_median": float(statistics.median(consistency_ious)) if consistency_ious else 0.0,
        "consistency_iou_min": float(min(consistency_ious)) if consistency_ious else 0.0,
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "mode",
        "consistency_iou_mean",
        "consistency_iou_median",
        "yellow_box_rate_on_requests",
        "yellow_from_04c_count",
        "yellow_from_02_count",
        "passed_count",
        "failed_or_unverified_count",
        "dropped_count",
    ]
    display_rows = []
    for row in rows:
        display_rows.append(
            [
                str(row.get("mode", "")),
                f"{float(row.get('consistency_iou_mean', 0.0)):.4f}",
                f"{float(row.get('consistency_iou_median', 0.0)):.4f}",
                f"{float(row.get('yellow_box_rate_on_requests', 0.0)):.4f}",
                str(int(row.get("yellow_from_04c_count", 0))),
                str(int(row.get("yellow_from_02_count", 0))),
                str(int(row.get("passed_count", 0))),
                str(int(row.get("failed_or_unverified_count", 0))),
                str(int(row.get("dropped_count", 0))),
            ]
        )
    widths = [len(header) for header in headers]
    for row in display_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    body = [" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) for row in display_rows]
    return "\n".join([line, sep, *body])


if __name__ == "__main__":
    main()