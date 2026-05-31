#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "server_infer_worker.py"


def extract_records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "records", "data", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("JSON root must be a list, or a dict containing one of: items, records, data, samples.")


def replace_last_part(path: Path, source_name: str, target_name: str) -> tuple[Path, bool]:
    parts = list(path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == source_name:
            parts[idx] = target_name
            return Path(*parts), True
    return path, False


def build_depth_path(image_path: Path, source_dir_name: str, target_dir_name: str, depth_suffix: str) -> tuple[Path, bool]:
    mapped_path, replaced = replace_last_part(image_path, source_dir_name, target_dir_name)
    if not replaced:
        mapped_path = image_path.parent / target_dir_name / image_path.name
    return mapped_path.with_suffix(depth_suffix), replaced


def split_round_robin(items: list[dict], shard_count: int) -> list[list[dict]]:
    shards = [[] for _ in range(shard_count)]
    for idx, item in enumerate(items):
        shards[idx % shard_count].append(item)
    return shards


def write_jsonl(rows: list[dict], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "DAP" / "config" / "infer.yaml"))
    parser.add_argument("--weights", default=str(SCRIPT_DIR / "models" / "hf_model" / "model.pth"))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--worker-device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metric-scale", type=float, default=100.0)
    parser.add_argument("--depth-key", default="depth_path")
    parser.add_argument("--source-dir-name", default="images")
    parser.add_argument("--target-dir-name", default="images_dep")
    parser.add_argument("--output-format", default="png16", choices=["png16", "npy"])
    parser.add_argument("--depth-suffix", default=None)
    parser.add_argument("--png-depth-scale", type=float, default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--backup-json", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--vis-range", default="100m", choices=["100m", "10m"])
    parser.add_argument("--cmap", default="Spectral")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    png_depth_scale = float(args.metric_scale) if args.png_depth_scale is None else float(args.png_depth_scale)

    if args.depth_suffix is None:
        depth_suffix = ".png" if args.output_format == "png16" else ".npy"
    else:
        depth_suffix = args.depth_suffix

    expected_suffix = ".png" if args.output_format == "png16" else ".npy"
    if depth_suffix != expected_suffix:
        raise SystemExit(f"Output format {args.output_format} requires depth suffix {expected_suffix}.")

    json_path = Path(args.json_path).resolve()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = extract_records(payload)
    gpu_ids = [token.strip() for token in args.gpu_ids.split(",") if token.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU IDs provided.")

    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else json_path.parent / f".{json_path.stem}_dap_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    missing_images = []
    fallback_count = 0
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} is not an object.")
        image_value = record.get("image_path")
        if not image_value:
            missing_images.append(idx)
            continue

        image_path = Path(image_value)
        depth_path, replaced = build_depth_path(
            image_path=image_path,
            source_dir_name=args.source_dir_name,
            target_dir_name=args.target_dir_name,
            depth_suffix=depth_suffix,
        )
        record[args.depth_key] = str(depth_path)
        record["depth_scale"] = float(args.metric_scale)
        record["depth_format"] = args.output_format
        if args.output_format == "png16" and png_depth_scale != float(args.metric_scale):
            record["png_depth_scale"] = png_depth_scale

        if not replaced:
            fallback_count += 1

        if depth_path.exists() and not args.overwrite:
            continue

        jobs.append(
            {
                "json_index": idx,
                "image_path": str(image_path),
                "metric_depth_path": str(depth_path),
            }
        )

    if missing_images:
        raise SystemExit(f"Found records without image_path at indices: {missing_images[:10]}")

    if args.backup_json:
        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(json_path, backup_path)

    if jobs:
        shards = split_round_robin(jobs, len(gpu_ids))
        procs: list[tuple[str, subprocess.Popen]] = []
        log_handles = []
        logs_dir = artifacts_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        for gpu_id, shard in zip(gpu_ids, shards):
            if not shard:
                continue
            shard_path = artifacts_dir / "shards" / f"gpu{gpu_id}.jsonl"
            write_jsonl(shard, shard_path)

            log_path = logs_dir / f"gpu{gpu_id}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            log_handles.append(log_handle)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd = [
                sys.executable,
                str(WORKER_SCRIPT),
                "--job-file",
                str(shard_path),
                "--output-dir",
                str(artifacts_dir),
                "--config",
                str(Path(args.config).resolve()),
                "--weights",
                str(Path(args.weights).resolve()),
                "--device",
                args.worker_device,
                "--batch-size",
                str(args.batch_size),
                "--metric-scale",
                str(args.metric_scale),
                "--output-format",
                args.output_format,
                "--png-depth-scale",
                str(png_depth_scale),
                "--vis-range",
                args.vis_range,
                "--cmap",
                args.cmap,
                "--worker-name",
                f"gpu{gpu_id}",
            ]
            if args.save_raw:
                cmd.append("--save-raw")
            if args.save_vis:
                cmd.append("--save-vis")
            if args.save_mask:
                cmd.append("--save-mask")
            if args.amp:
                cmd.append("--amp")
            if args.overwrite:
                cmd.append("--overwrite")

            proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            procs.append((gpu_id, proc))

        failed = []
        for gpu_id, proc in procs:
            code = proc.wait()
            if code != 0:
                failed.append((gpu_id, code))

        for handle in log_handles:
            handle.close()

        if failed:
            messages = ", ".join(f"gpu{gpu_id}: exit {code}" for gpu_id, code in failed)
            raise SystemExit(f"One or more workers failed: {messages}")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Updated JSON: {json_path}")
    print(f"Depth key: {args.depth_key}")
    print(f"Prepared jobs: {len(jobs)}")
    print(f"Artifacts directory: {artifacts_dir}")
    if fallback_count:
        print(
            f"Warning: {fallback_count} paths did not contain '{args.source_dir_name}', "
            f"so outputs were written under each image's local '{args.target_dir_name}' subdirectory."
        )


if __name__ == "__main__":
    main()
