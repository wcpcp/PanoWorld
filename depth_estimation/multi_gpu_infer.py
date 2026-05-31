#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "server_infer_worker.py"


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_input_paths(input_dir: Path | None, input_list: Path | None) -> tuple[list[Path], Path | None]:
    if input_dir is not None:
        images = discover_images(input_dir.resolve())
        return images, input_dir.resolve()
    assert input_list is not None
    with input_list.open("r", encoding="utf-8") as handle:
        images = [Path(line.strip()).resolve() for line in handle if line.strip()]
    return images, None


def split_round_robin(items: list[Path], shard_count: int) -> list[list[Path]]:
    shards = [[] for _ in range(shard_count)]
    for idx, item in enumerate(items):
        shards[idx % shard_count].append(item)
    return shards


def write_shard_list(paths: list[Path], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(str(path) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir")
    group.add_argument("--input-list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "DAP" / "config" / "infer.yaml"))
    parser.add_argument("--weights", default=str(SCRIPT_DIR / "models" / "hf_model" / "model.pth"))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--worker-device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metric-scale", type=float, default=100.0)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--vis-range", default="100m", choices=["100m", "10m"])
    parser.add_argument("--cmap", default="Spectral")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve() if args.input_dir else None
    input_list = Path(args.input_list).resolve() if args.input_list else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, input_root = load_input_paths(input_dir, input_list)
    if not image_paths:
        raise SystemExit("No input images found.")

    gpu_ids = [token.strip() for token in args.gpu_ids.split(",") if token.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU IDs provided.")

    shards = split_round_robin(image_paths, len(gpu_ids))
    procs: list[tuple[str, subprocess.Popen]] = []
    log_handles = []
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for local_worker_idx, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        if not shard:
            continue
        shard_path = output_dir / "shards" / f"gpu{gpu_id}.txt"
        write_shard_list(shard, shard_path)

        log_path = logs_dir / f"gpu{gpu_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        log_handles.append(log_handle)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--input-list",
            str(shard_path),
            "--output-dir",
            str(output_dir),
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
            "--vis-range",
            args.vis_range,
            "--cmap",
            args.cmap,
            "--worker-name",
            f"gpu{gpu_id}",
        ]
        if input_root is not None:
            cmd.extend(["--input-root", str(input_root)])
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
        procs.append((str(gpu_id), proc))

    failed = []
    for gpu_id, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((gpu_id, code))

    if failed:
        messages = ", ".join(f"gpu{gpu_id}: exit {code}" for gpu_id, code in failed)
        for handle in log_handles:
            handle.close()
        raise SystemExit(f"One or more workers failed: {messages}")

    for handle in log_handles:
        handle.close()
    print(f"Completed {len(image_paths)} images across {len(procs)} GPU workers.")
    print(f"Output directory: {output_dir}")
    print(f"Worker logs: {logs_dir}")


if __name__ == "__main__":
    main()
