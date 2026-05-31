#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DAP_ROOT = SCRIPT_DIR / "DAP"
sys.path.append(str(DAP_ROOT))

from networks.models import make  # noqa: E402


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def configure_runtime(device: str) -> None:
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def load_model(config_path: Path, weights_path: Path, device: str):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    state = torch.load(weights_path, map_location=device)
    model = make(config["model"]).to(device)
    model_state = model.state_dict()
    cleaned_state = {}
    for key, value in state.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        if cleaned_key in model_state:
            cleaned_state[cleaned_key] = value
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    return model, config


def colorize_depth_metric(depth_m: np.ndarray, vis_range: str = "100m", cmap: str = "Spectral") -> tuple[np.ndarray, np.ndarray]:
    if vis_range == "100m":
        depth_gray = (np.clip(depth_m, 0.0, 100.0) / 100.0 * 255.0).astype(np.uint8)
    elif vis_range == "10m":
        depth_gray = (np.clip(depth_m, 0.0, 10.0) / 10.0 * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown vis_range: {vis_range}")

    disp = depth_gray.astype(np.float32) / 255.0
    depth_color = matplotlib.colormaps[cmap](disp)[..., :3]
    depth_color = (depth_color * 255).astype(np.uint8)
    return depth_gray, np.ascontiguousarray(depth_color)


def build_output_stem(image_path: Path, input_root: Path | None) -> Path:
    if input_root is not None:
        try:
            return image_path.resolve().relative_to(input_root.resolve()).with_suffix("")
        except ValueError:
            pass

    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    return Path(f"{image_path.stem}_{digest}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_metric_depth(metric_depth: np.ndarray, metric_path: Path, output_format: str, png_depth_scale: float) -> None:
    ensure_parent(metric_path)
    if output_format == "npy":
        np.save(metric_path, metric_depth.astype(np.float32))
        return

    if output_format != "png16":
        raise ValueError(f"Unsupported output_format: {output_format}")

    encoded = np.clip(np.rint(metric_depth * png_depth_scale), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(metric_path)
    scale_path = metric_path.parent / "depth_scale.txt"
    scale_path.write_text(f"{png_depth_scale:g}", encoding="utf-8")


def load_jobs(input_list: str | None, job_file: str | None) -> list[dict]:
    if job_file:
        jobs = []
        with Path(job_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if "image_path" not in job:
                    raise ValueError(f"Missing image_path in job: {job}")
                jobs.append(job)
        return jobs

    assert input_list is not None
    with Path(input_list).open("r", encoding="utf-8") as handle:
        return [{"image_path": line.strip()} for line in handle if line.strip()]


def load_batch(image_paths: list[Path], input_size: tuple[int, int] | None):
    batch = []
    metas = []
    skipped = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            skipped.append(
                {
                    "image_path": str(image_path),
                    "status": "skipped_unreadable_image",
                    "reason": "cv2.imread returned None",
                }
            )
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        if input_size is not None:
            resized = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_CUBIC)
        else:
            resized = image_rgb
        batch.append(resized.astype(np.float32) / 255.0)
        metas.append({"orig_size": (orig_w, orig_h), "image_path": image_path})

    if not batch:
        return None, [], skipped

    stacked = np.stack([img.transpose(2, 0, 1) for img in batch], axis=0)
    return stacked, metas, skipped


def run_batch(model, device: str, image_paths: list[Path], input_size: tuple[int, int] | None, amp: bool):
    batch, metas, skipped = load_batch(image_paths, input_size)
    if batch is None:
        return [], skipped
    tensor = torch.from_numpy(batch).to(device)

    autocast_enabled = amp and device == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(tensor)
            pred_depth = outputs["pred_depth"].detach().cpu().numpy().astype(np.float32)
            pred_mask = outputs["pred_mask"].detach().cpu().numpy().astype(np.float32)

    records = []
    for idx, meta in enumerate(metas):
        image_path = meta["image_path"]
        depth = pred_depth[idx].squeeze()
        raw_mask = pred_mask[idx].squeeze()
        orig_w, orig_h = meta["orig_size"]
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if raw_mask.shape != (orig_h, orig_w):
            raw_mask = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        model_valid = (1.0 - raw_mask) > 0.5
        records.append((image_path, depth, model_valid))
    return records, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-list")
    input_group.add_argument("--job-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--config", default=str(DAP_ROOT / "config" / "infer.yaml"))
    parser.add_argument("--weights", default=str(SCRIPT_DIR / "models" / "hf_model" / "model.pth"))
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metric-scale", type=float, default=100.0)
    parser.add_argument("--output-format", default="npy", choices=["npy", "png16"])
    parser.add_argument("--png-depth-scale", type=float, default=None)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--vis-range", default="100m", choices=["100m", "10m"])
    parser.add_argument("--cmap", default="Spectral")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-name", default="worker")
    args = parser.parse_args()

    os.chdir(DAP_ROOT)
    device = resolve_device(args.device)
    configure_runtime(device)
    png_depth_scale = float(args.metric_scale) if args.png_depth_scale is None else float(args.png_depth_scale)

    input_root = Path(args.input_root).resolve() if args.input_root else None
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()

    jobs = load_jobs(args.input_list, args.job_file)
    image_paths = [Path(job["image_path"]).resolve() for job in jobs]

    model, config = load_model(config_path, weights_path, device)
    input_cfg = config.get("input", {})
    input_size = None
    if input_cfg.get("width") and input_cfg.get("height"):
        input_size = (int(input_cfg["width"]), int(input_cfg["height"]))

    manifest_path = output_dir / "manifests" / f"{args.worker_name}.jsonl"
    ensure_parent(manifest_path)
    manifest_handle = manifest_path.open("w", encoding="utf-8")

    try:
        for start in tqdm(range(0, len(jobs), args.batch_size), desc=f"{args.worker_name}@{device}"):
            batch_jobs = jobs[start : start + args.batch_size]
            batch_paths = [Path(job["image_path"]).resolve() for job in batch_jobs]
            records, skipped = run_batch(model, device, batch_paths, input_size, args.amp)
            skipped_by_image = {item["image_path"]: item for item in skipped}

            for job in batch_jobs:
                image_path_str = str(Path(job["image_path"]).resolve())
                if image_path_str not in skipped_by_image:
                    continue
                skipped_record = {
                    "json_index": job.get("json_index"),
                    "input_image": image_path_str,
                    "metric_depth_npy": str(Path(job["metric_depth_path"]).resolve()) if job.get("metric_depth_path") else None,
                    "metric_depth_format": args.output_format,
                    "png_depth_scale": png_depth_scale if args.output_format == "png16" else None,
                    "status": skipped_by_image[image_path_str]["status"],
                    "reason": skipped_by_image[image_path_str]["reason"],
                }
                manifest_handle.write(json.dumps(skipped_record, ensure_ascii=False) + "\n")
                manifest_handle.flush()

            record_jobs = {str(image_path): job for job, (image_path, _, _) in zip(
                [job for job in batch_jobs if str(Path(job["image_path"]).resolve()) not in skipped_by_image],
                records,
            )}

            for image_path, pred_depth_raw, model_valid in records:
                job = record_jobs[str(image_path)]
                output_stem = build_output_stem(image_path, input_root)
                metric_depth = pred_depth_raw * float(args.metric_scale)

                metric_suffix = ".png" if args.output_format == "png16" else ".npy"
                metric_path = Path(job["metric_depth_path"]).resolve() if job.get("metric_depth_path") else output_dir / "depth_npy_metric" / output_stem.with_suffix(metric_suffix)
                skipped_existing = metric_path.exists() and not args.overwrite
                if not skipped_existing:
                    save_metric_depth(metric_depth, metric_path, args.output_format, png_depth_scale)

                raw_path = Path(job["raw_depth_path"]).resolve() if job.get("raw_depth_path") else output_dir / "depth_npy_raw" / output_stem.with_suffix(".npy")
                if args.save_raw:
                    ensure_parent(raw_path)
                    if not skipped_existing or args.overwrite:
                        np.save(raw_path, pred_depth_raw)

                mask_path = Path(job["mask_path"]).resolve() if job.get("mask_path") else output_dir / "pred_mask_valid" / output_stem.with_suffix(".png")
                if args.save_mask:
                    ensure_parent(mask_path)
                    if not skipped_existing or args.overwrite:
                        Image.fromarray((model_valid.astype(np.uint8) * 255)).save(mask_path)

                gray_path = Path(job["vis_gray_path"]).resolve() if job.get("vis_gray_path") else output_dir / f"depth_vis_gray_{args.vis_range}" / output_stem.with_suffix(".png")
                color_path = Path(job["vis_color_path"]).resolve() if job.get("vis_color_path") else output_dir / f"depth_vis_color_{args.vis_range}" / output_stem.with_suffix(".png")
                if args.save_vis:
                    depth_gray, depth_color = colorize_depth_metric(metric_depth, vis_range=args.vis_range, cmap=args.cmap)
                    ensure_parent(gray_path)
                    ensure_parent(color_path)
                    if not skipped_existing or args.overwrite:
                        cv2.imwrite(str(gray_path), depth_gray)
                        cv2.imwrite(str(color_path), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))

                record = {
                    "json_index": job.get("json_index"),
                    "input_image": str(image_path),
                    "metric_depth_npy": str(metric_path),
                    "metric_depth_format": args.output_format,
                    "png_depth_scale": png_depth_scale if args.output_format == "png16" else None,
                    "raw_depth_npy": str(raw_path) if args.save_raw else None,
                    "mask_path": str(mask_path) if args.save_mask else None,
                    "vis_gray_path": str(gray_path) if args.save_vis else None,
                    "vis_color_path": str(color_path) if args.save_vis else None,
                    "skipped_existing": skipped_existing,
                    "metric_scale": float(args.metric_scale),
                    "model_valid_ratio": float(model_valid.mean()),
                    "pred_raw_min": float(pred_depth_raw.min()),
                    "pred_raw_median": float(np.median(pred_depth_raw)),
                    "pred_raw_max": float(pred_depth_raw.max()),
                    "pred_metric_median": float(np.median(metric_depth)),
                }
                manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_handle.flush()
    finally:
        manifest_handle.close()


if __name__ == "__main__":
    main()
