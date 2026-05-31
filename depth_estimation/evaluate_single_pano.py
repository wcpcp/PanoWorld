#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


SCRIPT_DIR = Path(__file__).resolve().parent
DAP_ROOT = SCRIPT_DIR / "DAP"
sys.path.append(str(DAP_ROOT))

from networks.models import make  # noqa: E402


def resolve_device(device_arg: str = "auto") -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    return device_arg


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


def infer_depth(
    model,
    image_path: Path,
    input_size: tuple[int, int] | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    if input_size is not None:
        resized_rgb = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_CUBIC)
    else:
        resized_rgb = image_rgb

    image_f32 = resized_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_f32.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        pred_depth = outputs["pred_depth"][0].detach().cpu().squeeze().numpy().astype(np.float32)
        pred_mask = outputs["pred_mask"][0].detach().cpu().squeeze().numpy().astype(np.float32)

    if pred_depth.shape != (orig_h, orig_w):
        pred_depth = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    model_valid = (1.0 - pred_mask) > 0.5
    return pred_depth, model_valid


def compute_metrics(gt_m: np.ndarray, pred_m: np.ndarray, eval_mask: np.ndarray) -> dict[str, float]:
    gt = gt_m[eval_mask].astype(np.float64)
    pred = pred_m[eval_mask].astype(np.float64)

    positive = (gt > 0) & (pred > 0)
    gt = gt[positive]
    pred = pred[positive]
    if gt.size == 0:
        raise ValueError("No positive valid pixels available for metric computation.")

    diff = pred - gt
    thresh = np.maximum(gt / pred, pred / gt)
    pearson = float(np.corrcoef(gt, pred)[0, 1]) if gt.size > 1 else float("nan")

    metrics = {
        "pixel_count": int(gt.size),
        "gt_mean_m": float(gt.mean()),
        "gt_median_m": float(np.median(gt)),
        "pred_mean_m": float(pred.mean()),
        "pred_median_m": float(np.median(pred)),
        "mae_m": float(np.mean(np.abs(diff))),
        "rmse_m": float(np.sqrt(np.mean(diff**2))),
        "median_abs_error_m": float(np.median(np.abs(diff))),
        "abs_rel": float(np.mean(np.abs(diff) / gt)),
        "log10": float(np.mean(np.abs(np.log10(pred / gt)))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25**2)),
        "delta3": float(np.mean(thresh < 1.25**3)),
        "pearson_r": pearson,
    }
    return metrics


def build_visual_map(values: np.ndarray, valid_mask: np.ndarray, vmin: float, vmax: float, cmap: str = "Spectral") -> np.ndarray:
    arr = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return arr

    clipped = np.clip((values - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    colored = matplotlib.colormaps[cmap](clipped)[..., :3]
    arr[valid_mask] = (colored[valid_mask] * 255).astype(np.uint8)
    return arr


def build_error_map(error_m: np.ndarray, valid_mask: np.ndarray, vmax: float, cmap: str = "magma") -> np.ndarray:
    arr = np.zeros((*error_m.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return arr
    clipped = np.clip(error_m / max(vmax, 1e-8), 0.0, 1.0)
    colored = matplotlib.colormaps[cmap](clipped)[..., :3]
    arr[valid_mask] = (colored[valid_mask] * 255).astype(np.uint8)
    return arr


def save_artifacts(
    output_dir: Path,
    image_path: Path,
    rgb: np.ndarray,
    gt_depth_m: np.ndarray,
    pred_depth_raw: np.ndarray,
    pred_depth_aligned: np.ndarray,
    pred_depth_fixed_scaled: np.ndarray | None,
    gt_valid_mask: np.ndarray,
    eval_mask_with_model: np.ndarray,
    model_valid_mask: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "pred_depth_raw.npy", pred_depth_raw)
    np.save(output_dir / "pred_depth_median_aligned.npy", pred_depth_aligned)
    if pred_depth_fixed_scaled is not None:
        np.save(output_dir / "pred_depth_fixed_scaled.npy", pred_depth_fixed_scaled)
    np.save(output_dir / "gt_depth_m.npy", gt_depth_m)

    Image.fromarray((gt_valid_mask.astype(np.uint8) * 255)).save(output_dir / "gt_valid_mask.png")
    Image.fromarray((model_valid_mask.astype(np.uint8) * 255)).save(output_dir / "pred_model_valid_mask.png")
    Image.fromarray((eval_mask_with_model.astype(np.uint8) * 255)).save(output_dir / "eval_mask_gt_and_model.png")

    valid_for_vis = gt_valid_mask
    gt_values = gt_depth_m[valid_for_vis]
    if gt_values.size == 0:
        vis_min, vis_max = 0.0, 1.0
    else:
        vis_min = float(np.quantile(gt_values, 0.01))
        vis_max = float(np.quantile(gt_values, 0.99))
        if vis_max <= vis_min:
            vis_max = vis_min + 1e-6

    gt_vis = build_visual_map(gt_depth_m, valid_for_vis, vis_min, vis_max)
    raw_vis = build_visual_map(pred_depth_raw, valid_for_vis, vis_min, vis_max)
    aligned_vis = build_visual_map(pred_depth_aligned, valid_for_vis, vis_min, vis_max)
    fixed_scaled_vis = (
        build_visual_map(pred_depth_fixed_scaled, valid_for_vis, vis_min, vis_max)
        if pred_depth_fixed_scaled is not None
        else None
    )
    abs_error = np.abs(pred_depth_aligned - gt_depth_m)
    error_vis = build_error_map(abs_error, valid_for_vis, vmax=max(1.0, float(np.quantile(abs_error[valid_for_vis], 0.99)) if np.any(valid_for_vis) else 1.0))

    Image.fromarray(gt_vis).save(output_dir / "gt_depth_vis.png")
    Image.fromarray(raw_vis).save(output_dir / "pred_depth_raw_vis.png")
    Image.fromarray(aligned_vis).save(output_dir / "pred_depth_aligned_vis.png")
    if fixed_scaled_vis is not None:
        Image.fromarray(fixed_scaled_vis).save(output_dir / "pred_depth_fixed_scaled_vis.png")
    Image.fromarray(error_vis).save(output_dir / "abs_error_vis.png")

    panels = [
        ("RGB", rgb),
        ("GT depth (masked)", gt_vis),
        ("Pred depth raw", raw_vis),
        ("Pred depth aligned", aligned_vis),
    ]
    if fixed_scaled_vis is not None:
        panels.append(("Pred depth x fixed scale", fixed_scaled_vis))
    panels.append(("Abs error aligned", error_vis))

    title_h = 40
    gap = 16
    panel_h = 260
    resized_panels = []
    for title, panel in panels:
        panel_resized = np.array(Image.fromarray(panel).resize((int(panel.shape[1] * panel_h / panel.shape[0]), panel_h)))
        canvas = np.full((panel_h + title_h, panel_resized.shape[1], 3), 245, dtype=np.uint8)
        canvas[title_h:, :, :] = panel_resized
        cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        resized_panels.append(canvas)

    total_w = sum(panel.shape[1] for panel in resized_panels) + gap * (len(resized_panels) - 1)
    total_h = max(panel.shape[0] for panel in resized_panels)
    sheet = np.full((total_h, total_w, 3), 250, dtype=np.uint8)
    x = 0
    for panel in resized_panels:
        sheet[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    Image.fromarray(sheet).save(output_dir / "comparison_sheet.png")

    meta = {
        "input_image": str(image_path),
        "output_dir": str(output_dir),
    }
    with (output_dir / "artifacts.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt-depth", required=True)
    parser.add_argument("--depth-scale", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DAP_ROOT / "config" / "infer_local.yaml"))
    parser.add_argument("--weights", default=str(DAP_ROOT / "checkpoints" / "model.pth"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--fixed-scale", type=float, default=None)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    gt_depth_path = Path(args.gt_depth).resolve()
    depth_scale_path = Path(args.depth_scale).resolve()
    mask_path = Path(args.mask).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()

    depth_scale = float(depth_scale_path.read_text(encoding="utf-8").strip())
    device = resolve_device(args.device)
    os.chdir(DAP_ROOT)
    model, config = load_model(config_path, weights_path, device)

    input_cfg = config.get("input", {})
    input_size = None
    if input_cfg.get("width") and input_cfg.get("height"):
        input_size = (int(input_cfg["width"]), int(input_cfg["height"]))

    pred_depth_raw, model_valid_mask = infer_depth(model, image_path, input_size, device)

    rgb = np.array(Image.open(image_path).convert("RGB"))
    gt_depth_raw = np.array(Image.open(gt_depth_path)).astype(np.float32)
    gt_depth_m = gt_depth_raw / depth_scale
    pole_mask = np.array(Image.open(mask_path)) > 0
    gt_valid_mask = pole_mask & (gt_depth_raw > 0)
    eval_mask_with_model = gt_valid_mask & model_valid_mask

    raw_metrics = compute_metrics(gt_depth_m, pred_depth_raw, gt_valid_mask)
    median_scale = raw_metrics["gt_median_m"] / max(raw_metrics["pred_median_m"], 1e-8)
    pred_depth_aligned = pred_depth_raw * median_scale
    aligned_metrics = compute_metrics(gt_depth_m, pred_depth_aligned, gt_valid_mask)
    pred_depth_fixed_scaled = None
    fixed_scale_metrics = None
    fixed_scale_metrics_intersection = None
    if args.fixed_scale is not None:
        pred_depth_fixed_scaled = pred_depth_raw * float(args.fixed_scale)
        fixed_scale_metrics = compute_metrics(gt_depth_m, pred_depth_fixed_scaled, gt_valid_mask)

    if np.any(eval_mask_with_model):
        raw_metrics_intersection = compute_metrics(gt_depth_m, pred_depth_raw, eval_mask_with_model)
        aligned_metrics_intersection = compute_metrics(gt_depth_m, pred_depth_aligned, eval_mask_with_model)
        if pred_depth_fixed_scaled is not None:
            fixed_scale_metrics_intersection = compute_metrics(gt_depth_m, pred_depth_fixed_scaled, eval_mask_with_model)
    else:
        raw_metrics_intersection = None
        aligned_metrics_intersection = None

    report = {
        "image": str(image_path),
        "gt_depth": str(gt_depth_path),
        "mask": str(mask_path),
        "depth_scale": depth_scale,
        "device": device,
        "input_size_for_inference": list(input_size) if input_size else None,
        "image_shape": list(rgb.shape[:2]),
        "mask_valid_ratio": float(pole_mask.mean()),
        "gt_valid_ratio": float(gt_valid_mask.mean()),
        "model_valid_ratio": float(model_valid_mask.mean()),
        "gt_and_model_valid_ratio": float(eval_mask_with_model.mean()),
        "pred_depth_raw_quantiles": {
            "q00": float(np.quantile(pred_depth_raw, 0.00)),
            "q50": float(np.quantile(pred_depth_raw, 0.50)),
            "q95": float(np.quantile(pred_depth_raw, 0.95)),
            "q99": float(np.quantile(pred_depth_raw, 0.99)),
            "q100": float(np.quantile(pred_depth_raw, 1.00)),
        },
        "gt_depth_quantiles_m": {
            "q00": float(np.quantile(gt_depth_m[gt_valid_mask], 0.00)),
            "q50": float(np.quantile(gt_depth_m[gt_valid_mask], 0.50)),
            "q95": float(np.quantile(gt_depth_m[gt_valid_mask], 0.95)),
            "q99": float(np.quantile(gt_depth_m[gt_valid_mask], 0.99)),
            "q100": float(np.quantile(gt_depth_m[gt_valid_mask], 1.00)),
        },
        "median_scale_factor_to_match_gt": float(median_scale),
        "metrics_on_gt_valid_only": {
            "raw_pred": raw_metrics,
            "median_aligned_pred": aligned_metrics,
            "fixed_scaled_pred": fixed_scale_metrics,
        },
        "metrics_on_gt_valid_and_model_valid": {
            "raw_pred": raw_metrics_intersection,
            "median_aligned_pred": aligned_metrics_intersection,
            "fixed_scaled_pred": fixed_scale_metrics_intersection,
        },
    }

    save_artifacts(
        output_dir=output_dir,
        image_path=image_path,
        rgb=rgb,
        gt_depth_m=gt_depth_m,
        pred_depth_raw=pred_depth_raw,
        pred_depth_aligned=pred_depth_aligned,
        pred_depth_fixed_scaled=pred_depth_fixed_scaled,
        gt_valid_mask=gt_valid_mask,
        eval_mask_with_model=eval_mask_with_model,
        model_valid_mask=model_valid_mask,
    )

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
