from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def crop_with_bbox(pil_img: Image.Image, bbox_xyxy: Tuple[int, int, int, int], pad: int = 16) -> Image.Image:
    w, h = pil_img.size
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return pil_img.crop((x1, y1, x2, y2))


def crop_with_bbox_info(
    pil_img: Image.Image, bbox_xyxy: Tuple[int, int, int, int], pad: int = 16
) -> tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    w, h = pil_img.size
    x1, y1, x2, y2 = bbox_xyxy
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(w, x2 + pad)
    crop_y2 = min(h, y2 + pad)
    crop = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    local_bbox = (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2), local_bbox


def expand_bbox_xyxy(
    bbox_xyxy: Tuple[int, int, int, int], image_size: Tuple[int, int], scale: float = 2.5, min_pad: int = 32
) -> Tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = max(box_w * scale * 0.5, box_w * 0.5 + min_pad)
    half_h = max(box_h * scale * 0.5, box_h * 0.5 + min_pad)
    nx1 = max(0, int(round(cx - half_w)))
    ny1 = max(0, int(round(cy - half_h)))
    nx2 = min(width, int(round(cx + half_w)))
    ny2 = min(height, int(round(cy + half_h)))
    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)
    return nx1, ny1, nx2, ny2


def draw_bbox_outline(
    pil_img: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 3,
) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_xyxy
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    return img


def overlay_mask(pil_img: Image.Image, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.35) -> Image.Image:
    img = np.array(pil_img.convert("RGB"), dtype=np.float32)
    m = (mask > 0).astype(np.float32)
    if m.shape[:2] != img.shape[:2]:
        raise ValueError("mask size mismatch")
    col = np.array(color, dtype=np.float32)[None, None, :]
    img = img * (1 - alpha * m[..., None]) + col * (alpha * m[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
