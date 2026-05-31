from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np


try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


FaceName = Literal["front", "right", "back", "left", "up", "down"]


@dataclass(frozen=True)
class ERPSize:
    width: int
    height: int


def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, erp_w: int, erp_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """lon in [-pi, pi), lat in [-pi/2, pi/2]. Returns float x,y in ERP pixel coords."""
    x = (lon + math.pi) / (2 * math.pi) * erp_w
    y = (math.pi / 2 - lat) / math.pi * erp_h
    return x, y


def direction_to_lonlat(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(x, z)
    hyp = np.sqrt(x * x + z * z)
    lat = np.arctan2(y, hyp)
    return lon, lat


def _face_dir(face: FaceName, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map cube face coords (a,b in [-1,1]) to direction vector."""
    if face == "front":
        x, y, z = a, -b, np.ones_like(a)
    elif face == "right":
        x, y, z = np.ones_like(a), -b, -a
    elif face == "back":
        x, y, z = -a, -b, -np.ones_like(a)
    elif face == "left":
        x, y, z = -np.ones_like(a), -b, a
    elif face == "up":
        x, y, z = a, np.ones_like(a), b
    elif face == "down":
        x, y, z = a, -np.ones_like(a), -b
    else:
        raise ValueError(f"Unknown face: {face}")

    n = np.sqrt(x * x + y * y + z * z)
    return x / n, y / n, z / n


def cubemap_remap(face: FaceName, face_size: int, erp_w: int, erp_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (map_x, map_y) for cv2.remap from ERP to cube face image."""
    # pixel centers in [0, face_size)
    jj, ii = np.meshgrid(np.arange(face_size, dtype=np.float32), np.arange(face_size, dtype=np.float32))
    a = (jj + 0.5) / face_size * 2 - 1
    b = (ii + 0.5) / face_size * 2 - 1
    dx, dy, dz = _face_dir(face, a, b)
    lon, lat = direction_to_lonlat(dx, dy, dz)
    map_x, map_y = lonlat_to_xy(lon, lat, erp_w, erp_h)
    # wrap x seam
    map_x = np.mod(map_x, erp_w).astype(np.float32)
    map_y = np.clip(map_y, 0, erp_h - 1).astype(np.float32)
    return map_x, map_y


def remap_erp_to_view(erp_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """erp_rgb: HxWx3 uint8"""
    if _HAS_CV2:
        return cv2.remap(erp_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # Numpy-only bilinear sampling (slower):
    h, w = erp_rgb.shape[:2]
    x0 = np.floor(map_x).astype(np.int32) % w
    y0 = np.floor(map_y).astype(np.int32)
    x1 = (x0 + 1) % w
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (map_x - np.floor(map_x))[..., None]
    wy = (map_y - np.floor(map_y))[..., None]

    Ia = erp_rgb[y0, x0]
    Ib = erp_rgb[y0, x1]
    Ic = erp_rgb[y1, x0]
    Id = erp_rgb[y1, x1]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy
    return out.astype(np.uint8)


def backproject_mask_to_erp(mask_view: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, erp_w: int, erp_h: int) -> np.ndarray:
    """Backproject a view mask (HxW bool/uint8) onto ERP grid by nearest mapping."""
    ys, xs = np.nonzero(mask_view)
    if ys.size == 0:
        return np.zeros((erp_h, erp_w), dtype=np.uint8)
    ex = np.round(map_x[ys, xs]).astype(np.int32) % erp_w
    ey = np.round(map_y[ys, xs]).astype(np.int32)
    valid = (ey >= 0) & (ey < erp_h)
    ex = ex[valid]
    ey = ey[valid]
    out = np.zeros((erp_h, erp_w), dtype=np.uint8)
    out[ey, ex] = 1
    return out
