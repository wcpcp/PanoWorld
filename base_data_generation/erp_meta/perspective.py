from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .erp_projection import direction_to_lonlat, lonlat_to_xy


def perspective_remap(
    out_size: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    erp_w: int,
    erp_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (map_x, map_y) for sampling ERP into a perspective view.

    Coordinate convention matches erp_projection.direction_to_lonlat:
      - +x: right
      - +y: up
      - +z: forward

    yaw rotates around +y, pitch rotates around +x (positive pitch looks up).
    """
    if out_size <= 0:
        raise ValueError("out_size must be positive")
    if fov_deg <= 0 or fov_deg >= 179:
        raise ValueError("fov_deg must be in (0,179)")

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # pixel centers
    jj, ii = np.meshgrid(np.arange(out_size, dtype=np.float32), np.arange(out_size, dtype=np.float32))
    cx = (out_size - 1) / 2.0
    cy = (out_size - 1) / 2.0

    f = (out_size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    x = (jj - cx) / f
    y = -(ii - cy) / f
    z = np.ones_like(x)

    # normalize
    n = np.sqrt(x * x + y * y + z * z)
    x /= n
    y /= n
    z /= n

    # pitch around x axis
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    y1 = y * cp + z * sp
    z1 = -y * sp + z * cp
    x1 = x

    # yaw around y axis
    cyaw = math.cos(yaw)
    syaw = math.sin(yaw)
    x2 = x1 * cyaw + z1 * syaw
    z2 = -x1 * syaw + z1 * cyaw
    y2 = y1

    lon, lat = direction_to_lonlat(x2, y2, z2)
    map_x, map_y = lonlat_to_xy(lon, lat, erp_w, erp_h)
    map_x = np.mod(map_x, erp_w).astype(np.float32)
    map_y = np.clip(map_y, 0, erp_h - 1).astype(np.float32)
    return map_x, map_y


def yaw_candidates_4() -> list[float]:
    return [0.0, 90.0, 180.0, 270.0]


def yaw_candidates(n: int, offset_deg: float = 0.0) -> list[float]:
    """Evenly spaced yaw candidates covering 360 degrees.

    Example:
      n=8 -> 0,45,90,...,315
    """
    if n <= 0:
        raise ValueError("n must be positive")
    step = 360.0 / float(n)
    return [offset_deg + i * step for i in range(n)]
