from __future__ import annotations

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def mask_centroid_lonlat(mask: np.ndarray) -> tuple[float, float]:
    """Return (lon, lat) centroid in radians using circular mean for lon."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0.0, 0.0)

    lon = (xs + 0.5) / w * (2 * np.pi) - np.pi
    lat = np.pi / 2 - (ys + 0.5) / h * np.pi

    # circular mean for lon
    sin_mean = np.sin(lon).mean()
    cos_mean = np.cos(lon).mean()
    lon_mean = float(np.arctan2(sin_mean, cos_mean))
    lat_mean = float(lat.mean())
    return lon_mean, lat_mean


def spherical_distance(lonlat_a: tuple[float, float], lonlat_b: tuple[float, float]) -> float:
    lon1, lat1 = lonlat_a
    lon2, lat2 = lonlat_b
    # haversine on unit sphere
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sa = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * np.arcsin(np.sqrt(sa)))
