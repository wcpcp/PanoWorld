from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _encode_with_pycocotools(mask: np.ndarray) -> Dict[str, Any] | None:
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # pycocotools expects Fortran order HxWx1
    rle = mask_utils.encode(np.asfortranarray(mask))
    # rle is dict with bytes counts; convert to utf-8 string for JSON.
    if isinstance(rle.get("counts"), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = [int(rle["size"][0]), int(rle["size"][1])]
    return rle


def _encode_rle_f_order(mask: np.ndarray) -> list[int]:
    """Encode binary mask to COCO counts (uncompressed) in Fortran order."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Flatten in column-major (Fortran) order
    pixels = mask.T.reshape(-1)
    counts = []
    count = 0
    prev = 0
    for p in pixels:
        if p != prev:
            counts.append(count)
            count = 1
            prev = int(p)
        else:
            count += 1
    counts.append(count)
    return counts


def _decode_compressed_counts(counts_text: str) -> list[int]:
    """Decode pycocotools compressed COCO RLE counts without external deps."""
    counts: list[int] = []
    idx = 0
    text_len = len(counts_text)
    while idx < text_len:
        value = 0
        shift = 0
        more = True
        while more:
            char_code = ord(counts_text[idx]) - 48
            idx += 1
            value |= (char_code & 0x1F) << (5 * shift)
            more = (char_code & 0x20) != 0
            shift += 1
            if not more and (char_code & 0x10):
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """Return COCO-style RLE dict.

    Uses uncompressed counts list so saved JSON stays portable across
    environments, even when `pycocotools` is unavailable downstream.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape
    counts = _encode_rle_f_order(mask)
    return {"size": [int(h), int(w)], "counts": counts}


def decode_binary_mask(rle: Dict[str, Any]) -> np.ndarray:
    # pycocotools compressed RLE
    if isinstance(rle.get("counts"), str):
        try:
            from pycocotools import mask as mask_utils  # type: ignore

            m = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("utf-8")})
            return m.astype(np.uint8)
        except Exception:
            counts = _decode_compressed_counts(rle["counts"])
    else:
        counts = rle["counts"]

    h, w = rle["size"]
    # Reconstruct flattened array in Fortran order
    total = h * w
    out = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if c:
            out[idx : idx + c] = val
        idx += c
        val = 1 - val
    if idx != total:
        raise ValueError("RLE counts do not match mask size")
    mask = out.reshape((w, h)).T
    return mask


def mask_area(rle: Dict[str, Any]) -> int:
    if isinstance(rle.get("counts"), str):
        try:
            from pycocotools import mask as mask_utils  # type: ignore

            return int(mask_utils.area({"size": rle["size"], "counts": rle["counts"].encode("utf-8")}))
        except Exception:
            counts = _decode_compressed_counts(rle["counts"])
    else:
        counts = rle["counts"]

    # counts start with zeros
    area = 0
    val = 0
    for c in counts:
        if val == 1:
            area += int(c)
        val = 1 - val
    return area


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)
