from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .erp_projection import backproject_mask_to_erp
from .rle import decode_binary_mask
from .view_sampling import ViewSpec, view_to_erp_maps


@dataclass(frozen=True)
class OverlapPairMetrics:
    view_id_a: str
    view_id_b: str
    overlap_area: int
    fg_area_a: int
    fg_area_b: int
    fg_intersection: int
    fg_union: int
    fg_iou: float
    fg_precision: float
    fg_recall: float


def footprint_erp(view: ViewSpec) -> np.ndarray:
    """Binary ERP mask indicating which ERP pixels are hit by this view's remap.

    This is a geometry-only footprint (independent of segmentation).
    """
    map_x, map_y = view_to_erp_maps(view)
    h, w = map_x.shape[:2]
    ones = np.ones((h, w), dtype=np.uint8)
    return backproject_mask_to_erp(ones, map_x, map_y, view.erp_w, view.erp_h).astype(np.uint8)


def seg_fg_mask_erp(
    view: ViewSpec,
    segs: list[dict],
    pano_mask01: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Union of all instance masks in this view, backprojected to ERP."""
    map_x, map_y = view_to_erp_maps(view)
    fg = np.zeros((view.erp_h, view.erp_w), dtype=np.uint8)
    for s in segs:
        try:
            mask_view = decode_binary_mask(s["rle"]).astype(np.uint8)
        except Exception:
            continue
        mask_erp = backproject_mask_to_erp(mask_view, map_x, map_y, view.erp_w, view.erp_h).astype(np.uint8)
        if pano_mask01 is not None and pano_mask01.shape[:2] == mask_erp.shape[:2]:
            mask_erp = (mask_erp & pano_mask01).astype(np.uint8)
        fg |= (mask_erp > 0).astype(np.uint8)
    return fg


def det_fg_mask_erp(
    view: ViewSpec,
    dets: list[dict],
    pano_mask01: Optional[np.ndarray] = None,
) -> np.ndarray:
    map_x, map_y = view_to_erp_maps(view)
    h, w = map_x.shape[:2]
    mask_view = np.zeros((h, w), dtype=np.uint8)
    for d in dets:
        bbox = d.get("bbox") or d.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = int(np.floor(x1))
        y1 = int(np.floor(y1))
        x2 = int(np.ceil(x2))
        y2 = int(np.ceil(y2))
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        mask_view[y1:y2, x1:x2] = 1

    fg = backproject_mask_to_erp(mask_view, map_x, map_y, view.erp_w, view.erp_h).astype(np.uint8)
    if pano_mask01 is not None and pano_mask01.shape[:2] == fg.shape[:2]:
        fg = (fg & pano_mask01).astype(np.uint8)
    return fg


def overlap_pair_metrics(
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    overlap01: np.ndarray,
    view_id_a: str,
    view_id_b: str,
) -> OverlapPairMetrics:
    """Compute overlap-region foreground agreement metrics."""
    overlap01 = (overlap01 > 0).astype(np.uint8)
    a = ((fg_a > 0).astype(np.uint8) & overlap01).astype(np.uint8)
    b = ((fg_b > 0).astype(np.uint8) & overlap01).astype(np.uint8)

    overlap_area = int(overlap01.sum())
    fg_area_a = int(a.sum())
    fg_area_b = int(b.sum())

    inter = int((a & b).sum())
    union = int(((a | b) > 0).sum())

    fg_iou = float(inter / union) if union > 0 else 1.0
    fg_precision = float(inter / fg_area_a) if fg_area_a > 0 else 1.0
    fg_recall = float(inter / fg_area_b) if fg_area_b > 0 else 1.0

    return OverlapPairMetrics(
        view_id_a=view_id_a,
        view_id_b=view_id_b,
        overlap_area=overlap_area,
        fg_area_a=fg_area_a,
        fg_area_b=fg_area_b,
        fg_intersection=inter,
        fg_union=union,
        fg_iou=fg_iou,
        fg_precision=fg_precision,
        fg_recall=fg_recall,
    )
