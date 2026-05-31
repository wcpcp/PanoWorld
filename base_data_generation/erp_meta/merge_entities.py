from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .mask_ops import mask_centroid_lonlat, mask_iou, spherical_distance
from .rle import bbox_from_mask, decode_binary_mask, encode_binary_mask


def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().replace("_", " ").split())
    tb = set(b.lower().replace("_", " ").split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


@dataclass
class ProjectedInstance:
    view_id: str
    label: str
    score: float
    mask_erp: np.ndarray  # uint8 (H,W)


def merge_projected_instances(
    instances: List[ProjectedInstance],
    iou_thr: float = 0.25,
    dist_thr_rad: float = 0.25,
    sem_thr: float = 0.2,
) -> List[Dict]:
    """Greedy agglomerative merge.

    Each merged entity stores union mask and aggregated label/score.
    """
    entities: List[Dict] = []

    for inst in instances:
        placed = False
        for e in entities:
            iou = mask_iou(inst.mask_erp, e["mask"])
            lonlat_i = mask_centroid_lonlat(inst.mask_erp)
            lonlat_e = e["lonlat"]
            dist = spherical_distance(lonlat_i, lonlat_e)
            sem = _token_jaccard(inst.label, e["label_open"])
            if (iou >= iou_thr) or ((dist <= dist_thr_rad) and (sem >= sem_thr)):
                e["mask"] = np.maximum(e["mask"], inst.mask_erp)
                e["scores"].append(inst.score)
                e["labels"].append(inst.label)
                e["views"].append(inst.view_id)
                # update representative label by max token overlap frequency
                e["label_open"] = max(set(e["labels"]), key=e["labels"].count)
                e["lonlat"] = mask_centroid_lonlat(e["mask"])
                placed = True
                break
        if not placed:
            entities.append(
                {
                    "mask": inst.mask_erp.copy(),
                    "scores": [inst.score],
                    "labels": [inst.label],
                    "label_open": inst.label,
                    "views": [inst.view_id],
                    "lonlat": mask_centroid_lonlat(inst.mask_erp),
                }
            )

    out = []
    for idx, e in enumerate(entities):
        mask = e["mask"].astype(np.uint8)
        rle = encode_binary_mask(mask)
        bbox = bbox_from_mask(mask)
        conf = float(np.mean(e["scores"])) if e["scores"] else 0.0
        conf_max = float(np.max(e["scores"])) if e["scores"] else 0.0
        area_ratio = float(mask.sum()) / float(mask.size)
        label_votes = {label: e["labels"].count(label) for label in sorted(set(e["labels"]))}
        source_views = sorted(set(e["views"]))
        out.append(
            {
                "entity_id": f"E{idx:06d}",
                "label_open": e["label_open"],
                "confidence": conf,
                "confidence_max": conf_max,
                "mask_rle": rle,
                "bbox_xyxy": list(map(int, bbox)),
                "lon_lat": [float(e["lonlat"][0]), float(e["lonlat"][1])],
                "area_ratio": area_ratio,
                "source_views": source_views,
                "support_count": int(len(e["views"])),
                "source_view_count": int(len(source_views)),
                "label_votes": label_votes,
            }
        )
    return out
