from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class DetBox:
    bbox_xyxy: Tuple[float, float, float, float]
    label: str
    score: float


@dataclass
class SegMask:
    # COCO-style RLE dict: {"size": [h,w], "counts": ...}
    rle: Dict[str, Any]
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    label: Optional[str] = None


@dataclass
class Entity:
    entity_id: str
    label_open: str
    confidence: float
    mask_rle: Dict[str, Any]
    bbox_xyxy: Tuple[int, int, int, int]
    lon_lat: Tuple[float, float]
    area_ratio: float
    source_views: List[str]


RelationType = Literal[
    "left_of",
    "right_of",
    "above",
    "below",
    "near",
    "overlap",
    "inside",
    "contains",
]


@dataclass
class Relation:
    subject: str
    object: str
    relation: RelationType
    confidence: float
