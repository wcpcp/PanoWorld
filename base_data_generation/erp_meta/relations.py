from __future__ import annotations

import math
from typing import Dict, List


def _wrap_pi(x: float) -> float:
    while x <= -math.pi:
        x += 2 * math.pi
    while x > math.pi:
        x -= 2 * math.pi
    return x


def _lon_sector(lon: float) -> str:
    sector_names = [
        "front",
        "front_right",
        "right",
        "back_right",
        "back",
        "back_left",
        "left",
        "front_left",
    ]
    idx = int(round((_wrap_pi(lon) % (2 * math.pi)) / (math.pi / 4))) % 8
    return sector_names[idx]


def _lat_band(lat: float) -> str:
    if lat <= -0.45:
        return "ceiling_side"
    if lat <= -0.15:
        return "upper"
    if lat <= 0.15:
        return "middle"
    if lat <= 0.45:
        return "lower"
    return "floor_side"


def build_entity_contexts(entities: List[Dict]) -> List[Dict]:
    contexts: List[Dict] = []
    for entity in entities:
        lon, lat = entity["lon_lat"]
        sector = _lon_sector(float(lon))
        band = _lat_band(float(lat))
        source_views = entity.get("source_views", [])
        contexts.append(
            {
                "entity_id": entity["entity_id"],
                "erp_sector": sector,
                "vertical_band": band,
                "source_views": source_views,
                "text_hints": [
                    f"located in the {sector} sector of the panorama",
                    f"appears in the {band} vertical band",
                ]
                + ([f"supported by views: {', '.join(source_views[:3])}"] if source_views else []),
            }
        )
    return contexts


def build_relations(entities: List[Dict], near_thr_rad: float = 0.35) -> List[Dict]:
    rels: List[Dict] = []
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue
            a = entities[i]
            b = entities[j]
            lon1, lat1 = a["lon_lat"]
            lon2, lat2 = b["lon_lat"]
            dlon = _wrap_pi(lon2 - lon1)
            dlat = lat2 - lat1

            # left/right/above/below: low-precision but stable
            if abs(dlon) > 0.2 and abs(dlon) > abs(dlat):
                rels.append(
                    {
                        "subject": a["entity_id"],
                        "object": b["entity_id"],
                        "relation": "right_of" if dlon > 0 else "left_of",
                        "confidence": min(1.0, abs(dlon) / math.pi),
                    }
                )
            if abs(dlat) > 0.15 and abs(dlat) > abs(dlon) / 2:
                rels.append(
                    {
                        "subject": a["entity_id"],
                        "object": b["entity_id"],
                        "relation": "below" if dlat > 0 else "above",
                        "confidence": min(1.0, abs(dlat) / (math.pi / 2)),
                    }
                )

            # near
            dist = math.sqrt((dlon * math.cos((lat1 + lat2) / 2)) ** 2 + (dlat) ** 2)
            if dist <= near_thr_rad:
                rels.append(
                    {
                        "subject": a["entity_id"],
                        "object": b["entity_id"],
                        "relation": "near",
                        "confidence": max(0.0, 1.0 - dist / near_thr_rad),
                    }
                )

            if _lon_sector(float(lon1)) == _lon_sector(float(lon2)):
                rels.append(
                    {
                        "subject": a["entity_id"],
                        "object": b["entity_id"],
                        "relation": "same_sector",
                        "confidence": 0.6,
                    }
                )

    return rels
