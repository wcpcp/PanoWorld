#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_generate_sft_qa.py

根据 erp_subskills_spec.md，从 entities 元数据构建 ERP SFT/Benchmark QA。

设计目标:
1) 可判定任务优先: 用几何/深度/语义字段直接生成高可靠标签。
2) 开放任务可选: 用 VLM 生成场景级描述类 QA, 并做基础校验。
3) 避免组合爆炸: 每 scene 仅采样核心实体与有限 pair。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from _common import load_cfg
from erp_meta.models.vlm_qwen import _extract_json
from erp_meta.pipeline import build_vlm
try:
    from scripts.erp_qa_templates import COUNTABLE_CATEGORIES_HINT, NEGATIVE_EXISTENCE_CANDIDATES, OPEN_TASK_PROMPTS, QA_TEMPLATES  # pyright: ignore[reportMissingImports]
except Exception:
    from erp_qa_templates import COUNTABLE_CATEGORIES_HINT, NEGATIVE_EXISTENCE_CANDIDATES, OPEN_TASK_PROMPTS, QA_TEMPLATES  # type: ignore


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _norm_label(text: str) -> str:
    return str(text or "").strip().lower()


def _get_identify(entity: dict[str, Any]) -> str:
    semantic = entity.get("semantic", {})
    identify = semantic.get("identify") or semantic.get("semantic_type") or entity.get("label_open") or "object"
    return str(identify).strip()


def _get_caption(entity: dict[str, Any]) -> str:
    semantic = entity.get("semantic", {})
    caption = semantic.get("caption_brief") or semantic.get("caption_dense") or _get_identify(entity)
    return str(caption).strip()


def _get_semantic_conf(entity: dict[str, Any]) -> float:
    semantic = entity.get("semantic", {})
    return _safe_float(semantic.get("confidence", semantic.get("semantic_confidence", 0.0)), 0.0)


def _get_depth_valid_ratio(entity: dict[str, Any]) -> float:
    depth = entity.get("depth", {})
    return _safe_float(depth.get("valid_ratio", 0.0), 0.0)


def _get_distance_m(entity: dict[str, Any]) -> float | None:
    spatial = entity.get("spatial", {})
    range_m = spatial.get("range_m", None)
    if range_m is not None:
        d = _safe_float(range_m, -1.0)
        return d if d > 0 else None
    depth = entity.get("depth", {})
    median = depth.get("median", None)
    if median is not None:
        d = _safe_float(median, -1.0)
        return d if d > 0 else None
    return None


def _get_yaw_pitch(entity: dict[str, Any]) -> tuple[float | None, float | None]:
    spatial = entity.get("spatial", {})
    yaw = spatial.get("yaw", None)
    pitch = spatial.get("pitch", None)
    if yaw is None and pitch is None:
        return None, None
    return _safe_float(yaw, 0.0), _safe_float(pitch, 0.0)


def _valid_bbox(bbox: list[int] | tuple[int, int, int, int]) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [_safe_int(v, 0) for v in bbox]
    return x2 > x1 and y2 > y1


def _bbox_area(bbox: list[int] | tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = [_safe_int(v, 0) for v in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_to_box_string(bbox: list[int] | tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = [_safe_int(v, 0) for v in bbox]
    return f"[{y1},{x1},{y2},{x2}]"


def _normalize_angle_deg(angle: float) -> float:
    a = float(angle)
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def _yaw_to_direction8(yaw: float) -> str:
    y = _normalize_angle_deg(yaw)
    bins = [
        (-22.5, 22.5, "前方"),
        (22.5, 67.5, "右前方"),
        (67.5, 112.5, "右侧"),
        (112.5, 157.5, "右后方"),
        (-67.5, -22.5, "左前方"),
        (-112.5, -67.5, "左侧"),
        (-157.5, -112.5, "左后方"),
    ]
    for lo, hi, label in bins:
        if lo <= y < hi:
            return label
    return "后方"


def _yaw_to_clock(yaw: float) -> str:
    y = _normalize_angle_deg(yaw)
    if y < 0:
        y += 360.0
    clock = int(round(y / 30.0)) % 12
    if clock == 0:
        return "12点钟方向"
    return f"{clock}点钟方向"


def _relative_position(yaw_a: float, pitch_a: float, yaw_b: float, pitch_b: float) -> str:
    dyaw = _normalize_angle_deg(yaw_a - yaw_b)
    dpitch = float(pitch_a - pitch_b)
    horizontal = _yaw_to_direction8(dyaw)
    vertical = ""
    if dpitch >= 12.0:
        vertical = "偏上"
    elif dpitch <= -12.0:
        vertical = "偏下"
    if not vertical:
        return horizontal
    if horizontal == "前方":
        return vertical
    return f"{horizontal}{vertical}"


def _distance_bucket(distance_m: float) -> str:
    d = float(distance_m)
    if d < 2.0:
        return "0-2m"
    if d < 4.0:
        return "2-4m"
    if d < 8.0:
        return "4-8m"
    return "8m以上"


def _scene_identity(scene_obj: dict[str, Any], fallback_name: str = "unknown") -> tuple[str, str]:
    scene_id = str(scene_obj.get("scene_id", "")).strip() or fallback_name
    viewpoint_id = str(scene_obj.get("viewpoint_id", "")).strip() or str(scene_obj.get("vp", "")).strip() or ""
    return scene_id, viewpoint_id


def _choose_key_entities(entities: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    if not entities:
        return []
    max_area = max(_bbox_area(e.get("bbox_xyxy", [0, 0, 0, 0])) for e in entities) or 1.0

    def score(e: dict[str, Any]) -> float:
        conf = _get_semantic_conf(e)
        depth_ratio = _get_depth_valid_ratio(e)
        area = _bbox_area(e.get("bbox_xyxy", [0, 0, 0, 0])) / max_area
        return 1.8 * conf + 0.8 * depth_ratio + 0.8 * area

    ranked = sorted(entities, key=score, reverse=True)
    return ranked[: max(1, int(max_count))]


def is_valid_entity(entity: dict[str, Any], min_sem_conf: float, min_depth_valid: float) -> bool:
    if not _valid_bbox(entity.get("bbox_xyxy", [])):
        return False
    if not _get_identify(entity):
        return False
    if _get_semantic_conf(entity) < min_sem_conf:
        return False
    depth_ratio = _get_depth_valid_ratio(entity)
    if depth_ratio > 0 and depth_ratio < min_depth_valid:
        return False
    return True


def _qa_row(
    *,
    scene_id: str,
    viewpoint_id: str,
    task_level: str,
    task_name: str,
    question: str,
    answer: str,
    source: str,
    reliability: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "viewpoint_id": viewpoint_id,
        "task_level": task_level,
        "task": task_name,
        "question": question,
        "answer": answer,
        "source": source,
        "reliability": reliability,
        "meta": meta,
    }


def build_basic_qa(
    entities: list[dict[str, Any]],
    scene_id: str,
    viewpoint_id: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    key_entities = _choose_key_entities(entities, args.max_grounding_qas)

    cat_counter = Counter(_norm_label(_get_identify(e)) for e in entities)
    original_name = {_norm_label(_get_identify(e)): _get_identify(e) for e in entities}

    prompts = QA_TEMPLATES["1_basic_understanding"]["1.1_grounding_to_text"]["prompts"]
    ans_fmt = QA_TEMPLATES["1_basic_understanding"]["1.1_grounding_to_text"]["answer_format"]
    for ent in key_entities:
        bbox = ent.get("bbox_xyxy", [0, 0, 0, 0])
        q = rng.choice(prompts).format(bbox=_bbox_to_box_string(bbox))
        a = ans_fmt.format(identify=_get_identify(ent), caption_brief=_get_caption(ent))
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="basic",
                task_name="region_to_text",
                question=q,
                answer=a,
                source="rule",
                reliability="high",
                meta={"entity_id": ent.get("entity_id", ""), "bbox_xyxy": bbox},
            )
        )

    prompts = QA_TEMPLATES["1_basic_understanding"]["1.1_text_to_grounding"]["prompts"]
    ans_fmt = QA_TEMPLATES["1_basic_understanding"]["1.1_text_to_grounding"]["answer_format"]
    text2region_count = 0
    for ent in key_entities:
        if text2region_count >= args.max_text_to_region_qas:
            break
        identify = _get_identify(ent)
        label = _norm_label(identify)
        if not label or cat_counter.get(label, 0) != 1:
            continue
        bbox = ent.get("bbox_xyxy", [0, 0, 0, 0])
        q = rng.choice(prompts).format(unique_description=identify)
        a = ans_fmt.format(bbox=_bbox_to_box_string(bbox))
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="basic",
                task_name="text_to_region",
                question=q,
                answer=a,
                source="rule",
                reliability="high",
                meta={"entity_id": ent.get("entity_id", ""), "identify": identify, "bbox_xyxy": bbox},
            )
        )
        text2region_count += 1

    exist_prompts = QA_TEMPLATES["1_basic_understanding"]["1.2_existence"]["prompts"]
    exist_ans = QA_TEMPLATES["1_basic_understanding"]["1.2_existence"]["answer_format"]
    present_categories = [original_name[k] for k in cat_counter.keys() if k]
    rng.shuffle(present_categories)
    for category in present_categories[: args.max_existence_positive_qas]:
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="basic",
                task_name="existence",
                question=rng.choice(exist_prompts).format(category=category),
                answer=exist_ans.format(yes_no="有"),
                source="rule",
                reliability="high",
                meta={"category": category, "label": "positive"},
            )
        )

    absent_candidates = [c for c in NEGATIVE_EXISTENCE_CANDIDATES if _norm_label(c) not in cat_counter]
    rng.shuffle(absent_candidates)
    for category in absent_candidates[: args.max_existence_negative_qas]:
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="basic",
                task_name="existence",
                question=rng.choice(exist_prompts).format(category=category),
                answer=exist_ans.format(yes_no="没有"),
                source="rule",
                reliability="high",
                meta={"category": category, "label": "negative"},
            )
        )

    cnt_prompts = QA_TEMPLATES["1_basic_understanding"]["1.2_counting"]["prompts"]
    cnt_ans = QA_TEMPLATES["1_basic_understanding"]["1.2_counting"]["answer_format"]
    count_candidates: list[tuple[str, int]] = []
    for norm_name, cnt in cat_counter.items():
        if cnt <= 0 or cnt > args.max_counting_value:
            continue
        name = original_name.get(norm_name, norm_name)
        if COUNTABLE_CATEGORIES_HINT and not any(h in name for h in COUNTABLE_CATEGORIES_HINT):
            continue
        count_candidates.append((name, cnt))
    rng.shuffle(count_candidates)
    for category, cnt in count_candidates[: args.max_counting_qas]:
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="basic",
                task_name="counting",
                question=rng.choice(cnt_prompts).format(category=category),
                answer=cnt_ans.format(count=cnt, category=category),
                source="rule",
                reliability="high",
                meta={"category": category, "count": cnt},
            )
        )

    return qas


def build_omni_qa(
    entities: list[dict[str, Any]],
    scene_id: str,
    viewpoint_id: str,
    rng: random.Random,
    args: argparse.Namespace,
    erp_size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    key_entities = _choose_key_entities(entities, max(args.max_direction_qas, args.max_relative_direction_qas + 1))

    prompts = QA_TEMPLATES["2_omnidirectional_understanding"]["2.1_directive_direction"]["prompts"]
    ans_fmt = QA_TEMPLATES["2_omnidirectional_understanding"]["2.1_directive_direction"]["answer_format"]
    direction_added = 0
    for ent in key_entities:
        if direction_added >= args.max_direction_qas:
            break
        yaw, pitch = _get_yaw_pitch(ent)
        if yaw is None:
            continue
        q = rng.choice(prompts).format(
            bbox=_bbox_to_box_string(ent.get("bbox_xyxy", [0, 0, 0, 0])),
            unique_description=_get_identify(ent),
        )
        a = ans_fmt.format(
            direction_8=_yaw_to_direction8(yaw),
            yaw_angle=int(round(yaw)),
            clock_direction=_yaw_to_clock(yaw),
        )
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="omni",
                task_name="directive_direction",
                question=q,
                answer=a,
                source="rule",
                reliability="high",
                meta={"entity_id": ent.get("entity_id", ""), "yaw": yaw, "pitch": pitch},
            )
        )
        direction_added += 1

    prompts = QA_TEMPLATES["2_omnidirectional_understanding"]["2.2_relative_direction"]["prompts"]
    ans_fmt = QA_TEMPLATES["2_omnidirectional_understanding"]["2.2_relative_direction"]["answer_format"]
    pair_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    yaw_entities = [e for e in key_entities if _get_yaw_pitch(e)[0] is not None]
    for i in range(len(yaw_entities)):
        for j in range(i + 1, len(yaw_entities)):
            pair_candidates.append((yaw_entities[i], yaw_entities[j]))
    rng.shuffle(pair_candidates)

    rel_added = 0
    for a_ent, b_ent in pair_candidates:
        if rel_added >= args.max_relative_direction_qas:
            break
        yaw_a, pitch_a = _get_yaw_pitch(a_ent)
        yaw_b, pitch_b = _get_yaw_pitch(b_ent)
        if yaw_a is None or yaw_b is None:
            continue
        rel = _relative_position(yaw_a, pitch_a or 0.0, yaw_b, pitch_b or 0.0)
        q = rng.choice(prompts).format(
            unique_description_A=_get_identify(a_ent),
            unique_description_B=_get_identify(b_ent),
            bbox_A=_bbox_to_box_string(a_ent.get("bbox_xyxy", [0, 0, 0, 0])),
            bbox_B=_bbox_to_box_string(b_ent.get("bbox_xyxy", [0, 0, 0, 0])),
        )
        a = ans_fmt.format(
            unique_description_A=_get_identify(a_ent),
            unique_description_B=_get_identify(b_ent),
            relative_position=rel,
        )
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="omni",
                task_name="relative_direction",
                question=q,
                answer=a,
                source="rule",
                reliability="high",
                meta={
                    "entity_id_A": a_ent.get("entity_id", ""),
                    "entity_id_B": b_ent.get("entity_id", ""),
                    "yaw_A": yaw_a,
                    "yaw_B": yaw_b,
                },
            )
        )
        rel_added += 1

    if erp_size is not None and args.max_boundary_qas > 0:
        width, _height = erp_size
        prompts = QA_TEMPLATES["2_omnidirectional_understanding"]["2.3_boundary_continuity"]["prompts"]
        ans_fmt = QA_TEMPLATES["2_omnidirectional_understanding"]["2.3_boundary_continuity"]["answer_format"]
        boundary_entities = []
        for ent in entities:
            bbox = ent.get("bbox_xyxy", [0, 0, 0, 0])
            if not _valid_bbox(bbox):
                continue
            x1, _y1, x2, _y2 = [_safe_int(v, 0) for v in bbox]
            if x1 <= args.boundary_margin_px or x2 >= (width - args.boundary_margin_px):
                boundary_entities.append(ent)

        rng.shuffle(boundary_entities)
        for ent in boundary_entities[: args.max_boundary_qas]:
            bbox = ent.get("bbox_xyxy", [0, 0, 0, 0])
            qas.append(
                _qa_row(
                    scene_id=scene_id,
                    viewpoint_id=viewpoint_id,
                    task_level="omni",
                    task_name="boundary_continuity",
                    question=rng.choice(prompts),
                    answer=ans_fmt.format(bbox_boundary=_bbox_to_box_string(bbox), identify=_get_identify(ent)),
                    source="rule",
                    reliability="medium",
                    meta={"entity_id": ent.get("entity_id", ""), "bbox_xyxy": bbox},
                )
            )

    if args.max_distortion_qas > 0:
        prompts = QA_TEMPLATES["2_omnidirectional_understanding"]["2.4_distortion_awareness"]["prompts"]
        polar_entities = []
        for ent in entities:
            _yaw, pitch = _get_yaw_pitch(ent)
            if pitch is None:
                continue
            if abs(float(pitch)) >= args.polar_pitch_deg:
                polar_entities.append(ent)

        rng.shuffle(polar_entities)
        for ent in polar_entities[: args.max_distortion_qas]:
            bbox = ent.get("bbox_xyxy", [0, 0, 0, 0])
            identify = _get_identify(ent)
            answer = (
                f"该目标位于 ERP 极区，图像中会出现横向拉伸。真实空间中它仍是正常比例的{identify}，"
                "拉伸主要来自球面到平面的投影畸变。"
            )
            qas.append(
                _qa_row(
                    scene_id=scene_id,
                    viewpoint_id=viewpoint_id,
                    task_level="omni",
                    task_name="distortion_awareness",
                    question=rng.choice(prompts).format(bbox=_bbox_to_box_string(bbox)),
                    answer=answer,
                    source="rule",
                    reliability="medium",
                    meta={"entity_id": ent.get("entity_id", ""), "pitch": _get_yaw_pitch(ent)[1]},
                )
            )

    return qas


def build_3d_qa(
    entities: list[dict[str, Any]],
    scene_id: str,
    viewpoint_id: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    depth_entities = [e for e in entities if (_get_distance_m(e) is not None and _get_depth_valid_ratio(e) >= args.min_depth_valid_ratio)]
    if not depth_entities:
        return qas

    key_entities = _choose_key_entities(depth_entities, max(args.max_distance_qas, args.max_relative_depth_qas + 1))

    prompts = QA_TEMPLATES["3_3d_spatial_understanding"]["3.1_distance_estimation"]["prompts"]
    ans_fmt = QA_TEMPLATES["3_3d_spatial_understanding"]["3.1_distance_estimation"]["answer_format"]
    added = 0
    for ent in key_entities:
        if added >= args.max_distance_qas:
            break
        d = _get_distance_m(ent)
        if d is None:
            continue
        q = rng.choice(prompts).format(
            bbox=_bbox_to_box_string(ent.get("bbox_xyxy", [0, 0, 0, 0])),
            unique_description=_get_identify(ent),
        )
        a = ans_fmt.format(distance_bucket=_distance_bucket(d), distance_m=float(d))
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="3d",
                task_name="distance_estimation",
                question=q,
                answer=a,
                source="rule",
                reliability="high",
                meta={"entity_id": ent.get("entity_id", ""), "distance_m": d},
            )
        )
        added += 1

    prompts = QA_TEMPLATES["3_3d_spatial_understanding"]["3.2_relative_positioning"]["prompts"]
    ans_fmt = QA_TEMPLATES["3_3d_spatial_understanding"]["3.2_relative_positioning"]["answer_format"]
    pair_candidates: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    for i in range(len(key_entities)):
        for j in range(i + 1, len(key_entities)):
            ea = key_entities[i]
            eb = key_entities[j]
            da = _get_distance_m(ea)
            db = _get_distance_m(eb)
            if da is None or db is None:
                continue
            rel_gap = abs(da - db) / max(da, db, 1e-6)
            if rel_gap < args.min_relative_depth_gap:
                continue
            pair_candidates.append((ea, eb, da, db))

    rng.shuffle(pair_candidates)
    rel_added = 0
    for ea, eb, da, db in pair_candidates:
        if rel_added >= args.max_relative_depth_qas:
            break
        closer = _get_identify(ea) if da < db else _get_identify(eb)
        q = rng.choice(prompts).format(
            bbox_A=_bbox_to_box_string(ea.get("bbox_xyxy", [0, 0, 0, 0])),
            bbox_B=_bbox_to_box_string(eb.get("bbox_xyxy", [0, 0, 0, 0])),
            unique_description_A=_get_identify(ea),
            unique_description_B=_get_identify(eb),
        )
        a = ans_fmt.format(closer_item=closer, distance_A=float(da), distance_B=float(db))
        qas.append(
            _qa_row(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                task_level="3d",
                task_name="relative_depth",
                question=q,
                answer=a,
                source="rule",
                reliability="high",
                meta={
                    "entity_id_A": ea.get("entity_id", ""),
                    "entity_id_B": eb.get("entity_id", ""),
                    "distance_A": da,
                    "distance_B": db,
                },
            )
        )
        rel_added += 1

    return qas


def _build_entity_summary_text(entities: list[dict[str, Any]], max_items: int = 8) -> str:
    rows = []
    for ent in _choose_key_entities(entities, max_items):
        yaw, pitch = _get_yaw_pitch(ent)
        d = _get_distance_m(ent)
        rows.append(
            {
                "identify": _get_identify(ent),
                "bbox": ent.get("bbox_xyxy", [0, 0, 0, 0]),
                "yaw": None if yaw is None else round(float(yaw), 2),
                "pitch": None if pitch is None else round(float(pitch), 2),
                "distance_m": None if d is None else round(float(d), 2),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def _validate_llm_qa(question: str, answer: str) -> bool:
    q = str(question or "").strip()
    a = str(answer or "").strip()
    if len(q) < 8 or len(a) < 8:
        return False
    if "```" in q or "```" in a:
        return False
    return True


def build_llm_open_qa(
    entities: list[dict[str, Any]],
    scene_obj: dict[str, Any],
    scene_id: str,
    viewpoint_id: str,
    vlm: Any | None,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if vlm is None:
        return []
    erp_path = str(scene_obj.get("erp_path", "")).strip()
    if not erp_path or not Path(erp_path).exists():
        return []

    qas: list[dict[str, Any]] = []
    erp_img = Image.open(erp_path).convert("RGB")
    summary = _build_entity_summary_text(entities)

    scene_prompt = OPEN_TASK_PROMPTS["scene_layout"] + f"\n关键实体摘要: {summary}"
    try:
        out_text = vlm.chat_multi_image([erp_img], scene_prompt)
        obj = _extract_json(out_text)
        for row in obj.get("qas", [])[: args.max_llm_scene_qas]:
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not _validate_llm_qa(q, a):
                continue
            qas.append(
                _qa_row(
                    scene_id=scene_id,
                    viewpoint_id=viewpoint_id,
                    task_level="basic",
                    task_name="scene_layout_open",
                    question=q,
                    answer=a,
                    source="llm",
                    reliability="llm_medium",
                    meta={"generator": "scene_layout", "verified": "basic"},
                )
            )
    except Exception:
        pass

    polar_entities = []
    for ent in entities:
        _yaw, pitch = _get_yaw_pitch(ent)
        if pitch is not None and abs(float(pitch)) >= args.polar_pitch_deg:
            polar_entities.append(ent)
    if polar_entities:
        target = rng.choice(polar_entities)
        bbox = _bbox_to_box_string(target.get("bbox_xyxy", [0, 0, 0, 0]))
        identify = _get_identify(target)
        polar_prompt = OPEN_TASK_PROMPTS["polar_distortion"] + f"\n目标: {identify}, bbox={bbox}"
        try:
            out_text = vlm.chat_multi_image([erp_img], polar_prompt)
            obj = _extract_json(out_text)
            for row in obj.get("qas", [])[: args.max_llm_distortion_qas]:
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if not _validate_llm_qa(q, a):
                    continue
                qas.append(
                    _qa_row(
                        scene_id=scene_id,
                        viewpoint_id=viewpoint_id,
                        task_level="omni",
                        task_name="distortion_open",
                        question=q,
                        answer=a,
                        source="llm",
                        reliability="llm_medium",
                        meta={"generator": "polar_distortion", "target_identify": identify, "bbox": bbox},
                    )
                )
        except Exception:
            pass

    return qas


def _dedup_qas(qas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniq: list[dict[str, Any]] = []
    seen = set()
    for row in qas:
        key = (row.get("scene_id", ""), row.get("task", ""), row.get("question", "").strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def _resolve_erp_size(scene_obj: dict[str, Any]) -> tuple[int, int] | None:
    erp_path = str(scene_obj.get("erp_path", "")).strip()
    if not erp_path:
        return None
    p = Path(erp_path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def process_scene(scene_obj: dict[str, Any], rng: random.Random, args: argparse.Namespace, vlm: Any | None) -> list[dict[str, Any]]:
    scene_id, viewpoint_id = _scene_identity(scene_obj)
    all_entities = scene_obj.get("entities", [])
    valid_entities = [
        e
        for e in all_entities
        if is_valid_entity(e, min_sem_conf=args.min_semantic_confidence, min_depth_valid=args.min_depth_valid_ratio_soft)
    ]
    if not valid_entities:
        return []

    erp_size = _resolve_erp_size(scene_obj)
    qas: list[dict[str, Any]] = []
    qas.extend(build_basic_qa(valid_entities, scene_id, viewpoint_id, rng, args))
    qas.extend(build_omni_qa(valid_entities, scene_id, viewpoint_id, rng, args, erp_size))
    qas.extend(build_3d_qa(valid_entities, scene_id, viewpoint_id, rng, args))

    if args.enable_llm_open_tasks:
        qas.extend(build_llm_open_qa(valid_entities, scene_obj, scene_id, viewpoint_id, vlm, args, rng))

    qas = _dedup_qas(qas)
    if args.max_qas_per_scene > 0 and len(qas) > args.max_qas_per_scene:
        rng.shuffle(qas)
        qas = qas[: args.max_qas_per_scene]
    return qas


def _load_scenes(args: argparse.Namespace) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    if args.json_in:
        scenes.append(load_json(args.json_in))

    if args.index_views:
        index_obj = load_json(args.index_views)
        for item in index_obj.get("items", []):
            scene_id = str(item.get("scene_id", ""))
            viewpoint_id = str(item.get("viewpoint_id", ""))
            entities_path = Path(args.entities_root) / scene_id / viewpoint_id / args.entities_filename
            if not entities_path.exists():
                continue
            obj = load_json(entities_path)
            obj.setdefault("scene_id", scene_id)
            obj.setdefault("viewpoint_id", viewpoint_id)
            if not obj.get("erp_path"):
                views_json = str(item.get("views_json", ""))
                if views_json and Path(views_json).exists():
                    try:
                        views_obj = load_json(views_json)
                        obj["erp_path"] = views_obj.get("erp_path", "")
                    except Exception:
                        pass
            scenes.append(obj)

    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    return scenes


def _split_rows(rows: list[dict[str, Any]], rng: random.Random, train_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * max(0.0, min(1.0, float(train_ratio))))
    return shuffled[:split_idx], shuffled[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_in", default="", help="Single entities json input (e.g. entities_with_depth.json)")
    parser.add_argument("--index_views", default="", help="Index json for batch mode")
    parser.add_argument("--entities_root", default="results/05c_depth_spatial", help="Entities root for index mode")
    parser.add_argument("--entities_filename", default="entities_with_depth.json", help="Entities filename under scene/viewpoint")
    parser.add_argument("--out_jsonl", required=True, help="Output QA jsonl path")
    parser.add_argument("--out_benchmark_jsonl", default="", help="Optional: output high-reliability benchmark subset")
    parser.add_argument("--split_out_dir", default="", help="Optional: write train/val jsonl split under this folder")

    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--max_qas_per_scene", type=int, default=20)

    parser.add_argument("--min_semantic_confidence", type=float, default=0.45)
    parser.add_argument("--min_depth_valid_ratio_soft", type=float, default=0.05)
    parser.add_argument("--min_depth_valid_ratio", type=float, default=0.2, help="Hard threshold for depth tasks")
    parser.add_argument("--min_relative_depth_gap", type=float, default=0.15)

    parser.add_argument("--max_grounding_qas", type=int, default=4)
    parser.add_argument("--max_text_to_region_qas", type=int, default=2)
    parser.add_argument("--max_existence_positive_qas", type=int, default=2)
    parser.add_argument("--max_existence_negative_qas", type=int, default=2)
    parser.add_argument("--max_counting_qas", type=int, default=2)
    parser.add_argument("--max_counting_value", type=int, default=8)

    parser.add_argument("--max_direction_qas", type=int, default=3)
    parser.add_argument("--max_relative_direction_qas", type=int, default=2)
    parser.add_argument("--max_boundary_qas", type=int, default=1)
    parser.add_argument("--max_distortion_qas", type=int, default=1)
    parser.add_argument("--boundary_margin_px", type=int, default=20)
    parser.add_argument("--polar_pitch_deg", type=float, default=55.0)

    parser.add_argument("--max_distance_qas", type=int, default=3)
    parser.add_argument("--max_relative_depth_qas", type=int, default=2)

    parser.add_argument("--enable_llm_open_tasks", action="store_true", help="Enable LLM-generated open-ended QA")
    parser.add_argument("--cfg", default="", help="Config path for semantic VLM (required when enabling LLM tasks)")
    parser.add_argument("--max_llm_scene_qas", type=int, default=1)
    parser.add_argument("--max_llm_distortion_qas", type=int, default=1)

    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    if not args.json_in and not args.index_views:
        raise SystemExit("Provide --json_in or --index_views")
    if args.enable_llm_open_tasks and not args.cfg:
        raise SystemExit("--cfg is required when --enable_llm_open_tasks is set")

    rng = random.Random(args.seed)
    scenes = _load_scenes(args)
    if not scenes:
        raise SystemExit("No valid scenes found from input")

    vlm = None
    if args.enable_llm_open_tasks:
        cfg = load_cfg(args.cfg)
        vlm = build_vlm(cfg)

    all_qas: list[dict[str, Any]] = []
    for scene_obj in scenes:
        scene_qas = process_scene(scene_obj, rng, args, vlm)
        all_qas.extend(scene_qas)

    all_qas = _dedup_qas(all_qas)
    dump_jsonl(args.out_jsonl, all_qas)

    if args.out_benchmark_jsonl:
        benchmark_rows = [row for row in all_qas if row.get("reliability") == "high" and row.get("source") == "rule"]
        dump_jsonl(args.out_benchmark_jsonl, benchmark_rows)

    if args.split_out_dir:
        split_dir = Path(args.split_out_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        train_rows, val_rows = _split_rows(all_qas, rng, args.train_ratio)
        dump_jsonl(split_dir / "train.jsonl", train_rows)
        dump_jsonl(split_dir / "val.jsonl", val_rows)

    stats = Counter((row.get("task_level", ""), row.get("task", ""), row.get("source", "")) for row in all_qas)
    print(f"scenes={len(scenes)}, qas={len(all_qas)}, out={args.out_jsonl}")
    for (lvl, task, src), cnt in sorted(stats.items()):
        print(f"  - {lvl}/{task}/{src}: {cnt}")


if __name__ == "__main__":
    main()
