#!/usr/bin/env python
from __future__ import annotations

"""
05_region_semantic.py

Entity-centered semantic generation (context-only version):

- One entity => one task.
- Select one best perspective view per entity (fallback to ERP).
- Build one context crop centered around the target.
- Optionally draw a very thin visual hint box for target identification.
- Batch-call VLM and return semantic-only fields.

This script intentionally removes detail-crop input to keep semantics stable and
reduce model confusion in multi-image prompt settings.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.multiprocessing as mp

from _common import load_cfg

from erp_meta.crop_utils import crop_with_bbox_info, expand_bbox_xyxy, overlay_mask
from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.models.vlm_qwen import _extract_json
from erp_meta.pipeline import build_vlm
from erp_meta.rle import decode_binary_mask
from erp_meta.view_sampling import ViewSpec, view_to_erp_maps


def _apply_vlm_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    mcfg = cfg.setdefault("models", {}).setdefault("semantic_vlm", {})
    if args.vlm_backend:
        mcfg["backend"] = str(args.vlm_backend)
    if args.vlm_base_url:
        mcfg["base_url"] = str(args.vlm_base_url)
    if args.vlm_model_name:
        mcfg["model_name"] = str(args.vlm_model_name)
    if args.vlm_api_key:
        mcfg["api_key"] = str(args.vlm_api_key)


def _run_single(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.cfg)
    _apply_vlm_overrides(cfg, args)
    vlm = build_vlm(cfg)
    _run_single_with_vlm(args, vlm)


def _run_single_with_vlm(args: argparse.Namespace, vlm) -> None:

    src_json = args.entities_json or args.instance_vote_json
    obj = load_json(src_json)
    all_entities = obj.get("entities", [])
    if args.instance_vote_json:
        all_entities = _clusters_to_entities(obj)
    if not all_entities:
        if args.index_views:
            return
        raise SystemExit("No usable `entities` in input json.")

    selected_records = _select_entities(
        all_entities,
        entity_id=str(args.entity_id),
        entity_index=int(args.entity_index),
        max_entities=int(args.max_entities),
    )
    if not selected_records:
        if args.index_views:
            return
        raise SystemExit("No entities selected. Check --entity_id/--entity_index/--max_entities.")

    views_obj = None
    views_json = args.views_json or obj.get("views_json") or ""
    if views_json:
        try:
            views_obj = load_json(views_json)
        except Exception:
            views_obj = None

    erp_path = obj.get("erp_path") or (views_obj.get("erp_path") if views_obj is not None else "")
    if not erp_path:
        raise SystemExit("Cannot find `erp_path` in input json. Provide --views_json or use a newer 04c output.")
    erp_img = Image.open(erp_path).convert("RGB")

    prepared_views = _prepare_view_cache(views_obj.get("views", [])) if views_obj is not None else []
    semantic_batch_size = int(args.semantic_batch_size or getattr(vlm, "batch_size", 1) or 1)

    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir is not None:
        ensure_dir(viz_dir)

    tasks = _build_entity_tasks(
        selected_records,
        prepared_views,
        erp_img,
        mark_mode=str(args.mark_mode),
        context_scale=max(1.0, float(args.context_scale)),
        context_min_pad=max(0, int(args.context_min_pad)),
        erp_context_scale=max(1.0, float(args.erp_context_scale)),
        view_context_weight=max(0.0, float(args.view_context_weight)),
        view_min_object_ratio=max(0.0, float(args.view_min_object_ratio)),
        view_context_topk=max(0, int(args.view_context_topk)),
        prepare_workers=max(1, int(args.prepare_workers)),
    )

    if viz_dir is not None:
        viz_limit = int(getattr(args, "viz_limit", 0) or 0)
        viz_tasks = tasks if viz_limit <= 0 else tasks[:viz_limit]
        for task in viz_tasks:
            _save_entity_viz(viz_dir, task)

    fallback_single_count = 0
    for start in range(0, len(tasks), max(1, semantic_batch_size)):
        chunk = tasks[start : start + max(1, semantic_batch_size)]
        chunk_images = [task["images"] for task in chunk]
        chunk_prompts = [task["prompt"] for task in chunk]
        responses = vlm.chat_multi_image_batch(chunk_images, chunk_prompts)

        for task, response in zip(chunk, responses):
            parsed = _parse_semantic_response(response)
            raw_response = response
            if parsed is None:
                retry_text = vlm.chat_multi_image(
                    task["images"],
                    task["prompt"] + " Return exactly one JSON object only. Do not add markdown fences or extra text.",
                )
                raw_response = retry_text
                parsed = _parse_semantic_response(retry_text)
            if parsed is None:
                fallback_single_count += 1
                parsed = _fallback_single_entity(vlm, task)
                raw_response = json.dumps(parsed, ensure_ascii=False, indent=2)

            task["raw_response"] = raw_response
            task["semantic_parsed"] = parsed

    enriched_updates: dict[int, dict[str, Any]] = {}
    for task in tasks:
        orig_idx = int(task["orig_idx"])
        entity = dict(task["entity"])
        semantic_out = _normalize_semantic_output(task.get("semantic_parsed", {}), hint_label=str(entity.get("label_open", "")))

        entity["semantic"] = semantic_out
        entity["semantic_source"] = {
            "source": task["source"],
            "view_id": task["view_id"],
        }
        enriched_updates[orig_idx] = entity

    if viz_dir is not None:
        viz_limit = int(getattr(args, "viz_limit", 0) or 0)
        viz_tasks = tasks if viz_limit <= 0 else tasks[:viz_limit]
        for task in viz_tasks:
            _save_entity_response(viz_dir, task)

    single_case = bool(str(args.entity_id).strip()) or int(args.entity_index) >= 0
    if single_case:
        selected_indices = [int(row["orig_idx"]) for row in selected_records]
        out_entities = [enriched_updates[i] for i in selected_indices if i in enriched_updates]
    else:
        out_entities = []
        for idx, entity in enumerate(all_entities):
            out_entities.append(enriched_updates.get(idx, dict(entity)))

    out = dict(obj)
    out["entities"] = out_entities
    quality_stats = dict(obj.get("quality_stats", {}))
    quality_stats.update(
        {
            "semantic_entity_count": int(len(out_entities)),
            "semantic_mode": "entity_centered_context_only",
            "semantic_fallback_single_count": int(fallback_single_count),
            "semantic_prepare_workers": int(max(1, args.prepare_workers)),
            "semantic_batch_size": int(max(1, semantic_batch_size)),
        }
    )
    out["quality_stats"] = quality_stats

    out_path = Path(args.out_json)
    ensure_dir(out_path.parent)
    dump_json(out_path, out)
    print(f"semantic -> {out_path}")


def _run_index(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.cfg)
    _apply_vlm_overrides(cfg, args)
    vlm = build_vlm(cfg)
    index = load_json(args.index_views)
    items = index["items"]
    if args.start_scene:
        started = False
        filtered = []
        for it in items:
            scene_id = str(it.get("scene_id", ""))
            if not started:
                if scene_id == args.start_scene:
                    started = True
                else:
                    continue
            filtered.append(it)
        items = filtered
    if args.shard_count > 1:
        items = [it for i, it in enumerate(items) if (i % args.shard_count) == args.shard_id]

    inst_root = Path(args.instance_vote_root)
    out_root = Path(args.out_root)
    viz_root = Path(args.viz_root) if args.viz_root else None

    for it in items:
        scene_id = it["scene_id"]
        vp = it["viewpoint_id"]
        inst = inst_root / scene_id / vp / "instance_vote.json"
        if not inst.exists():
            continue
        out_json = out_root / scene_id / vp / "entities_enriched.json"
        if args.skip_existing and out_json.exists():
            continue

        viz_dir = ""
        if viz_root is not None:
            viz_path = viz_root / scene_id / vp
            ensure_dir(viz_path)
            viz_dir = str(viz_path)

        local_args = argparse.Namespace(**vars(args))
        local_args.instance_vote_json = str(inst)
        local_args.entities_json = ""
        local_args.views_json = it["views_json"]
        local_args.out_json = str(out_json)
        local_args.viz_dir = viz_dir

        _run_single_with_vlm(local_args, vlm)


def _run_worker(local_rank: int, args: argparse.Namespace) -> None:
    if args.num_gpus > 1:
        visible = _resolve_visible_devices(args)
        if visible:
            if local_rank >= len(visible):
                raise SystemExit(
                    f"local_rank {local_rank} exceeds CUDA_VISIBLE_DEVICES length {len(visible)}; "
                    "set --num_gpus to match the visible device count."
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible[local_rank])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(0)
        args.shard_id = local_rank
        args.shard_count = args.num_gpus

    if args.index_views:
        _run_index(args)
    else:
        _run_single(args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    src_group = ap.add_mutually_exclusive_group()
    src_group.add_argument("--entities_json", help="Merged entities json from 04_merge_entities.py")
    src_group.add_argument("--instance_vote_json", help="Instance vote json from 04c_instance_vote.py")
    src_group.add_argument("--index_views", help="index_views.json from 01_make_views.py")
    ap.add_argument("--out_json", required=False, default="")
    ap.add_argument("--views_json", default="", help="Optional override views.json")
    ap.add_argument("--out_root", default="", help="Output root for index mode")
    ap.add_argument("--viz_root", default="", help="Viz root for index mode")
    ap.add_argument("--instance_vote_root", default="results_test/04c_instance_vote", help="Instance vote root for index mode")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use via multiprocessing")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--start_scene", default="", help="Skip processing until this scene_id is reached in index_views")

    ap.add_argument("--max_entities", type=int, default=0, help="0=all selected entities")
    ap.add_argument("--entity_index", type=int, default=-1, help="Run single-case on this entity index (0-based)")
    ap.add_argument("--entity_id", default="", help="Run single-case on this entity_id")

    ap.add_argument("--semantic_batch_size", type=int, default=0, help="Override semantic VLM batch size; 0 uses config/model default")
    ap.add_argument("--prepare_workers", type=int, default=4, help="Workers used for view projection and crop preparation")

    ap.add_argument("--mark_mode", default="thin_box", choices=["thin_box", "mask", "none"], help="How to visually mark the target inside the context crop")
    ap.add_argument("--context_scale", type=float, default=4.0, help="Scale used to expand target bbox into context crop")
    ap.add_argument("--context_min_pad", type=int, default=0, help="Minimum padding around target for context crop")
    ap.add_argument("--erp_context_scale", type=float, default=5.0, help="ERP fallback context scale")

    ap.add_argument("--view_context_weight", type=float, default=0.2, help="Secondary weight for nearby context utility during best-view selection")
    ap.add_argument("--view_min_object_ratio", type=float, default=0.70, help="Candidates below this object-score ratio are filtered before context re-ranking")
    ap.add_argument("--view_context_topk", type=int, default=3, help="How many nearby objects contribute to context utility")

    ap.add_argument("--viz_dir", default="", help="Optional directory to save crop/prompt/response debug artifacts")
    ap.add_argument("--viz_limit", type=int, default=10, help="Max number of entities to visualize per view (0=all)")
    ap.add_argument("--vlm_backend", default="", help="Override semantic VLM backend: transformers/openai_compatible/vllm")
    ap.add_argument("--vlm_base_url", default="", help="Override semantic VLM base_url (OpenAI-compatible server)")
    ap.add_argument("--vlm_model_name", default="", help="Override semantic VLM model name for OpenAI-compatible server")
    ap.add_argument("--vlm_api_key", default="", help="Override semantic VLM API key for OpenAI-compatible server")
    args = ap.parse_args()
    if args.index_views:
        if not args.out_root:
            raise SystemExit("--out_root is required when using --index_views")
        if args.num_gpus > 1:
            mp.spawn(_run_worker, nprocs=args.num_gpus, args=(args,))
        else:
            _run_index(args)
        return

    if not args.entities_json and not args.instance_vote_json:
        raise SystemExit("Provide --entities_json or --instance_vote_json, or use --index_views")
    if not args.out_json:
        raise SystemExit("--out_json is required for single-case execution")

    if args.num_gpus > 1:
        mp.spawn(_run_worker, nprocs=args.num_gpus, args=(args,))
    else:
        _run_single(args)


def _resolve_visible_devices(args: argparse.Namespace) -> list[str]:
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env_value:
        return []
    return [v.strip() for v in env_value.split(",") if v.strip()]


def _clusters_to_entities(obj: dict[str, Any]) -> list[dict[str, Any]]:
    entities = obj.get("entities")
    if isinstance(entities, list) and entities:
        return [dict(row) for row in entities]
    clusters = obj.get("clusters")
    if isinstance(clusters, list) and clusters:
        kept = [dict(row) for row in clusters if bool(row.get("keep", True))]
        return kept
    return []


def _select_entities(
    entities: list[dict[str, Any]],
    *,
    entity_id: str,
    entity_index: int,
    max_entities: int,
) -> list[dict[str, Any]]:
    if not entities:
        return []

    if entity_id:
        selected = [dict(entity) for entity in entities if str(entity.get("entity_id", "")) == str(entity_id)]
        return [{"orig_idx": idx, "entity": ent} for idx, ent in enumerate(entities) if ent in selected]

    if entity_index >= 0:
        if entity_index >= len(entities):
            return []
        return [{"orig_idx": int(entity_index), "entity": dict(entities[int(entity_index)])}]

    limit = int(max_entities) if int(max_entities) > 0 else len(entities)
    return [{"orig_idx": idx, "entity": dict(ent)} for idx, ent in enumerate(entities[:limit])]


def _build_entity_tasks(
    selected_records: list[dict[str, Any]],
    prepared_views: list[dict[str, Any]],
    erp_img: Image.Image,
    *,
    mark_mode: str,
    context_scale: float,
    context_min_pad: int,
    erp_context_scale: float,
    view_context_weight: float,
    view_min_object_ratio: float,
    view_context_topk: int,
    prepare_workers: int,
) -> list[dict[str, Any]]:
    entities = [row["entity"] for row in selected_records]
    masks: list[np.ndarray] = []
    for entity in entities:
        mask_rle = entity.get("mask_rle")
        if not mask_rle:
            raise SystemExit("entity is missing mask_rle; ensure 04c output includes mask_rle.")
        masks.append(decode_binary_mask(mask_rle).astype(np.uint8))

    prepared_views_by_id = {str(item["view"].view_id): item for item in prepared_views}
    best_candidates = _build_entity_view_candidates(
        entities,
        masks,
        prepared_views,
        prepared_views_by_id,
        view_context_weight=view_context_weight,
        view_min_object_ratio=view_min_object_ratio,
        view_context_topk=view_context_topk,
    )

    tasks: list[dict[str, Any]] = []
    if prepare_workers <= 1 or len(entities) <= 1:
        for local_idx, row in enumerate(selected_records):
            tasks.append(
                _prepare_single_entity_task(
                    local_idx=local_idx,
                    orig_idx=int(row["orig_idx"]),
                    entity=row["entity"],
                    mask_erp01=masks[local_idx],
                    best_candidate=best_candidates[local_idx],
                    erp_img=erp_img,
                    mark_mode=mark_mode,
                    context_scale=context_scale,
                    context_min_pad=context_min_pad,
                    erp_context_scale=erp_context_scale,
                )
            )
        return tasks

    with ThreadPoolExecutor(max_workers=min(int(prepare_workers), len(entities))) as executor:
        futures = []
        for local_idx, row in enumerate(selected_records):
            futures.append(
                executor.submit(
                    _prepare_single_entity_task,
                    local_idx,
                    int(row["orig_idx"]),
                    row["entity"],
                    masks[local_idx],
                    best_candidates[local_idx],
                    erp_img,
                    mark_mode,
                    context_scale,
                    context_min_pad,
                    erp_context_scale,
                )
            )
        for future in futures:
            tasks.append(future.result())
    return tasks


def _build_entity_view_candidates(
    entities: list[dict[str, Any]],
    entity_masks: list[np.ndarray],
    prepared_views: list[dict[str, Any]],
    prepared_views_by_id: dict[str, dict[str, Any]],
    *,
    view_context_weight: float,
    view_min_object_ratio: float,
    view_context_topk: int,
) -> list[dict[str, Any] | None]:
    candidate_rows_by_entity: list[list[dict[str, Any]]] = []
    candidates_by_view: dict[str, list[dict[str, Any]]] = {}

    for ent_idx, entity in enumerate(entities):
        mask = entity_masks[ent_idx]
        candidate_items = _resolve_candidate_cached_views(entity, prepared_views, prepared_views_by_id)
        entity_candidates: list[dict[str, Any]] = []
        for item in candidate_items:
            vm = _project_cached_mask_to_view(mask, item)
            bbox = _mask_to_bbox_xyxy(vm)
            if bbox is None:
                continue
            candidate = {
                "ent_idx": ent_idx,
                "view_id": str(item["view"].view_id),
                "view": item["view"],
                "view_image": item["image"],
                "view_bbox": bbox,
                "view_mask": vm,
                "object_score": float(_score_object_view(vm, bbox)),
            }
            entity_candidates.append(candidate)
            candidates_by_view.setdefault(str(item["view"].view_id), []).append(candidate)
        candidate_rows_by_entity.append(entity_candidates)

    best_by_entity: list[dict[str, Any] | None] = []
    for entity_candidates in candidate_rows_by_entity:
        if not entity_candidates:
            best_by_entity.append(None)
            continue

        best_object_score = max(candidate["object_score"] for candidate in entity_candidates)
        admissible = [
            candidate
            for candidate in entity_candidates
            if candidate["object_score"] >= max(1e-6, best_object_score * max(0.0, view_min_object_ratio))
        ]
        if not admissible:
            admissible = entity_candidates

        best_candidate = None
        best_total_score = -1.0
        for candidate in admissible:
            context_score, neighbor_count = _estimate_view_context_score(
                candidate,
                candidates_by_view.get(candidate["view_id"], []),
                topk=max(0, view_context_topk),
            )
            total_score = float(candidate["object_score"]) + float(view_context_weight) * float(context_score)
            if total_score > best_total_score:
                best_total_score = total_score
                best_candidate = dict(candidate)
                best_candidate["context_score"] = float(context_score)
                best_candidate["neighbor_count"] = int(neighbor_count)
                best_candidate["total_score"] = float(total_score)
        best_by_entity.append(best_candidate)
    return best_by_entity


def _prepare_single_entity_task(
    local_idx: int,
    orig_idx: int,
    entity: dict[str, Any],
    mask_erp01: np.ndarray,
    best_candidate: dict[str, Any] | None,
    erp_img: Image.Image,
    mark_mode: str,
    context_scale: float,
    context_min_pad: int,
    erp_context_scale: float,
) -> dict[str, Any]:
    if best_candidate is None:
        source = "erp"
        view_id = "erp"
        base_image = erp_img
        target_bbox = tuple(map(int, entity.get("bbox_xyxy", [0, 0, 0, 0])))
        context_bbox = expand_bbox_xyxy(target_bbox, base_image.size, scale=erp_context_scale, min_pad=context_min_pad)
        mask_view = mask_erp01
        view_score = 0.0
        view_score_object = 0.0
        view_score_context = 0.0
        neighbor_count = 0
    else:
        source = "persp"
        view_id = str(best_candidate["view_id"])
        base_image = best_candidate["view_image"]
        target_bbox = tuple(map(int, best_candidate["view_bbox"]))
        context_bbox = expand_bbox_xyxy(target_bbox, base_image.size, scale=context_scale, min_pad=context_min_pad)
        mask_view = best_candidate["view_mask"]
        view_score = float(best_candidate.get("total_score", 0.0))
        view_score_object = float(best_candidate.get("object_score", 0.0))
        view_score_context = float(best_candidate.get("context_score", 0.0))
        neighbor_count = int(best_candidate.get("neighbor_count", 0))

    context_crop, context_crop_box, _ = crop_with_bbox_info(base_image, context_bbox, pad=0)
    local_bbox = _translate_bbox_to_crop(target_bbox, context_crop_box, context_crop.size)
    local_mask = _crop_mask(mask_view, context_crop_box)
    context_marked = _mark_context_crop(context_crop, local_bbox, local_mask, mark_mode)
    prompt = _build_entity_prompt()

    return {
        "local_idx": int(local_idx),
        "orig_idx": int(orig_idx),
        "entity": entity,
        "entity_id": entity.get("entity_id", f"E{orig_idx:06d}"),
        "source": source,
        "view_id": view_id,
        "view_score": float(view_score),
        "view_score_object": float(view_score_object),
        "view_score_context": float(view_score_context),
        "view_context_neighbor_count": int(neighbor_count),
        "mark_mode": str(mark_mode),
        "target_bbox_xyxy": [int(v) for v in target_bbox],
        "context_bbox_xyxy": [int(v) for v in context_bbox],
        "context_crop": context_crop,
        "context_marked": context_marked,
        "images": [context_marked],
        "prompt": prompt,
        "context_scale": float(context_scale),
    }


def _normalize_semantic_output(parsed: dict[str, Any], hint_label: str) -> dict[str, Any]:
    identify = str(
        parsed.get("identify", "")
        or parsed.get("semantic_type", "")
        or parsed.get("name_refined", "")
        or hint_label
        or "object"
    )
    caption_brief = str(parsed.get("caption_brief", "") or identify)
    caption_dense = str(parsed.get("caption_dense", "") or caption_brief)
    reground_query = str(parsed.get("reground_query", "") or caption_dense or caption_brief or identify)

    attributes = parsed.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}

    event_status = parsed.get("event_status", "")
    if isinstance(event_status, list):
        event_status = ", ".join(str(v) for v in event_status if str(v).strip())
    event_status = str(event_status or "").strip()

    confidence = parsed.get("confidence", parsed.get("semantic_confidence", 0.0))
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "identify": identify,
        "attributes": attributes,
        "event_status": event_status,
        "caption_brief": caption_brief,
        "caption_dense": caption_dense,
        "reground_query": reground_query,
        "confidence": confidence,
    }


def _resolve_candidate_cached_views(
    entity: dict[str, Any],
    prepared_views: list[dict[str, Any]],
    prepared_views_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred_view_id = str(entity.get("semantic_source", {}).get("view_id", ""))
    ordered_ids: list[str] = []
    if preferred_view_id and preferred_view_id != "erp":
        ordered_ids.append(preferred_view_id)
    for view_id in entity.get("source_views", []):
        view_id = str(view_id)
        if view_id and view_id not in ordered_ids:
            ordered_ids.append(view_id)

    if ordered_ids:
        resolved: list[dict[str, Any]] = []
        for view_id in ordered_ids:
            item = prepared_views_by_id.get(view_id)
            if item is not None:
                resolved.append(item)
        if resolved:
            return resolved
    return prepared_views


def _parse_semantic_response(text: str) -> dict[str, Any] | None:
    try:
        obj = _extract_json(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _fallback_single_entity(vlm, task: dict[str, Any]) -> dict[str, Any]:
    text = vlm.chat_multi_image(
        task["images"],
        task["prompt"] + " Return exactly one JSON object only. Do not add markdown fences or extra text.",
    )
    parsed = _parse_semantic_response(text)
    if parsed is not None:
        return parsed

    hint = str(task["entity"].get("label_open", "object") or "object")
    return {
        "identify": hint,
        "attributes": {},
        "event_status": "",
        "caption_brief": hint,
        "caption_dense": hint,
        "reground_query": hint,
        "confidence": 0.0,
    }


def _build_entity_prompt() -> str:
    return '''
        You are building semantic metadata for exactly ONE target object.

        Input:
        - You are given ONE image, which is an entity-centered context crop.
        - A thin rectangle may be drawn only to indicate the target object.
        - The rectangle itself is NOT part of the scene and NOT part of the object.

        Task:
        Describe only the object INSIDE the rectangle.
        Do not describe the surrounding scene, nearby objects, or background unless they are directly visible as part of the target object itself.

        Important constraints:
        - The target object is strictly the object enclosed by the rectangle.
        - Do NOT describe the rectangle border.
        - Stay object-centric.
        - Only describe what is visually supported by the image.
        - Do not infer hidden properties, invisible parts, or uncertain functions.
        - If uncertain, be conservative and lower confidence.

        Return STRICT JSON only with exactly these keys:
        {
        "identify": string,
        "attributes": object,
        "event_status": string,
        "caption_brief": string,
        "caption_dense": string,
        "reground_query": string,
        "confidence": number
        }

        Field definitions:
        - identify:
        The semantic type, category, or component name of the target object.
        Use the most specific visually justified name possible.

        - attributes:
        A structured object containing only visually observable properties of the target object.
        Include concise entries such as color, material, shape, parts, text, logo, pattern, condition, and other visible traits when available.
        Do not invent attributes that are not visible.

        - event_status:
        The visible event about the object, functional state, usage state, or condition of the object, if any.
        If no clear status is visible, return an empty string.

        - caption_brief:
        A short object-centric summary in one sentence.

        - caption_dense:
        A richer and more detailed object-centric description that integrates identity, visible attributes, notable parts, text/logo, event, and visible state/function.
        Keep it focused on the target object.

        - reground_query:
        A short, highly discriminative referring phrase for re-localizing the same object.
        It should be concise and distinctive.

        - confidence:
        A number from 0 to 1 indicating how confident you are in your understanding of the target object in this crop.

        Output JSON only.
    '''



def _save_entity_viz(viz_dir: Path, task: dict[str, Any]) -> None:
    ent_dir = viz_dir / f"{int(task['orig_idx']):04d}_{task['entity_id']}"
    ent_dir.mkdir(parents=True, exist_ok=True)
    task["context_marked"].save(ent_dir / "context_marked.jpg", quality=95)
    meta = {
        "entity_id": task["entity_id"],
        "source": task["source"],
        "view_id": task["view_id"],
        "target_bbox_xyxy": task["target_bbox_xyxy"],
        "context_bbox_xyxy": task["context_bbox_xyxy"],
        "view_score": task["view_score"],
        "view_score_object": task["view_score_object"],
        "view_score_context": task["view_score_context"],
        "view_context_neighbor_count": task["view_context_neighbor_count"],
        "label_open": task["entity"].get("label_open"),
        "mark_mode": task["mark_mode"],
    }
    (ent_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (ent_dir / "prompt.txt").write_text(task["prompt"], encoding="utf-8")


def _save_entity_response(viz_dir: Path, task: dict[str, Any]) -> None:
    ent_dir = viz_dir / f"{int(task['orig_idx']):04d}_{task['entity_id']}"
    ent_dir.mkdir(parents=True, exist_ok=True)
    (ent_dir / "response.txt").write_text(str(task.get("raw_response", "")), encoding="utf-8")
    (ent_dir / "semantic.json").write_text(json.dumps(task.get("semantic_parsed", {}), ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_view_cache(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cached = []
    for v in views:
        if v.get("view_type") not in ("persp", "persp_pair"):
            continue
        view = ViewSpec(**v)
        map_x, map_y = view_to_erp_maps(view)
        cached.append(
            {
                "view": view,
                "map_x": np.round(map_x).astype(np.int32),
                "map_y": np.round(map_y).astype(np.int32),
                "image": Image.open(view.image_path).convert("RGB"),
            }
        )
    return cached


def _project_cached_mask_to_view(mask_erp01: np.ndarray, cached_view: dict[str, Any]) -> np.ndarray:
    erp_h, erp_w = mask_erp01.shape[:2]
    ex = cached_view["map_x"] % erp_w
    ey = np.clip(cached_view["map_y"], 0, erp_h - 1)
    return mask_erp01[ey, ex].astype(np.uint8)


def _mark_context_crop(
    crop: Image.Image,
    local_bbox: tuple[int, int, int, int],
    local_mask: np.ndarray | None,
    mark_mode: str,
) -> Image.Image:
    if mark_mode == "mask" and local_mask is not None and local_mask.size > 0:
        return overlay_mask(crop, local_mask)
    if mark_mode == "none":
        return crop.copy()
    return _draw_thin_box(crop, local_bbox)


def _draw_thin_box(image: Image.Image, bbox_xyxy: tuple[int, int, int, int]) -> Image.Image:
    img = image.convert("RGB").copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    width, height = img.size
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    draw = ImageDraw.Draw(img)
    # Use a visible thin outline to avoid blending into bright backgrounds.
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 0, 0), width=2)
    return img


def _translate_bbox_to_crop(
    bbox_xyxy: tuple[int, int, int, int],
    crop_box_xyxy: tuple[int, int, int, int],
    crop_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cx1, cy1, _, _ = [int(v) for v in crop_box_xyxy]
    lx1 = x1 - cx1
    ly1 = y1 - cy1
    lx2 = x2 - cx1
    ly2 = y2 - cy1
    width, height = crop_size
    lx1 = max(0, min(width - 1, lx1))
    ly1 = max(0, min(height - 1, ly1))
    lx2 = max(lx1 + 1, min(width, lx2))
    ly2 = max(ly1 + 1, min(height, ly2))
    return int(lx1), int(ly1), int(lx2), int(ly2)


def _crop_mask(mask01: np.ndarray, crop_box_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = crop_box_xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    if mask01.shape[0] < y2 or mask01.shape[1] < x2:
        return None
    return mask01[y1:y2, x1:x2].astype(np.uint8)


def _score_object_view(mask_view01: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> float:
    area = float(mask_view01.sum())
    x1, y1, x2, y2 = bbox_xyxy
    height, width = mask_view01.shape[:2]
    border_margin = min(x1, y1, max(0, width - x2), max(0, height - y2))
    return area + 0.15 * float(border_margin)


def _estimate_view_context_score(
    target_candidate: dict[str, Any],
    view_candidates: list[dict[str, Any]],
    *,
    topk: int,
) -> tuple[float, int]:
    if topk <= 0:
        return 0.0, 0

    target_bbox = tuple(target_candidate["view_bbox"])
    image_w, image_h = target_candidate["view_image"].size
    diag = max(1.0, float(np.hypot(float(image_w), float(image_h))))
    contributions: list[float] = []

    for other in view_candidates:
        if int(other["ent_idx"]) == int(target_candidate["ent_idx"]):
            continue
        distance = _bbox_center_distance(target_bbox, tuple(other["view_bbox"]))
        distance_weight = max(0.0, 1.0 - float(distance) / (0.45 * diag))
        if distance_weight <= 0.0:
            continue
        clarity_bonus = min(1.0, float(other.get("object_score", 0.0)) / max(1.0, float(target_candidate.get("object_score", 1.0))))
        contributions.append(distance_weight * (0.75 + 0.25 * clarity_bonus))

    contributions.sort(reverse=True)
    kept = contributions[:topk]
    return float(sum(kept)), int(len(kept))


def _mask_to_bbox_xyxy(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask01)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    return float(np.hypot(acx - bcx, acy - bcy))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return 0.5 * float(x1 + x2), 0.5 * float(y1 + y2)


if __name__ == "__main__":
    main()
