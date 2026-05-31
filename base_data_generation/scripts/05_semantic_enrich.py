#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _common import load_cfg

from erp_meta.crop_utils import crop_with_bbox_info, draw_bbox_outline, overlay_mask
from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.pipeline import build_vlm
from erp_meta.rle import decode_binary_mask
from erp_meta.view_sampling import ViewSpec, view_to_erp_maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--entities_json", help="Merged entities json from 04_merge_entities.py")
    src_group.add_argument("--instance_vote_json", help="Instance vote json from 04c_instance_vote.py")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--views_json", default="", help="Optional override views.json; enables persp-view verification")
    ap.add_argument("--max_entities", type=int, default=0, help="0=all")
    ap.add_argument("--semantic_batch_size", type=int, default=0, help="Override semantic VLM batch size; 0 uses config/model default")
    ap.add_argument("--mark_mode", default="box", choices=["box", "mask", "none"], help="How to mark the target inside the semantic crop")
    ap.add_argument("--viz_dir", default="", help="Optional directory to save semantic crops/overlays")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    vlm = build_vlm(cfg)

    src_json = args.entities_json or args.instance_vote_json
    obj = load_json(src_json)

    views_obj = None
    views_json = args.views_json or obj.get("views_json") or ""
    if views_json:
        try:
            views_obj = load_json(views_json)
        except Exception:
            views_obj = None

    erp_path = obj.get("erp_path") or (views_obj.get("erp_path") if views_obj is not None else "")
    if not erp_path:
        raise SystemExit(
            "Cannot find `erp_path` in the input json. Please provide `--views_json` or rerun `04c_instance_vote.py` with the latest code and `--overwrite`."
        )
    erp_img = Image.open(erp_path).convert("RGB")

    entities = obj.get("entities", [])
    if args.instance_vote_json:
        entities = _clusters_to_entities(obj)
    if not entities:
        raise SystemExit(
            "No usable `entities` found in the input json. This usually means the `instance_vote.json` was produced by an older 04c format. Please rerun `scripts/04c_instance_vote.py` with the latest code and `--overwrite`, then run 05 again."
        )
    if args.max_entities:
        entities = entities[: args.max_entities]

    semantic_batch_size = int(args.semantic_batch_size or getattr(vlm, "batch_size", 1) or 1)

    enriched = []
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir is not None:
        ensure_dir(viz_dir)

    prepared_views = _prepare_view_cache(views_obj.get("views", [])) if views_obj is not None else []
    prepared_entities: list[dict[str, Any]] = []

    for ent_idx, e in enumerate(entities):
        bbox = tuple(map(int, e["bbox_xyxy"]))
        # `04c` 里保存的是 ERP 坐标系下的实例 mask（RLE 压缩格式）。
        # 这里先把它解码成 0/1 二值图，后面所有“选哪张透视图最适合看这个实例”
        # 都是基于这个 ERP mask 来做反投影，而不是只看一个 bbox。
        mask = decode_binary_mask(e["mask_rle"]).astype(np.uint8)

        # 优先从 01_make_views 生成的现有透视图里选一张“最适合描述这个实例”的 view。
        # 这里不会重新渲染新视角，只是在已有 persp / persp_pair 里挑一张。
        crop, mask_crop = None, None
        if prepared_views:
            try:
                # 做法是：把当前 ERP mask 投回每张候选透视图，
                # 选“投影面积最大，同时尽量不贴边”的那张。
                # 这样做的目标是尽量让目标本体完整、清晰地落在一张透视图里，
                # 便于语义识别；它不是 relation-aware 的挑选策略。
                view_pick = _pick_best_persp_view(prepared_views, mask)
                if view_pick is not None:
                    view, view_mask, view_score, view_img = view_pick
                    v_bbox = _mask_to_bbox_xyxy(view_mask)
                    if v_bbox is not None:
                        crop, _, local_bbox = crop_with_bbox_info(view_img, v_bbox, pad=32)
                        x1, y1, x2, y2 = v_bbox
                        cx1 = max(0, x1 - 32)
                        cy1 = max(0, y1 - 32)
                        cx2 = min(view_mask.shape[1], x2 + 32)
                        cy2 = min(view_mask.shape[0], y2 + 32)
                        # `mask_crop` 与 `crop` 对齐，用于后面叠 overlay 和可选 fidelity 验证。
                        mask_crop = view_mask[cy1:cy2, cx1:cx2]
                        target_local_bbox = local_bbox
                        e.setdefault("semantic_source", {})
                        e["semantic_source"].update({"view_id": view.view_id, "view_score": float(view_score), "source": "persp"})
            except Exception:
                crop, mask_crop = None, None

        if crop is None or mask_crop is None:
            # 如果没有任何透视图能把该实例较完整地覆盖，就退回 ERP 本身裁剪。
            # 这通常发生在：实例很小、跨缝、或投回透视图后太碎/太边缘。
            crop, _, target_local_bbox = crop_with_bbox_info(erp_img, bbox, pad=32)
            x1, y1, x2, y2 = bbox
            # 从 ERP mask 中截出与 crop 对齐的 mask_patch。
            cx1 = max(0, x1 - 32)
            cy1 = max(0, y1 - 32)
            cx2 = min(mask.shape[1], x2 + 32)
            cy2 = min(mask.shape[0], y2 + 32)
            mask_crop = mask[cy1:cy2, cx1:cx2]
            e.setdefault("semantic_source", {})
            e["semantic_source"].update({"source": "erp"})

        if args.mark_mode == "mask":
            marked_crop = overlay_mask(crop, mask_crop)
        elif args.mark_mode == "box":
            marked_crop = draw_bbox_outline(crop, target_local_bbox)
        else:
            marked_crop = crop
        if viz_dir is not None:
            _save_semantic_viz(viz_dir, ent_idx, e, crop, marked_crop)

        prepared_entities.append(
            {
                "entity": e,
                "ent_idx": ent_idx,
                "crop": crop,
                "mask_crop": mask_crop,
                "marked_crop": marked_crop,
                "hint_label": e.get("label_open", ""),
            }
        )

    semantic_outputs: list[dict[str, Any]] = []
    for start in range(0, len(prepared_entities), max(1, semantic_batch_size)):
        chunk = prepared_entities[start : start + max(1, semantic_batch_size)]
        chunk_overlays = [row["marked_crop"] for row in chunk]
        chunk_labels = [str(row["hint_label"]) for row in chunk]
        semantic_outputs.extend(vlm.entity_enrich_batch(chunk_overlays, chunk_labels))

    for row, semantic in zip(prepared_entities, semantic_outputs):
        e = row["entity"]

        e2 = dict(e)
        e2["semantic"] = {
            "name_refined": semantic.get("name_refined", e.get("label_open", "")),
            "semantic_type": semantic.get("semantic_type", "object"),
            "attributes": semantic.get("attributes", {}),
            "caption_brief": semantic.get("caption_brief", ""),
            "caption_dense": semantic.get("caption_dense", ""),
            "affordances": semantic.get("affordances", []),
            "semantic_confidence": float(semantic.get("semantic_confidence", 0.0)),
        }
        e2["referring"] = {
            "short": semantic.get("local_ref_short", semantic.get("name_refined", e.get("label_open", ""))),
            "full": semantic.get("local_ref_full", semantic.get("caption_brief", "")),
            "local_cues": semantic.get("local_cues", []),
            "salient_parts": semantic.get("salient_parts", []),
        }
        enriched.append(e2)

    out = dict(obj)
    out["entities"] = enriched
    quality_stats = dict(obj.get("quality_stats", {}))
    quality_stats.update({"semantic_entity_count": len(enriched)})
    out["quality_stats"] = quality_stats
    dump_json(args.out_json, out)
    print(f"enriched={len(enriched)} -> {args.out_json}")


def _clusters_to_entities(obj: dict) -> list[dict]:
    entities = []
    for idx, entity in enumerate(obj.get("entities", [])):
        row = dict(entity)
        row.setdefault("entity_id", f"E{idx:06d}")
        row.setdefault("bbox_xyxy", row.get("bbox_erp", [0, 0, 0, 0]))
        row.setdefault("source_views", row.get("source_views", []))
        row.setdefault("confidence", float(row.get("confidence", row.get("best_score", 0.0))))
        entities.append(row)
    return entities


def _save_semantic_viz(viz_dir: Path, ent_idx: int, entity: dict, crop: Image.Image, marked: Image.Image) -> None:
    ent_dir = viz_dir / f"{ent_idx:04d}_{entity.get('entity_id', 'entity')}"
    ent_dir.mkdir(parents=True, exist_ok=True)
    crop.save(ent_dir / "crop.jpg", quality=95)
    marked.save(ent_dir / "marked.jpg", quality=95)
    meta = {
        "entity_id": entity.get("entity_id"),
        "label_open": entity.get("label_open"),
        "bbox_xyxy": entity.get("bbox_xyxy"),
        "source_views": entity.get("source_views", []),
        "confidence": entity.get("confidence", 0.0),
    }
    (ent_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_to_bbox_xyxy(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask01)
    if ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return (x1, y1, x2, y2)


def _project_erp_mask_to_view(mask_erp01: np.ndarray, view: ViewSpec) -> np.ndarray:
    map_x, map_y = view_to_erp_maps(view)
    erp_h, erp_w = mask_erp01.shape[:2]
    ex = np.round(map_x).astype(np.int32) % erp_w
    ey = np.clip(np.round(map_y).astype(np.int32), 0, erp_h - 1)
    return mask_erp01[ey, ex].astype(np.uint8)


def _prepare_view_cache(views: list[dict]) -> list[dict[str, Any]]:
    cached = []
    for v in views:
        if v.get("view_type") not in ("persp", "persp_pair"):
            continue
        view = ViewSpec(**v)
        # 预先缓存每张透视图到 ERP 的映射关系和图像本身，
        # 避免每个 entity 都重复做一遍 view_to_erp_maps() 和读图。
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


def _pick_best_persp_view(views: list[dict[str, Any]], mask_erp01: np.ndarray) -> tuple[ViewSpec, np.ndarray, float, Image.Image] | None:
    best = None
    best_score = -1.0
    erp_h, erp_w = mask_erp01.shape[:2]
    for item in views:
        view = item["view"]
        ex = item["map_x"] % erp_w
        ey = np.clip(item["map_y"], 0, erp_h - 1)
        # `vm` 表示：当前 ERP 实例在这张透视图里会落到哪些像素位置。
        vm = mask_erp01[ey, ex].astype(np.uint8)
        area = int(vm.sum())
        bbox = _mask_to_bbox_xyxy(vm)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        h, w = vm.shape[:2]
        border_margin = min(x1, y1, max(0, w - x2), max(0, h - y2))
        # 当前打分只偏向“目标投影面积大 + 不要太贴边”，
        # 因此它本质是 object-centric 的选择标准。
        # 如果后面要生成 relation-aware 的 referring expression，
        # 这里就需要换成“保留足够上下文”的另一套选图/裁图策略。
        score = float(area) + 0.15 * float(border_margin)
        if score > best_score:
            best_score = score
            best = (view, vm, score, item["image"])
    return best


if __name__ == "__main__":
    main()
