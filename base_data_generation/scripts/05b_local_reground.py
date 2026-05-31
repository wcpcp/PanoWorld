#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import json
import importlib
import sys
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp

from _common import load_cfg

from erp_meta.crop_utils import draw_bbox_outline
from erp_meta.io_utils import dump_json, ensure_dir, load_json
from erp_meta.rle import decode_binary_mask
from erp_meta.view_sampling import ViewSpec, view_to_erp_maps


def _run_single(args: argparse.Namespace, backend: dict[str, Any]) -> None:
    ctx = _prepare_scene_context(args)
    if ctx["pending_jobs"]:
        _run_wedetect_ref_requests(
            requests=ctx["pending_jobs"],
            work_dir=ensure_dir(Path(args.out_json).with_suffix(".tmp_reground")),
            batch_size=max(1, int(args.ref_batch_size)),
            num_workers=max(1, int(args.ref_workers)),
            backend=backend,
        )
    _finalize_scene_context(ctx)


def _load_ref_backend(cfg_path: str) -> tuple[str, str, str]:
    cfg = load_cfg(cfg_path)
    reg_cfg = cfg.get("models", {}).get("wedetect_ref", {})
    ref_script = reg_cfg.get("script", "")
    ref_ckpt = reg_cfg.get("wedetect_ref_checkpoint", "")
    uni_ckpt = reg_cfg.get("wedetect_uni_checkpoint", "")
    if not ref_script or not ref_ckpt or not uni_ckpt:
        raise SystemExit(
            "`models.wedetect_ref` is incomplete in configs/default.json. "
            "Please set `script`, `wedetect_ref_checkpoint`, and `wedetect_uni_checkpoint`."
        )
    return str(ref_script), str(ref_ckpt), str(uni_ckpt)


def _load_wedetect_backend(cfg_path: str) -> dict[str, Any]:
    _, ref_ckpt, uni_ckpt = _load_ref_backend(cfg_path)
    module = _import_wedetect_infer()
    det_model = module._load_detector(uni_ckpt)
    model, processor, object_token_index = module._load_grounding_model(ref_ckpt)
    return {
        "module": module,
        "det_model": det_model,
        "model": model,
        "processor": processor,
        "object_token_index": object_token_index,
    }


def _import_wedetect_infer():
    base_dir = Path(__file__).resolve().parents[2]
    wedetect_dir = base_dir / "WeDetect"
    if wedetect_dir.exists():
        sys.path.insert(0, str(wedetect_dir))
    return importlib.import_module("infer_wedetect_ref")


def _run_wedetect_ref_inproc(requests: list[dict[str, Any]], backend: dict[str, Any], batch_size: int) -> None:
    module = backend["module"]
    module._run_requests(
        requests,
        backend["det_model"],
        backend["model"],
        backend["processor"],
        backend["object_token_index"],
        batch_size,
        -1.0,
    )


def _run_index(args: argparse.Namespace, backend: dict[str, Any]) -> None:


    index = load_json(args.index_views)
    indexed_items = list(enumerate(index["items"]))
    # start_scene 支持
    if getattr(args, "start_scene", None):
        started = False
        filtered = []
        for i, it in indexed_items:
            scene_id = str(it.get("scene_id", ""))
            if not started:
                if scene_id == args.start_scene:
                    started = True
                else:
                    continue
            filtered.append((i, it))
        indexed_items = filtered
    # 分片支持
    if args.shard_count > 1:
        indexed_items = [(i, it) for i, it in indexed_items if (i % args.shard_count) == args.shard_id]

    ent_root = Path(args.entities_root)
    out_root = Path(args.out_root)
    viz_root = Path(args.viz_root) if args.viz_root else None
    shard_tmp_dir = ensure_dir(out_root / f".tmp_reground_shard_{int(args.shard_id):02d}")

    run_mode = str(getattr(args, "index_run_mode", "scene")).strip().lower()
    if run_mode not in ("scene", "merged"):
        raise SystemExit("--index_run_mode must be one of: scene, merged")

    print(
        f"[05b][shard {int(args.shard_id)}] start index run: "
        f"mode={run_mode}, total_items={len(indexed_items)}"
    )

    try:
        if run_mode == "scene":
            skipped_existing_count = 0
            prepared_count = 0
            finalized_count = 0
            total_requests = 0

            for global_idx, it in indexed_items:
                scene_id = it["scene_id"]
                vp = it["viewpoint_id"]
                entities_json = ent_root / scene_id / vp / "entities_enriched.json"
                if not entities_json.exists():
                    continue
                out_json = out_root / scene_id / vp / "entities_reground.json"
                # skip_existing 支持
                if getattr(args, "skip_existing", False) and out_json.exists():
                    skipped_existing_count += 1
                    continue

                viz_dir = ""
                should_viz = viz_root is not None and (
                    int(getattr(args, "viz_erp_limit", 0) or 0) <= 0
                    or int(global_idx) < int(getattr(args, "viz_erp_limit", 0))
                )
                if should_viz and viz_root is not None:
                    viz_path = viz_root / scene_id / vp / "viz"
                    ensure_dir(viz_path)
                    viz_dir = str(viz_path)

                local_args = argparse.Namespace(**vars(args))
                local_args.entities_json = str(entities_json)
                local_args.out_json = str(out_json)
                local_args.views_json = it["views_json"]
                local_args.viz_dir = viz_dir
                local_args.print_summary = bool(
                    int(getattr(args, "print_limit", 0) or 0) <= 0
                    or int(global_idx) < int(getattr(args, "print_limit", 0))
                )

                ctx = _prepare_scene_context(local_args)
                prepared_count += 1
                req_count = len(ctx["pending_jobs"])
                total_requests += req_count
                print(
                    f"[05b][shard {int(args.shard_id)}] prepared "
                    f"scene={scene_id}/{vp} requests={req_count}"
                )

                if req_count > 0:
                    _run_wedetect_ref_requests(
                        requests=ctx["pending_jobs"],
                        work_dir=shard_tmp_dir,
                        batch_size=max(1, int(args.ref_batch_size)),
                        num_workers=max(1, int(args.ref_workers)),
                        backend=backend,
                    )
                else:
                    print(f"[05b][shard {int(args.shard_id)}] no pending requests for {scene_id}/{vp}")

                _finalize_scene_context(ctx)
                finalized_count += 1
                print(
                    f"[05b][shard {int(args.shard_id)}] finalized "
                    f"scene={scene_id}/{vp} ({finalized_count}/{prepared_count})"
                )

            print(
                f"[05b][shard {int(args.shard_id)}] done: prepared={prepared_count} "
                f"finalized={finalized_count} requests={total_requests} "
                f"skipped_existing={skipped_existing_count}"
            )
            if shard_tmp_dir.exists():
                shutil.rmtree(shard_tmp_dir, ignore_errors=True)
            return

        scene_contexts: list[dict[str, Any]] = []
        merged_requests: list[dict[str, Any]] = []
        skipped_existing_count = 0

        for global_idx, it in indexed_items:
            scene_id = it["scene_id"]
            vp = it["viewpoint_id"]
            entities_json = ent_root / scene_id / vp / "entities_enriched.json"
            if not entities_json.exists():
                continue
            out_json = out_root / scene_id / vp / "entities_reground.json"
            if args.skip_existing and out_json.exists():
                skipped_existing_count += 1
                continue

            viz_dir = ""
            should_viz = viz_root is not None and (
                int(getattr(args, "viz_erp_limit", 0) or 0) <= 0
                or int(global_idx) < int(getattr(args, "viz_erp_limit", 0))
            )
            if should_viz and viz_root is not None:
                viz_path = viz_root / scene_id / vp / "viz"
                ensure_dir(viz_path)
                viz_dir = str(viz_path)

            local_args = argparse.Namespace(**vars(args))
            local_args.entities_json = str(entities_json)
            local_args.out_json = str(out_json)
            local_args.views_json = it["views_json"]
            local_args.viz_dir = viz_dir
            local_args.print_summary = bool(
                int(getattr(args, "print_limit", 0) or 0) <= 0
                or int(global_idx) < int(getattr(args, "print_limit", 0))
            )

            ctx = _prepare_scene_context(local_args)
            scene_contexts.append(ctx)
            merged_requests.extend(ctx["pending_jobs"])

        print(
            f"[05b][shard {int(args.shard_id)}] prepared_scenes={len(scene_contexts)} "
            f"requests={len(merged_requests)} skipped_existing={skipped_existing_count}"
        )

        if not scene_contexts:
            print(f"[05b][shard {int(args.shard_id)}] nothing to do (all skipped or missing inputs).")
            return

        if merged_requests:
            print(
                f"[05b][shard {int(args.shard_id)}] running WeDetect-Ref once "
                f"with merged requests, batch_size={int(args.ref_batch_size)}, workers={int(args.ref_workers)}"
            )
            _run_wedetect_ref_requests(
                requests=merged_requests,
                work_dir=shard_tmp_dir,
                batch_size=max(1, int(args.ref_batch_size)),
                num_workers=max(1, int(args.ref_workers)),
                backend=backend,
            )
        else:
            print(f"[05b][shard {int(args.shard_id)}] no pending requests after preparation.")

        for ctx in scene_contexts:
            _finalize_scene_context(ctx)

        if shard_tmp_dir.exists():
            shutil.rmtree(shard_tmp_dir, ignore_errors=True)

        print(f"[05b][shard {int(args.shard_id)}] finalize done.")
    finally:
        pass


def _prepare_scene_context(args: argparse.Namespace) -> dict[str, Any]:
    # obj = load_json(args.entities_json)
    try:
        obj = load_json(args.entities_json)
    except Exception as e:
        raise RuntimeError(f"Failed to load entities_json: {args.entities_json}") from e
    entities = obj.get("entities", [])
    if args.max_entities:
        entities = entities[: args.max_entities]

    views_json = args.views_json or obj.get("views_json") or ""
    if not views_json:
        raise SystemExit("`views_json` is required for local reground. Provide `--views_json` or ensure it exists in the input json.")
    views_obj = load_json(views_json)
    det_root = _resolve_det_root(args.det_root, views_json)
    filtered_view_lookup = _build_filtered_view_lookup(obj.get("views", []))
    prepared_views = _prepare_view_cache(views_obj.get("views", []))
    if not prepared_views:
        raise SystemExit("No usable `persp` or `persp_pair` views found for local reground.")
    prepared_views_by_id = {str(item["view"].view_id): item for item in prepared_views}

    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir is not None:
        ensure_dir(viz_dir)
    tmp_dir = ensure_dir(Path(args.out_json).with_suffix(".tmp_reground"))

    results_by_ent_idx: dict[int, dict[str, Any]] = {}
    consistency_ious: list[float] = []
    iou_views: list[float] = []
    pending_jobs: list[dict[str, Any]] = []
    no_view_count = 0
    empty_projection_count = 0
    missing_mask_count = 0

    for ent_idx, entity in enumerate(entities):
        mask = _decode_entity_mask(entity)
        if mask is None:
            entity2 = dict(entity)
            entity2["local_reground"] = {
                "status": "missing_mask",
                "mode": "full_view_referring",
                "consistency_iou": 0.0,
                "passed": False,
            }
            results_by_ent_idx[ent_idx] = entity2
            missing_mask_count += 1
            consistency_ious.append(0.0)
            iou_views.append(0.0)
            continue

        preferred_view_id = str(entity.get("semantic_source", {}).get("view_id", ""))
        candidate_views = _resolve_candidate_cached_views(entity, prepared_views, prepared_views_by_id)
        pick = _pick_context_view(candidate_views, mask, preferred_view_id)
        if pick is None:
            entity2 = dict(entity)
            entity2["local_reground"] = {
                "status": "no_view",
                "mode": "full_view_referring",
                "consistency_iou": 0.0,
                "passed": False,
            }
            results_by_ent_idx[ent_idx] = entity2
            no_view_count += 1
            consistency_ious.append(0.0)
            iou_views.append(0.0)
            continue

        view, view_mask, _ = pick
        v_bbox = _mask_to_bbox_xyxy(view_mask)
        if v_bbox is None:
            entity2 = dict(entity)
            entity2["local_reground"] = {
                "status": "empty_projection",
                "mode": "full_view_referring",
                "consistency_iou": 0.0,
                "passed": False,
            }
            results_by_ent_idx[ent_idx] = entity2
            empty_projection_count += 1
            consistency_ious.append(0.0)
            iou_views.append(0.0)
            continue

        query, query_source = _build_reground_query(entity, mode=str(args.query_mode))
        pred_path = tmp_dir / f"{ent_idx:04d}_{entity.get('entity_id', 'entity')}.json"
        pending_jobs.append(
            {
                "ent_idx": ent_idx,
                "entity": entity,
                "query": str(query),
                "query_source": str(query_source),
                "view_id": view.view_id,
                "context_scale": float(args.context_scale),
                "target_bbox_xyxy": tuple(int(v) for v in v_bbox),
                "image_path": str(view.image_path),
                "out_json": str(pred_path),
                "cluster_id": int(entity.get("cluster_id", -1)),
                "det_json": str(_get_view_det_json(det_root, obj.get("scene_id", ""), obj.get("viewpoint_id", ""), view.view_id)),
            }
        )

    return {
        "args": args,
        "obj": obj,
        "entities": entities,
        "det_root": det_root,
        "filtered_view_lookup": filtered_view_lookup,
        "results_by_ent_idx": results_by_ent_idx,
        "consistency_ious": consistency_ious,
        "iou_views": iou_views,
        "pending_jobs": pending_jobs,
        "no_view_count": no_view_count,
        "empty_projection_count": empty_projection_count,
        "missing_mask_count": missing_mask_count,
        "viz_dir": viz_dir,
        "tmp_dir": tmp_dir,
    }


def _finalize_scene_context(ctx: dict[str, Any]) -> None:
    args = ctx["args"]
    obj = ctx["obj"]
    entities = ctx["entities"]
    filtered_view_lookup = ctx["filtered_view_lookup"]
    results_by_ent_idx = ctx["results_by_ent_idx"]
    consistency_ious = ctx["consistency_ious"]
    iou_views = ctx["iou_views"]
    pending_jobs = ctx["pending_jobs"]
    no_view_count = int(ctx["no_view_count"])
    empty_projection_count = int(ctx["empty_projection_count"])
    missing_mask_count = int(ctx["missing_mask_count"])
    viz_dir = ctx["viz_dir"]

    for job in pending_jobs:
        preds = load_json(job["out_json"]) if Path(job["out_json"]).exists() else []
        det_rows = load_json(job["det_json"]) if job.get("det_json") and Path(job["det_json"]).exists() else []

        pred_bbox = None
        pred_score = 0.0
        if preds:
            pred_bbox = tuple(map(float, preds[0].get("bbox", [0, 0, 0, 0])))
            pred_score = float(preds[0].get("score", 0.0))

        filtered_ref = _pick_filtered_reference_box(
            filtered_view_lookup.get(str(job["view_id"]), []),
            cluster_id=int(job.get("cluster_id", -1)),
            target_bbox=job["target_bbox_xyxy"],
        )
        best_det = _pick_best_detect_box(job["target_bbox_xyxy"], det_rows)

        ref_source = "04c_filtered_view"
        ref_row = filtered_ref
        if ref_row is None:
            ref_source = "02_detect_fallback"
            ref_row = best_det

        ref_bbox = tuple(map(float, ref_row.get("bbox", []))) if ref_row is not None else None
        ref_score = float(ref_row.get("score", 0.0)) if ref_row is not None else 0.0
        ref_iou_target = _bbox_iou_xyxy(job["target_bbox_xyxy"], ref_bbox) if ref_bbox is not None else 0.0

        best_det_bbox = tuple(map(float, best_det.get("bbox", []))) if best_det is not None else None
        best_det_score = float(best_det.get("score", 0.0)) if best_det is not None else 0.0
        best_det_iou_target = _bbox_iou_xyxy(job["target_bbox_xyxy"], best_det_bbox) if best_det_bbox is not None else 0.0

        iou_view = _bbox_iou_xyxy(job["target_bbox_xyxy"], pred_bbox) if pred_bbox is not None else 0.0
        iou_pred_vs_ref = _bbox_iou_xyxy(ref_bbox, pred_bbox) if pred_bbox is not None and ref_bbox is not None else 0.0
        iou_pred_vs_best_det = _bbox_iou_xyxy(best_det_bbox, pred_bbox) if pred_bbox is not None and best_det_bbox is not None else 0.0
        pass_iou = iou_pred_vs_ref if ref_bbox is not None else iou_view
        passed = pass_iou >= float(args.min_reground_iou)

        entity2 = dict(job["entity"])
        local_reground = {
            "status": "ok" if pred_bbox is not None else "no_prediction",
            "backend": "wedetect_ref",
            "mode": "full_view_referring",
            "query": job["query"],
            "query_source": job.get("query_source", ""),
            "view_id": job["view_id"],
            "target_bbox_xyxy": [int(v) for v in job["target_bbox_xyxy"]],
            "pred_bbox_xyxy": [float(v) for v in pred_bbox] if pred_bbox is not None else [],
            "pred_score": float(pred_score),
            "consistency_iou": float(pass_iou),
            "passed": bool(passed),
        }
        if bool(getattr(args, "save_debug_details", False)):
            local_reground["iou_view"] = float(iou_view)
            local_reground["context_scale"] = float(job["context_scale"])
            local_reground["proposal_consistency"] = {
                "reference_source": ref_source,
                "reference_bbox_xyxy": [float(v) for v in ref_bbox] if ref_bbox is not None else [],
                "reference_score": float(ref_score),
                "reference_iou_target": float(ref_iou_target),
                "best_det_bbox_xyxy": [float(v) for v in best_det_bbox] if best_det_bbox is not None else [],
                "best_det_score": float(best_det_score),
                "best_det_iou_target": float(best_det_iou_target),
                "pred_iou_reference": float(iou_pred_vs_ref),
                "pred_iou_best_det": float(iou_pred_vs_best_det),
                "det_json": job.get("det_json", ""),
            }
        entity2["local_reground"] = local_reground
        iou_views.append(float(iou_view))
        consistency_ious.append(float(pass_iou))
        results_by_ent_idx[int(job["ent_idx"])] = entity2

    kept_entities = []
    for ent_idx, entity in enumerate(entities):
        entity2 = results_by_ent_idx.get(ent_idx)
        if entity2 is None:
            entity2 = dict(entity)
        passed = bool(entity2.get("local_reground", {}).get("passed", False))
        if passed or not args.drop_failed:
            kept_entities.append(entity2)

    out = dict(obj)
    out["entities"] = kept_entities
    quality_stats = dict(obj.get("quality_stats", {}))
    quality_stats.update(
        {
            "local_reground_iou_view_mean": float(np.mean(iou_views)) if iou_views else 0.0,
            "local_reground_iou_view_min": float(np.min(iou_views)) if iou_views else 0.0,
            "local_reground_consistency_iou_mean": float(np.mean(consistency_ious)) if consistency_ious else 0.0,
            "local_reground_consistency_iou_min": float(np.min(consistency_ious)) if consistency_ious else 0.0,
            "local_reground_request_count": int(len(pending_jobs)),
            "local_reground_no_view_count": int(no_view_count),
            "local_reground_empty_projection_count": int(empty_projection_count),
            "local_reground_missing_mask_count": int(missing_mask_count),
            "local_reground_yellow_box_count": int(
                sum(
                    1
                    for entity in results_by_ent_idx.values()
                    if len(entity.get("local_reground", {}).get("proposal_consistency", {}).get("reference_bbox_xyxy", [])) == 4
                )
            ),
            "local_reground_yellow_box_rate": float(
                sum(
                    1
                    for entity in results_by_ent_idx.values()
                    if len(entity.get("local_reground", {}).get("proposal_consistency", {}).get("reference_bbox_xyxy", [])) == 4
                )
            ) / float(max(len(pending_jobs), 1)),
            "local_reground_ref_batch_size": int(max(1, args.ref_batch_size)),
            "local_reground_ref_workers": int(max(1, args.ref_workers)),
            "local_reground_det_root": str(ctx["det_root"]),
            "local_reground_query_mode": str(args.query_mode),
        }
    )
    out["quality_stats"] = quality_stats
    dump_json(args.out_json, out)
    if bool(getattr(args, "print_summary", True)):
        print(f"local_reground={len(kept_entities)} -> {args.out_json}")

    if viz_dir is not None:
        _save_filtered_erp_viz(viz_dir, obj, kept_entities)

    tmp_dir = ctx.get("tmp_dir")
    if tmp_dir and hasattr(tmp_dir, "exists") and tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

def _resolve_visible_devices(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "cuda_ids", "") or "").strip()
    if not raw:
        env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_value:
            return [v.strip() for v in env_value.split(",") if v.strip()]
        return []
    return [v.strip() for v in raw.split(",") if v.strip()]


def _run_worker(local_rank: int, args: argparse.Namespace) -> None:
    if args.num_gpus > 1:
        visible = _resolve_visible_devices(args)
        if visible:
            if local_rank >= len(visible):
                raise SystemExit(
                    f"local_rank {local_rank} exceeds cuda_ids length {len(visible)}; "
                    "set --num_gpus to match or update --cuda_ids."
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible[local_rank])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(0)
        args.shard_id = local_rank
        args.shard_count = args.num_gpus

    backend = _load_wedetect_backend(args.cfg)
    if args.index_views:
        _run_index(args, backend)
    else:
        _run_single(args, backend)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--entities_json", default="")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--views_json", default="")
    ap.add_argument("--index_views", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--viz_root", default="")
    ap.add_argument("--entities_root", default="results_test/05_semantic_output1")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--start_scene", default="", help="Skip processing until this scene_id is reached in index_views")
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument(
        "--cuda_ids",
        default="",
        help="Optional comma-separated CUDA device IDs (e.g., '6,7'). Overrides CUDA_VISIBLE_DEVICES mapping.",
    )
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--det_root", default="", help="Optional 02_detect output root. If omitted, infer from `views_json` by replacing `01_make_views_output` with `02_detect_output`.")
    ap.add_argument("--context_scale", type=float, default=3, help="Deprecated compatibility argument. The latest 05b uses full selected view for regrounding instead of a local crop.")
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--min_reground_iou", type=float, default=0.0)
    ap.add_argument(
        "--query_mode",
        default="reground_query",
        choices=["hybrid", "reground_query", "referring_full", "referring_short", "semantic_name", "caption_dense", "caption_brief", "label_open"],
        help="Which text field from 05 is used as the WeDetect-Ref query. Default is `reground_query`, then fallback is controlled by `hybrid` priority.",
    )
    ap.add_argument("--ref_batch_size", type=int, default=8, help="Batch size used inside the external WeDetect-Ref script")
    ap.add_argument("--ref_workers", type=int, default=1, help="Number of WeDetect-Ref workers. In-process mode effectively runs with 1.")
    ap.add_argument(
        "--index_run_mode",
        default="scene",
        choices=["scene", "merged"],
        help="Index mode scheduling strategy: `scene` (default, process one scene at a time with immediate logs) or `merged` (prepare all then run merged requests once per shard).",
    )
    ap.add_argument("--drop_failed", action="store_true")
    ap.add_argument("--viz_dir", default="")
    ap.add_argument("--save_debug_details", action="store_true", help="Keep extra debug fields in local_reground (proposal_consistency, det paths, etc.).")
    ap.add_argument("--viz_erp_limit", type=int, default=10, help="In index mode, only save visualization for the first N ERP items globally. 0 means no limit.")
    ap.add_argument("--print_limit", type=int, default=10, help="In index mode, only print summary lines for the first N ERP items globally. 0 means no limit.")
    args = ap.parse_args()

    if args.index_views:
        if not args.out_root:
            raise SystemExit("--out_root is required when using --index_views")
        if args.num_gpus > 1:
            mp.spawn(_run_worker, nprocs=args.num_gpus, args=(args,))
        else:
            backend = _load_wedetect_backend(args.cfg)
            _run_index(args, backend)
        return

    if not args.entities_json or not args.out_json:
        raise SystemExit("Provide --entities_json and --out_json, or use --index_views")

    if args.num_gpus > 1:
        mp.spawn(_run_worker, nprocs=args.num_gpus, args=(args,))
    else:
        backend = _load_wedetect_backend(args.cfg)
        _run_single(args, backend)


def _run_wedetect_ref_requests(
    requests: list[dict[str, Any]],
    work_dir: Path,
    batch_size: int,
    num_workers: int,
    backend: dict[str, Any],
) -> None:
    if not requests:
        return
    if num_workers > 1:
        num_workers = 1

    _run_wedetect_ref_inproc(requests, backend, batch_size)


def _bbox_iou_xyxy(a: tuple[int, int, int, int], b: tuple[float, float, float, float] | None) -> float:
    # 这里只做最简单的 bbox IoU，作为局部一致性分数。
    # 不再引入 mask IoU / segmentation / ERP 回投影，保持速度和解释性。
    if b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _mask_to_bbox_xyxy(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    # 把 0/1 mask 转成 bbox，供紧目标框 / context 框构造使用。
    ys, xs = np.nonzero(mask01)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _decode_entity_mask(entity: dict[str, Any]) -> np.ndarray | None:
    mask_rle = entity.get("mask_rle")
    if not mask_rle:
        return None
    try:
        return decode_binary_mask(mask_rle).astype(np.uint8)
    except Exception:
        return None


def _prepare_view_cache(views: list[dict]) -> list[dict[str, Any]]:
    cached = []
    for row in views:
        if row.get("view_type") not in ("persp", "persp_pair"):
            continue
        view = ViewSpec(**row)
        # 预缓存透视图 -> ERP 的投影映射和原始图像，
        # 避免每个 entity 都重复计算 `view_to_erp_maps()` 和重复读图。
        map_x, map_y = view_to_erp_maps(view)
        cached.append(
            {
                "view": view,
                "map_x": np.round(map_x).astype(np.int32),
                "map_y": np.round(map_y).astype(np.int32),
            }
        )
    return cached


def _build_filtered_view_lookup(views_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in views_rows:
        view_id = str(row.get("view_id", ""))
        if not view_id:
            continue
        dets = row.get("detections", [])
        if not isinstance(dets, list):
            continue
        lookup[view_id] = [dict(det) for det in dets]
    return lookup


def _build_reground_query(entity: dict[str, Any], mode: str = "hybrid") -> tuple[str, str]:
    referring = entity.get("referring", {}) or {}
    semantic = entity.get("semantic", {}) or {}
    candidates = {
        "reground_query": _normalize_query_text(semantic.get("reground_query", "")),
        "caption_dense": _normalize_query_text(semantic.get("caption_dense", "")),
        "caption_brief": _normalize_query_text(semantic.get("caption_brief", "")),
        "referring_full": _normalize_query_text(referring.get("full", "")),
        "referring_short": _normalize_query_text(referring.get("short", "")),
        # Backward compatibility for old 05 outputs.
        "semantic_name": _normalize_query_text(
            semantic.get("identify", "") or semantic.get("name_refined", "") or semantic.get("semantic_type", "")
        ),
        "label_open": _normalize_query_text(entity.get("label_open", "")),
    }

    if mode != "hybrid":
        text = candidates.get(mode, "") or candidates.get("label_open", "") or "object"
        source = mode if candidates.get(mode, "") else ("label_open" if candidates.get("label_open", "") else "fallback_object")
        return text, source

    ordered_sources = [
        "reground_query",
        "caption_dense",
        "caption_brief",
        "referring_full",
        "referring_short",
        "semantic_name",
        "label_open",
    ]
    for source in ordered_sources:
        text = candidates.get(source, "")
        if text:
            return text, source
    return "object", "fallback_object"


def _normalize_query_text(text: Any) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:220]


def _pick_filtered_reference_box(
    view_rows: list[dict[str, Any]],
    *,
    cluster_id: int,
    target_bbox: tuple[int, int, int, int],
) -> dict[str, Any] | None:
    kept_same_cluster = [
        row
        for row in view_rows
        if int(row.get("cluster_id", -999999)) == int(cluster_id) and bool(row.get("keep", False))
    ]
    if kept_same_cluster:
        kept_same_cluster.sort(
            key=lambda row: (
                int(bool(row.get("representative", False))),
                _bbox_iou_xyxy(target_bbox, tuple(float(v) for v in row.get("bbox", [0, 0, 0, 0]))),
                float(row.get("score", 0.0)),
            ),
            reverse=True,
        )
        return kept_same_cluster[0]
    return None


def _resolve_det_root(det_root_arg: str, views_json: str) -> Path:
    if det_root_arg:
        return Path(det_root_arg)
    views_path = Path(views_json)
    parts = list(views_path.parts)
    if "01_make_views_output_outdoor" in parts:
        idx = parts.index("01_make_views_output_outdoor")
        prefix = Path(*parts[:idx]) if idx > 0 else Path()
        return prefix / "02_detect_output"
    return Path("results_test/02_detect_output")


def _get_view_det_json(det_root: Path, scene_id: str, viewpoint_id: str, view_id: str) -> Path:
    if det_root.name == "detections":
        return det_root / f"{view_id}.json"
    return det_root / scene_id / viewpoint_id / "detections" / f"{view_id}.json"


def _pick_best_detect_box(target_bbox: tuple[int, int, int, int], det_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    best_key = (-1.0, -1.0)
    for row in det_rows:
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox_xyxy = tuple(float(v) for v in bbox)
        iou_target = _bbox_iou_xyxy(target_bbox, bbox_xyxy)
        score = float(row.get("score", 0.0))
        key = (float(iou_target), score)
        if key > best_key:
            best_key = key
            best = row
    return best


def _save_filtered_erp_viz(viz_dir: Path, obj: dict[str, Any], entities: list[dict[str, Any]]) -> None:
    erp_path = obj.get("erp_path", "")
    if not erp_path:
        return
    try:
        erp_img = Image.open(str(erp_path)).convert("RGB")
    except Exception:
        return

    # Draw bounding boxes for filtered entities (legacy viz behavior).
    viz = erp_img
    for entity in entities:
        bbox = entity.get("bbox_xyxy")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            mask_rle = entity.get("mask_rle")
            if mask_rle:
                mask = decode_binary_mask(mask_rle).astype(np.uint8)
                if mask.shape[:2] != (viz.height, viz.width):
                    mask_img = Image.fromarray(mask).resize((viz.width, viz.height), Image.Resampling.NEAREST)
                    mask = np.array(mask_img)
                bbox = _mask_to_bbox_xyxy(mask)
        if not bbox:
            continue
        sanitized = _sanitize_bbox_xyxy(bbox, viz.size)
        if sanitized is None:
            continue
        viz = draw_bbox_outline(viz, sanitized, color=(0, 255, 0), width=3)

    viz.save(viz_dir / "erp_filtered.jpg", quality=95)


def _sanitize_bbox_xyxy(bbox: list[float] | tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width, height = image_size
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _resolve_candidate_cached_views(
    entity: dict,
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


def _pick_context_view(views: list[dict[str, Any]], mask_erp01: np.ndarray, preferred_view_id: str = "") -> tuple[ViewSpec, np.ndarray, float] | None:
    # 优先使用 05 里已经选中的语义视图：
    # 这样语义理解和 re-localization 尽量使用同一张透视图。
    if preferred_view_id:
        for item in views:
            if item["view"].view_id == preferred_view_id:
                vm = _project_mask(mask_erp01, item)
                if vm.sum() > 0:
                    return item["view"], vm, float(vm.sum())
    best = None
    best_score = -1.0
    for item in views:
        # 如果没有 preferred view，就从候选透视图里重新选一张最适合做 full-view re-ground 的图。
        vm = _project_mask(mask_erp01, item)
        bbox = _mask_to_bbox_xyxy(vm)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        h, w = vm.shape[:2]
        border_margin = min(x1, y1, max(0, w - x2), max(0, h - y2))
        # 当前打分仍然偏向“目标在图中面积大 + 不贴边”，
        # 只是它服务的是 context 视图选择，而不是最终关系表达生成。
        score = float(vm.sum()) + 0.15 * float(border_margin)
        if score > best_score:
            best_score = score
            best = (item["view"], vm, score)
    return best


def _project_mask(mask_erp01: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    # 把 ERP mask 投到某一张透视图上，得到该实例在当前视图里的像素覆盖区域。
    erp_h, erp_w = mask_erp01.shape[:2]
    ex = item["map_x"] % erp_w
    ey = np.clip(item["map_y"], 0, erp_h - 1)
    return mask_erp01[ey, ex].astype(np.uint8)


def json_dumps(obj: Any) -> str:
    return __import__("json").dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()