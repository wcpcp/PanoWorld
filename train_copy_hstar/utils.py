import random
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import json
import os
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from modeling import attach_erp_adapter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank0_print(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _get_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _set_module_trainable(module, enabled: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = enabled


def _param_effective_numel(param) -> int:
    ds_numel = getattr(param, "ds_numel", None)
    if ds_numel is not None:
        return int(ds_numel)
    return int(param.numel())


def set_model(cfg, model):
    trainable_modules = cfg.model.trainable_modules or {}
    if not trainable_modules or trainable_modules.get("all"):
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    visual = getattr(getattr(model, "model", None), "visual", None)
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    lm_head = getattr(model, "lm_head", None)

    if trainable_modules.get("visual"):
        _set_module_trainable(visual, True)
    if trainable_modules.get("visual_merger") and visual is not None:
        _set_module_trainable(getattr(visual, "merger", None), True)
    if trainable_modules.get("language_model"):
        _set_module_trainable(language_model, True)
        _set_module_trainable(lm_head, True)

    if trainable_modules.get("erp_adapters"):
        _set_module_trainable(getattr(model, "erp_patch_adapter", None), True)
        _set_module_trainable(getattr(model, "erp_merger_adapter", None), True)
        _set_module_trainable(getattr(model, "erp_output_adapter", None), True)

    if trainable_modules.get("erp_patch_adapter"):
        _set_module_trainable(getattr(model, "erp_patch_adapter", None), True)
    if trainable_modules.get("erp_merger_adapter"):
        _set_module_trainable(getattr(model, "erp_merger_adapter", None), True)
    if trainable_modules.get("erp_output_adapter"):
        _set_module_trainable(getattr(model, "erp_output_adapter", None), True)


def print_model_parameters(model, max_names: int = 80):
    total_params = 0
    trainable_params = 0
    local_total_params = 0
    local_trainable_params = 0
    trainable_names: List[str] = []
    bucket_trainable = {
        "language_model": 0,
        "visual": 0,
        "lm_head": 0,
        "erp_adapters": 0,
        "other": 0,
    }
    bucket_total = {key: 0 for key in bucket_trainable}

    def _bucket_for_name(name: str) -> str:
        if ".language_model." in name or name.startswith("model.language_model."):
            return "language_model"
        if ".visual." in name or name.startswith("model.visual.") or name.startswith("visual."):
            return "visual"
        if name.startswith("lm_head.") or ".lm_head." in name:
            return "lm_head"
        if "erp_" in name:
            return "erp_adapters"
        return "other"

    for name, param in model.named_parameters():
        count = _param_effective_numel(param)
        local_count = int(param.numel())
        total_params += count
        local_total_params += local_count
        bucket = _bucket_for_name(name)
        bucket_total[bucket] += count
        if param.requires_grad:
            trainable_params += count
            local_trainable_params += local_count
            bucket_trainable[bucket] += count
            trainable_names.append(name)

    trainable_ratio = 0.0 if total_params == 0 else 100.0 * trainable_params / total_params
    print(
        f"trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_ratio:.2f}%)",
        flush=True,
    )
    if total_params != local_total_params or trainable_params != local_trainable_params:
        print(
            "local shard params (for ZeRO-style partitioned models): "
            f"{local_trainable_params:,} / {local_total_params:,}",
            flush=True,
        )

    print("trainable parameter breakdown:", flush=True)
    for bucket in ("language_model", "visual", "lm_head", "erp_adapters", "other"):
        bucket_total_params = bucket_total[bucket]
        bucket_trainable_params = bucket_trainable[bucket]
        bucket_ratio = (
            0.0 if bucket_total_params == 0 else 100.0 * bucket_trainable_params / bucket_total_params
        )
        print(
            f"  - {bucket}: {bucket_trainable_params:,} / {bucket_total_params:,} "
            f"({bucket_ratio:.2f}%)",
            flush=True,
        )

    preview = trainable_names[:max_names]
    if preview:
        print("trainable parameter names:", flush=True)
        for name in preview:
            print(f"  - {name}", flush=True)
    if len(trainable_names) > max_names:
        print(f"  ... and {len(trainable_names) - max_names} more", flush=True)


def load_model(cfg):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=_get_dtype(cfg.model.torch_dtype),
        cache_dir=cfg.model.cache_dir,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    attach_erp_adapter(model, cfg.erp)
    adapter_prefixes = _collect_post_attach_prefixes(model, cfg)
    _load_adapter_weights_if_present(model, cfg.model.name_or_path, adapter_prefixes)

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.config.use_cache = False

    return model


def _collect_post_attach_prefixes(model, cfg) -> List[str]:
    prefixes: List[str] = []

    if bool(getattr(getattr(cfg, "erp", None), "enabled", False)):
        stage_raw = str(getattr(cfg.erp, "stage", "") or "").strip().lower()
        stages = {part.strip() for part in stage_raw.split(",") if part.strip()}
        if not stages:
            stages = {"output"}

        if "patch" in stages and hasattr(model, "erp_patch_adapter"):
            prefixes.append("erp_patch_adapter.")
        if "merger" in stages and hasattr(model, "erp_merger_adapter"):
            prefixes.append("erp_merger_adapter.")
        if "output" in stages and hasattr(model, "erp_output_adapter"):
            prefixes.append("erp_output_adapter.")

    # Generic fallback: future adapter modules can follow the *_adapter naming convention.
    for attr_name in dir(model):
        if not attr_name.endswith("_adapter") or attr_name.startswith("_"):
            continue
        module = getattr(model, attr_name, None)
        if isinstance(module, torch.nn.Module):
            prefix = f"{attr_name}."
            if prefix not in prefixes:
                prefixes.append(prefix)

    return sorted(prefixes)


def _iter_checkpoint_safetensors(model_path: str) -> List[str]:
    if not os.path.isdir(model_path):
        return []

    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        return [single_file]

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            weight_map = payload.get("weight_map", {})
            shard_names = sorted({str(v) for v in weight_map.values() if v})
            shard_paths = [os.path.join(model_path, name) for name in shard_names]
            return [path for path in shard_paths if os.path.exists(path)]
        except Exception:
            return []

    candidates = sorted(
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
        if name.endswith(".safetensors")
    )
    return candidates


def _load_adapter_weights_if_present(model, model_path: str, adapter_prefixes: List[str]) -> None:
    if not adapter_prefixes:
        return

    files = _iter_checkpoint_safetensors(str(model_path))
    if not files:
        return

    try:
        from safetensors import safe_open
    except Exception:
        print("[adapter-load] safetensors is unavailable; skip explicit ERP adapter restore.", flush=True)
        return

    adapter_state: Dict[str, torch.Tensor] = {}
    has_any_candidate = False
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(tuple(adapter_prefixes)):
                    has_any_candidate = True
                    adapter_state[key] = f.get_tensor(key)

    if not has_any_candidate:
        return

    if not adapter_state:
        print("[adapter-load] candidate adapter tensors were found but none were readable.", flush=True)
        return

    incompatible = model.load_state_dict(adapter_state, strict=False)
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    loaded = len(adapter_state) - len(unexpected)

    model_state = model.state_dict()
    verified = 0
    mismatch = 0
    max_abs_diff = 0.0
    worst_key = None
    for key, ckpt_tensor in adapter_state.items():
        model_tensor = model_state.get(key)
        if model_tensor is None:
            mismatch += 1
            continue
        diff = (model_tensor.detach().cpu() - ckpt_tensor.detach().cpu()).abs().max().item()
        if diff <= 1e-6:
            verified += 1
        else:
            mismatch += 1
        if diff > max_abs_diff:
            max_abs_diff = float(diff)
            worst_key = key

    model_adapter_keys = [
        key
        for key in model_state.keys()
        if key.startswith(tuple(adapter_prefixes))
    ]
    missing_after_restore = sorted(set(model_adapter_keys) - set(adapter_state.keys()))

    print(
        "[adapter-load] restored adapter params "
        f"loaded={loaded}, provided={len(adapter_state)}, "
        f"missing_in_checkpoint={len(missing_after_restore)}, unexpected={len(unexpected)}",
        flush=True,
    )
    print(
        "[adapter-load] verification "
        f"verified={verified}, mismatch={mismatch}, max_abs_diff={max_abs_diff:.3e}, "
        f"worst_key={worst_key}",
        flush=True,
    )


def load_processor_and_tokenizer(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        cache_dir=cfg.model.cache_dir,
    )
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path,
            trust_remote_code=cfg.model.trust_remote_code,
            cache_dir=cfg.model.cache_dir,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append("<image>")
        return "".join(parts)
    return ""


def _apply_chat_template_compat(processor, messages, *, disable_thinking: bool = False) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        try:
            return processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass
        except Exception:
            pass
    return processor.apply_chat_template(messages, **kwargs)


def build_prompt_and_target(
    messages: List[Dict[str, Any]],
    prompt_format: str,
    processor=None,
    disable_thinking: bool = False,
) -> Dict[str, str]:
    if not messages:
        raise ValueError("messages 为空")
    if messages[-1].get("role") != "assistant":
        raise ValueError("messages 最后一个必须是 assistant")

    target_text = _content_to_text(messages[-1].get("content", ""))
    prompt_messages = messages[:-1]

    if prompt_format == "chat_template":
        if processor is None or not hasattr(processor, "apply_chat_template"):
            raise ValueError("chat_template 需要 processor.apply_chat_template")
        prompt_text = _apply_chat_template_compat(
            processor,
            prompt_messages,
            disable_thinking=disable_thinking,
        )
        return {"prompt": prompt_text, "target": target_text}

    chunks = []
    for msg in prompt_messages:
        role = msg.get("role", "user")
        text = _content_to_text(msg.get("content", ""))
        chunks.append(f"<|{role}|>\n{text}\n")
    chunks.append("<|assistant|>\n")
    return {"prompt": "".join(chunks), "target": target_text}


def init_wandb(wandb_cfg, training_args, rank):
    if wandb_cfg is None:
        return

    report_to = training_args.report_to
    if report_to is None:
        return
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to or rank != 0:
        return

    import wandb

    if is_dataclass(wandb_cfg):
        cfg_dict = asdict(wandb_cfg)
    elif isinstance(wandb_cfg, dict):
        cfg_dict = dict(wandb_cfg)
    else:
        cfg_dict = {
            key: getattr(wandb_cfg, key)
            for key in ("project", "entity", "name", "tags", "mode", "group", "notes", "id", "resume")
            if hasattr(wandb_cfg, key)
        }

    init_kwargs = {key: value for key, value in cfg_dict.items() if value is not None}
    if not init_kwargs.get("name") and getattr(training_args, "run_name", None):
        init_kwargs["name"] = training_args.run_name

    wandb.init(**init_kwargs)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _prepare_eval_texts(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    if preds.ndim == labels.ndim + 1:
        preds = np.argmax(preds, axis=-1)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    def _sanitize(x):
        x = np.asarray(x).astype(np.int64, copy=False)
        x = np.where(x < 0, pad_id, x)
        x = np.where(x >= vocab_size, pad_id, x)
        return x

    preds = _sanitize(preds)
    labels = labels.astype(np.int64, copy=False)

    shifted_preds = np.full_like(preds, pad_id)
    shifted_preds[:, 1:] = preds[:, :-1]

    pred_ids = np.where(labels != -100, shifted_preds, pad_id)
    label_ids = np.where(labels != -100, labels, pad_id)

    pred_ids = _sanitize(pred_ids)
    label_ids = _sanitize(label_ids)

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return pred_texts, label_texts


def _prepare_generation_eval_texts(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    def _sanitize(x):
        x = np.asarray(x).astype(np.int64, copy=False)
        x = np.where(x < 0, pad_id, x)
        x = np.where(x >= vocab_size, pad_id, x)
        return x

    preds = _sanitize(preds)
    labels = np.where(labels == -100, pad_id, labels)
    labels = _sanitize(labels)

    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return pred_texts, label_texts


def build_exact_match_metrics(tokenizer, generation_mode: bool = False):
    def compute_metrics(eval_preds):
        if generation_mode:
            pred_texts, label_texts = _prepare_generation_eval_texts(tokenizer, eval_preds)
        else:
            pred_texts, label_texts = _prepare_eval_texts(tokenizer, eval_preds)
        total = len(label_texts)
        if total == 0:
            return {"exact_match": 0.0}

        correct = 0
        for pred_text, label_text in zip(pred_texts, label_texts):
            if _normalize_text(pred_text) == _normalize_text(label_text):
                correct += 1
        return {"exact_match": float(correct / total)}

    return compute_metrics


def _extract_choice_key(
    text: str,
    valid_keys: List[str],
    option_text_to_key: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    raw_text = (text or "").strip()
    normalized = _normalize_text(raw_text)
    if not normalized:
        return None

    for key in valid_keys:
        key_norm = key.lower()
        if normalized == key_norm:
            return key

    single_char_keys = [key for key in valid_keys if len(key) == 1 and key.isalpha()]
    if single_char_keys:
        key_union = "".join(sorted({key.upper() for key in single_char_keys}))
        start_match = re.match(
            rf"^\s*[\(\[]?([{key_union}])(?:[\]\)\.\:\,\s\-]|$)",
            raw_text,
            flags=re.I,
        )
        if start_match:
            candidate = start_match.group(1).upper()
            if candidate in valid_keys:
                return candidate

        # explicit_answer_match = re.search(
        #     rf"\b(?:answer|correct answer|final answer|best answer)\b\s*(?:is|:)?\s*\**\(?\s*([{key_union}])(?:[\]\)\*\.:\,\s\-]|$)",
        #     raw_text,
        #     flags=re.I,
        # )
        explicit_answer_match = re.search(
            rf"\b(?:answer|correct answer|final answer|best answer)\b\s*(?:is)?\s*:?\s*\**\(?\s*([{key_union}])(?:[\]\)\*\.:\,\s\-]|$)",
            raw_text,
            flags=re.I,
        )
        if explicit_answer_match:
            candidate = explicit_answer_match.group(1).upper()
            if candidate in valid_keys:
                return candidate

    # patterns = [
    #     rf"(?:^|[^a-z0-9]){re.escape(key.lower())}(?:$|[^a-z0-9])"
    #     for key in valid_keys
    # ]
    # for key, pattern in zip(valid_keys, patterns):
    #     if re.search(pattern, normalized):
    #         return key

    if option_text_to_key:
        for option_text, key in sorted(option_text_to_key.items(), key=lambda item: len(item[0]), reverse=True):
            if option_text and option_text in normalized:
                return key

    return None


def _extract_numbers(text: str) -> List[float]:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return [float(x) for x in matches]


def _extract_bfov(text: str) -> Optional[List[float]]:
    lowered = (text or "").lower()

    named_patterns = {
        "yaw": r"\byaw\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        "pitch": r"\bpitch\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        "x_fov": r"\b(?:x[_\-\s]?(?:fov|bfov)|xfov|xbfov)\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        "y_fov": r"\b(?:y[_\-\s]?(?:fov|bfov)|yfov|ybfov)\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    }
    named_values = {}
    for key, pattern in named_patterns.items():
        match = re.search(pattern, lowered, flags=re.I)
        if match:
            named_values[key] = float(match.group(1))
    if len(named_values) == 4:
        return [
            named_values["yaw"],
            named_values["pitch"],
            named_values["x_fov"],
            named_values["y_fov"],
        ]

    for segment in re.findall(r"[\[\(]([^\]\)]{0,200})[\]\)]", text or ""):
        numbers = _extract_numbers(segment)
        if len(numbers) >= 4:
            return numbers[:4]

    numbers = _extract_numbers(text)
    has_bfov_hint = any(token in lowered for token in ("bfov", "yaw", "pitch", "x_fov", "y_fov", "xfov", "yfov"))
    if has_bfov_hint and len(numbers) >= 4:
        return numbers[:4]
    return None


def _extract_yaw_pitch(text: str) -> Optional[List[float]]:
    lowered = (text or "").lower()

    yaw_match = re.search(r"\byaw\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", lowered, flags=re.I)
    pitch_match = re.search(r"\bpitch\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", lowered, flags=re.I)
    if yaw_match and pitch_match:
        return [float(yaw_match.group(1)), float(pitch_match.group(1))]

    for segment in re.findall(r"[\[\(]([^\]\)]{0,200})[\]\)]", text or ""):
        numbers = _extract_numbers(segment)
        if len(numbers) >= 2:
            return numbers[:2]

    numbers = _extract_numbers(text)
    if len(numbers) >= 2:
        return numbers[:2]
    return None


def _normalize_yaw(yaw: float) -> float:
    return yaw % 360.0


def _normalize_signed_yaw(yaw: float) -> float:
    normalized = (float(yaw) + 180.0) % 360.0 - 180.0
    if abs(normalized + 180.0) < 1e-6 and float(yaw) > 0:
        return 180.0
    return normalized


def _yaw_to_360(yaw: float) -> float:
    return _normalize_yaw(_normalize_signed_yaw(float(yaw)))


def _contains_circular_yaw(yaw: float, start: float, end: float) -> bool:
    eps = 1e-6
    yaw = _yaw_to_360(float(yaw))
    start = _yaw_to_360(float(start))
    end = _yaw_to_360(float(end))

    if abs(start - end) <= eps:
        return abs(yaw - start) <= eps

    if start <= end:
        return (start - eps) <= yaw <= (end + eps)
    return yaw >= (start - eps) or yaw <= (end + eps)


def _contains_pitch(pitch: float, start: float, end: float) -> bool:
    low = min(float(start), float(end))
    high = max(float(start), float(end))
    return low <= float(pitch) <= high


def _is_zero_pitch_interval(interval: List[float], eps: float = 1e-6) -> bool:
    return abs(float(interval[0])) <= eps and abs(float(interval[1])) <= eps


def _parse_interval_pair(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except Exception:
            return None
    return None


def _parse_answer_interval_text(text: Any) -> Optional[Dict[str, List[float]]]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        raw = json.loads(text)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    yaw = _parse_interval_pair(raw.get("yaw"))
    pitch = _parse_interval_pair(raw.get("pitch"))
    if yaw is None or pitch is None:
        return None
    return {"yaw": yaw, "pitch": pitch}


def _get_thinking360_intervals(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[float]]]:
    if not isinstance(item, dict):
        return None

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        yaw = _parse_interval_pair(metadata.get("target_yaw_interval"))
        pitch = _parse_interval_pair(metadata.get("target_pitch_interval"))
        if yaw is not None and pitch is not None:
            return {"yaw": yaw, "pitch": pitch}

        parsed_text = _parse_answer_interval_text(metadata.get("answer_interval_text"))
        if parsed_text is not None:
            return parsed_text

    for key in ("target_yaw_interval", "yaw_interval"):
        yaw = _parse_interval_pair(item.get(key))
        if yaw is not None:
            break
    else:
        yaw = None

    for key in ("target_pitch_interval", "pitch_interval"):
        pitch = _parse_interval_pair(item.get(key))
        if pitch is not None:
            break
    else:
        pitch = None

    if yaw is not None and pitch is not None:
        return {"yaw": yaw, "pitch": pitch}
    return None


def _infer_thinking360_group(item: Optional[Dict[str, Any]]) -> str:
    if not isinstance(item, dict):
        return "other"

    candidates = [
        item.get("task_id"),
        item.get("ability_group"),
        item.get("scene_id"),
        item.get("item_id"),
        item.get("id"),
    ]

    images = item.get("images")
    if isinstance(images, list):
        candidates.extend(images)
    for key in ("image_path",):
        if item.get(key) is not None:
            candidates.append(item.get(key))

    for value in candidates:
        text = str(value or "").lower()
        if "hos" in text:
            return "hos"
        if "hps" in text:
            return "hps"
    return "other"


def _yaw_segments(center: float, width: float) -> List[tuple[float, float]]:
    width = max(0.0, min(float(width), 360.0))
    center = _normalize_yaw(center)
    half = width / 2.0
    start = center - half
    end = center + half
    if start >= 0.0 and end <= 360.0:
        return [(start, end)]
    if start < 0.0:
        return [(start + 360.0, 360.0), (0.0, end)]
    return [(start, 360.0), (0.0, end - 360.0)]


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _bfov_iou(pred: List[float], target: List[float]) -> Optional[float]:
    if len(pred) < 4 or len(target) < 4:
        return None

    pyaw, ppitch, pxfov, pyfov = [float(x) for x in pred[:4]]
    tyaw, tpitch, txfov, tyfov = [float(x) for x in target[:4]]

    p_top = min(90.0, ppitch + pyfov / 2.0)
    p_bottom = max(-90.0, ppitch - pyfov / 2.0)
    t_top = min(90.0, tpitch + tyfov / 2.0)
    t_bottom = max(-90.0, tpitch - tyfov / 2.0)

    p_height = max(0.0, p_top - p_bottom)
    t_height = max(0.0, t_top - t_bottom)
    if p_height <= 0.0 or t_height <= 0.0:
        return 0.0

    inter_height = _interval_overlap(p_bottom, p_top, t_bottom, t_top)
    if inter_height <= 0.0:
        return 0.0

    p_segments = _yaw_segments(pyaw, pxfov)
    t_segments = _yaw_segments(tyaw, txfov)
    inter_width = 0.0
    for ps, pe in p_segments:
        for ts, te in t_segments:
            inter_width += _interval_overlap(ps, pe, ts, te)

    if inter_width <= 0.0:
        return 0.0

    inter_area = inter_width * inter_height
    p_area = max(0.0, min(float(pxfov), 360.0)) * p_height
    t_area = max(0.0, min(float(txfov), 360.0)) * t_height
    union = p_area + t_area - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)

def _update_hit_counts(iou: Optional[float], hit_counts: Dict[float, int], thresholds: List[float]) -> None:
    if iou is None:
        return
    for thr in thresholds:
        if iou >= thr:
            hit_counts[thr] += 1

def _get_item_target_text(item: Dict[str, Any]) -> str:
    answer_format = str(item.get("answer_format") or "").lower()
    if answer_format == "bfov_regression":
        preferred_keys = ("answer_text", "answer", "response")
    else:
        preferred_keys = ("answer", "response", "answer_text")
    for key in preferred_keys:
        if item.get(key) is not None:
            return str(item[key])
    return ""


def _get_item_valid_keys(
    item: Optional[Dict[str, Any]],
    fallback_keys: Optional[List[str]] = None,
) -> List[str]:
    if item:
        options = item.get("options")
        if isinstance(options, list) and options:
            keys = [
                str(option.get("key")).upper()
                for option in options
                if isinstance(option, dict) and option.get("key") is not None
            ]
            if keys:
                return sorted(set(keys))

        answer_format = str(item.get("answer_format") or "").lower()
        if answer_format in {"4_way_multiple_choice", "multiple_choice", "5_way_multiple_choice"}:
            return ["A", "B", "C", "D", "E"]

        answer = item.get("answer")
        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer in {"A", "B", "C", "D", "E"}:
                return ["A", "B", "C", "D", "E"]

    return fallback_keys or ["A", "B", "C", "D", "E"]


def _get_item_option_text_to_key(item: Optional[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not item:
        return mapping
    options = item.get("options")
    if not isinstance(options, list):
        return mapping
    for option in options:
        if not isinstance(option, dict):
            continue
        key = option.get("key")
        text = option.get("text")
        if key and text:
            mapping[_normalize_text(str(text))] = str(key).upper()
    return mapping


def _get_item_label_key(
    item: Optional[Dict[str, Any]],
    label_text: str,
    valid_keys: List[str],
    option_text_to_key: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if item:
        answer = item.get("answer")
        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer in valid_keys:
                return answer
    return _extract_choice_key(label_text, valid_keys, option_text_to_key)



def build_choice_accuracy_metrics(
    tokenizer,
    valid_keys: Optional[List[str]] = None,
    option_text_to_key: Optional[Dict[str, str]] = None,
    generation_mode: bool = False,
    eval_items: Optional[List[Dict[str, Any]]] = None,
    print_predictions: bool = False,
):
    if not valid_keys:
        valid_keys = ["A", "B", "C", "D", "E"]
    valid_keys = [str(key).upper() for key in valid_keys]

    normalized_option_text_to_key = None
    if option_text_to_key:
        normalized_option_text_to_key = {
            _normalize_text(text): str(key).upper()
            for text, key in option_text_to_key.items()
            if text
        }

    bfov_thresholds = [0.1, 0.25, 0.5]

    def compute_metrics(eval_preds):
        if generation_mode:
            pred_texts, decoded_label_texts = _prepare_generation_eval_texts(tokenizer, eval_preds)
        else:
            pred_texts, decoded_label_texts = _prepare_eval_texts(tokenizer, eval_preds)

        if eval_items is not None and len(eval_items) == len(pred_texts):
            label_texts = [_get_item_target_text(item) for item in eval_items]
        else:
            label_texts = decoded_label_texts

        total = 0
        correct = 0
        exact_match = 0
        invalid_predictions = 0

        bfov_total = 0
        bfov_valid = 0
        bfov_iou_sum = 0.0
        bfov_hit_counts = {thr: 0 for thr in bfov_thresholds}

        # choice task: 统计 accuracy
        task_choice_stats: Dict[str, Dict[str, float]] = {}

        # bfov task: 统计 mean iou / valid rate / iou@k
        task_bfov_stats: Dict[str, Dict[str, Any]] = {}

        for idx, (pred_text, label_text) in enumerate(zip(pred_texts, label_texts)):
            item = eval_items[idx] if eval_items is not None and idx < len(eval_items) else None
            task_id = str(item.get("task_id", "unknown")) if item else "unknown"

            if _normalize_text(pred_text) == _normalize_text(label_text):
                exact_match += 1

            answer_format = str(item.get("answer_format") or "").lower() if item else ""
            is_bfov = answer_format == "bfov_regression"
            label_bfov = _extract_bfov(label_text) if is_bfov or not item else None

            # -------------------------
            # BFOV regression 分支
            # -------------------------
            if label_bfov is not None:
                bfov_total += 1
                pred_bfov = _extract_bfov(pred_text)
                iou = None

                if pred_bfov is not None:
                    bfov_valid += 1
                    iou = _bfov_iou(pred_bfov, label_bfov)
                    if iou is not None:
                        bfov_iou_sum += iou
                        for thr in bfov_thresholds:
                            if iou >= thr:
                                bfov_hit_counts[thr] += 1

                if task_id not in task_bfov_stats:
                    task_bfov_stats[task_id] = {
                        "count": 0.0,
                        "valid_count": 0.0,
                        "iou_sum": 0.0,
                        "hit_counts": {thr: 0.0 for thr in bfov_thresholds},
                    }

                task_bfov_stats[task_id]["count"] += 1.0
                if pred_bfov is not None:
                    task_bfov_stats[task_id]["valid_count"] += 1.0
                if iou is not None:
                    task_bfov_stats[task_id]["iou_sum"] += float(iou)
                    for thr in bfov_thresholds:
                        if iou >= thr:
                            task_bfov_stats[task_id]["hit_counts"][thr] += 1.0

                if print_predictions:
                    item_id = item.get("item_id") if item else idx
                    print(
                        "[eval-sample] "
                        f"item_id={item_id} "
                        f"task_id={task_id} "
                        f"type=bfov "
                        f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                        f"pred_extracted={json.dumps(pred_bfov, ensure_ascii=False)} "
                        f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
                        f"gold_extracted={json.dumps(label_bfov, ensure_ascii=False)} "
                        f"iou={json.dumps(iou, ensure_ascii=False)}",
                        flush=True,
                    )
                continue

            # -------------------------
            # Choice 分支
            # -------------------------
            sample_valid_keys = _get_item_valid_keys(item, valid_keys)
            sample_option_text_to_key = (
                _get_item_option_text_to_key(item)
                if item is not None
                else (normalized_option_text_to_key or {})
            )

            pred_key = _extract_choice_key(
                pred_text,
                sample_valid_keys,
                sample_option_text_to_key,
            )
            label_key = _get_item_label_key(
                item,
                label_text,
                sample_valid_keys,
                sample_option_text_to_key,
            )

            if label_key is None:
                if print_predictions:
                    item_id = item.get("item_id") if item else idx
                    print(
                        "[eval-sample] "
                        f"item_id={item_id} "
                        f"task_id={task_id} "
                        f"type=choice "
                        f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                        f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
                        f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
                        f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
                        "correct=null "
                        "note=\"gold key not parsable\"",
                        flush=True,
                    )
                continue

            total += 1
            if task_id not in task_choice_stats:
                task_choice_stats[task_id] = {"total": 0.0, "correct": 0.0}
            task_choice_stats[task_id]["total"] += 1.0

            is_correct = False
            if pred_key is None:
                invalid_predictions += 1
                if print_predictions:
                    item_id = item.get("item_id") if item else idx
                    print(
                        "[eval-sample] "
                        f"item_id={item_id} "
                        f"task_id={task_id} "
                        f"type=choice "
                        f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                        f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
                        f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
                        f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
                        f"correct={is_correct}",
                        flush=True,
                    )
                continue

            if pred_key == label_key:
                correct += 1
                task_choice_stats[task_id]["correct"] += 1.0
                is_correct = True

            if print_predictions:
                item_id = item.get("item_id") if item else idx
                print(
                    "[eval-sample] "
                    f"item_id={item_id} "
                    f"task_id={task_id} "
                    f"type=choice "
                    f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                    f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
                    f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
                    f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
                    f"correct={is_correct}",
                    flush=True,
                )

        metrics = {
            "exact_match": 0.0 if not label_texts else float(exact_match / len(label_texts)),
            "choice_accuracy": 0.0 if total == 0 else float(correct / total),
            "invalid_choice_rate": 0.0 if total == 0 else float(invalid_predictions / total),
            "bfov_iou": 0.0 if bfov_total == 0 else float(bfov_iou_sum / bfov_total),
            "bfov_valid_rate": 0.0 if bfov_total == 0 else float(bfov_valid / bfov_total),
        }

        for thr in bfov_thresholds:
            metrics[f"bfov_iou@{thr}"] = (
                0.0 if bfov_total == 0 else float(bfov_hit_counts[thr] / bfov_total)
            )

        # 每个 choice 任务的 accuracy
        for task_id, stat in task_choice_stats.items():
            safe_task_id = re.sub(r"[^a-zA-Z0-9_]+", "_", task_id)
            task_total = stat["total"]
            task_correct = stat["correct"]
            metrics[f"task_acc_{safe_task_id}"] = (
                0.0 if task_total == 0 else float(task_correct / task_total)
            )
            metrics[f"task_count_{safe_task_id}"] = float(task_total)

        # 每个 bfov 任务的完整指标
        for task_id, stat in task_bfov_stats.items():
            safe_task_id = re.sub(r"[^a-zA-Z0-9_]+", "_", task_id)
            task_count = stat["count"]
            task_valid_count = stat["valid_count"]
            task_iou_sum = stat["iou_sum"]

            metrics[f"task_bfov_iou_{safe_task_id}"] = (
                0.0 if task_count == 0 else float(task_iou_sum / task_count)
            )
            metrics[f"task_bfov_valid_rate_{safe_task_id}"] = (
                0.0 if task_count == 0 else float(task_valid_count / task_count)
            )
            metrics[f"task_bfov_count_{safe_task_id}"] = float(task_count)

            for thr in bfov_thresholds:
                metrics[f"task_bfov_iou@{thr}_{safe_task_id}"] = (
                    0.0 if task_count == 0 else float(stat["hit_counts"][thr] / task_count)
                )

        return metrics

    return compute_metrics


def build_thinking360_interval_metrics(
    tokenizer,
    generation_mode: bool = True,
    eval_items: Optional[List[Dict[str, Any]]] = None,
    print_predictions: bool = False,
):
    def compute_metrics(eval_preds):
        if generation_mode:
            pred_texts, label_texts = _prepare_generation_eval_texts(tokenizer, eval_preds)
        else:
            pred_texts, label_texts = _prepare_eval_texts(tokenizer, eval_preds)

        total = 0
        success = 0
        invalid_predictions = 0
        missing_intervals = 0
        exact_match = 0
        yaw_total = 0
        yaw_correct = 0
        pitch_total = 0
        pitch_correct = 0
        pitch_ignored_total = 0

        group_total = {"hos": 0, "hps": 0, "other": 0}
        group_success = {"hos": 0, "hps": 0, "other": 0}
        group_yaw_total = {"hos": 0, "hps": 0, "other": 0}
        group_yaw_correct = {"hos": 0, "hps": 0, "other": 0}
        group_pitch_total = {"hos": 0, "hps": 0, "other": 0}
        group_pitch_correct = {"hos": 0, "hps": 0, "other": 0}
        group_pitch_ignored = {"hos": 0, "hps": 0, "other": 0}

        for idx, pred_text in enumerate(pred_texts):
            label_text = label_texts[idx] if idx < len(label_texts) else ""
            if _normalize_text(pred_text) == _normalize_text(label_text):
                exact_match += 1

            item = eval_items[idx] if eval_items is not None and idx < len(eval_items) else None
            intervals = _get_thinking360_intervals(item)
            if intervals is None:
                missing_intervals += 1
                continue

            total += 1
            group = _infer_thinking360_group(item)
            group_total[group] += 1

            yaw_interval = intervals["yaw"]
            pitch_interval = intervals["pitch"]
            ignore_pitch = _is_zero_pitch_interval(pitch_interval)
            yaw_total += 1
            group_yaw_total[group] += 1
            if ignore_pitch:
                pitch_ignored_total += 1
                group_pitch_ignored[group] += 1
            else:
                pitch_total += 1
                group_pitch_total[group] += 1

            pred_pair = _extract_yaw_pitch(pred_text)
            if pred_pair is None:
                invalid_predictions += 1
                if print_predictions:
                    item_id = (item or {}).get("item_id") or (item or {}).get("id") or idx
                    print(
                        "[eval-sample] "
                        f"item_id={item_id} "
                        f"group={group} "
                        "type=thinking360 "
                        f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                        f"yaw_interval={json.dumps(yaw_interval, ensure_ascii=False)} "
                        f"pitch_interval={json.dumps(pitch_interval, ensure_ascii=False)} "
                        f"pitch_ignored={json.dumps(ignore_pitch)} "
                        "correct=false "
                        "note=\"prediction not parsable\"",
                        flush=True,
                    )
                continue

            pred_yaw, pred_pitch = float(pred_pair[0]), float(pred_pair[1])
            yaw_ok = _contains_circular_yaw(pred_yaw, yaw_interval[0], yaw_interval[1])
            if yaw_ok:
                yaw_correct += 1
                group_yaw_correct[group] += 1

            if ignore_pitch:
                pitch_ok = True
            else:
                pitch_ok = _contains_pitch(pred_pitch, pitch_interval[0], pitch_interval[1])
                if pitch_ok:
                    pitch_correct += 1
                    group_pitch_correct[group] += 1

            is_success = bool(yaw_ok and pitch_ok)
            if is_success:
                success += 1
                group_success[group] += 1

            if print_predictions:
                item_id = (item or {}).get("item_id") or (item or {}).get("id") or idx
                print(
                    "[eval-sample] "
                    f"item_id={item_id} "
                    f"group={group} "
                    "type=thinking360 "
                    f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                    f"pred_extracted={json.dumps([pred_yaw, pred_pitch], ensure_ascii=False)} "
                    f"pred_extracted_signed={json.dumps([_normalize_signed_yaw(pred_yaw), pred_pitch], ensure_ascii=False)} "
                    f"yaw_interval={json.dumps(yaw_interval, ensure_ascii=False)} "
                    f"yaw_interval_signed={json.dumps([_normalize_signed_yaw(yaw_interval[0]), _normalize_signed_yaw(yaw_interval[1])], ensure_ascii=False)} "
                    f"pitch_interval={json.dumps(pitch_interval, ensure_ascii=False)} "
                    f"pitch_ignored={json.dumps(ignore_pitch)} "
                    f"yaw_ok={json.dumps(yaw_ok)} "
                    f"pitch_ok={json.dumps(pitch_ok)} "
                    f"correct={json.dumps(is_success)}",
                    flush=True,
                )

        metrics = {
            "exact_match": 0.0 if not pred_texts else float(exact_match / len(pred_texts)),
            "thinking360_success_rate": 0.0 if total == 0 else float(success / total),
            "thinking360_overall_success_rate": 0.0 if total == 0 else float(success / total),
            "thinking360_invalid_prediction_rate": 0.0 if total == 0 else float(invalid_predictions / total),
            "thinking360_total": float(total),
            "thinking360_missing_interval": float(missing_intervals),
            "thinking360_yaw_accuracy": 0.0 if yaw_total == 0 else float(yaw_correct / yaw_total),
            "thinking360_pitch_accuracy": 1.0 if pitch_total == 0 else float(pitch_correct / pitch_total),
            "thinking360_yaw_total": float(yaw_total),
            "thinking360_pitch_total": float(pitch_total),
            "thinking360_pitch_ignored_total": float(pitch_ignored_total),
        }

        for group in ("hos", "hps", "other"):
            group_count = group_total[group]
            metrics[f"thinking360_{group}_success_rate"] = (
                0.0 if group_count == 0 else float(group_success[group] / group_count)
            )
            metrics[f"thinking360_{group}_total"] = float(group_count)

            group_yaw_count = group_yaw_total[group]
            group_pitch_count = group_pitch_total[group]
            metrics[f"thinking360_{group}_yaw_accuracy"] = (
                0.0 if group_yaw_count == 0 else float(group_yaw_correct[group] / group_yaw_count)
            )
            metrics[f"thinking360_{group}_pitch_accuracy"] = (
                1.0 if group_pitch_count == 0 else float(group_pitch_correct[group] / group_pitch_count)
            )
            metrics[f"thinking360_{group}_yaw_total"] = float(group_yaw_count)
            metrics[f"thinking360_{group}_pitch_total"] = float(group_pitch_count)
            metrics[f"thinking360_{group}_pitch_ignored_total"] = float(group_pitch_ignored[group])

        return metrics

    return compute_metrics


# def build_choice_accuracy_metrics(
#     tokenizer,
#     valid_keys: Optional[List[str]] = None,
#     option_text_to_key: Optional[Dict[str, str]] = None,
#     generation_mode: bool = False,
#     eval_items: Optional[List[Dict[str, Any]]] = None,
#     print_predictions: bool = False,
# ):
#     if not valid_keys:
#         valid_keys = ["A", "B", "C", "D", "E"]
#     valid_keys = [str(key).upper() for key in valid_keys]

#     normalized_option_text_to_key = None
#     if option_text_to_key:
#         normalized_option_text_to_key = {
#             _normalize_text(text): str(key).upper()
#             for text, key in option_text_to_key.items()
#             if text
#         }

#     def compute_metrics(eval_preds):
#         if generation_mode:
#             pred_texts, decoded_label_texts = _prepare_generation_eval_texts(tokenizer, eval_preds)
#         else:
#             pred_texts, decoded_label_texts = _prepare_eval_texts(tokenizer, eval_preds)

#         if eval_items is not None and len(eval_items) == len(pred_texts):
#             label_texts = [_get_item_target_text(item) for item in eval_items]
#         else:
#             label_texts = decoded_label_texts

#         total = 0
#         correct = 0
#         exact_match = 0
#         invalid_predictions = 0
#         bfov_total = 0
#         bfov_valid = 0
#         bfov_iou_sum = 0.0

#         # 分任务统计
#         # choice: 统计 acc
#         # bfov:   统计 mean iou
#         task_choice_stats: Dict[str, Dict[str, int]] = {}
#         task_bfov_stats: Dict[str, Dict[str, float]] = {}

#         for idx, (pred_text, label_text) in enumerate(zip(pred_texts, label_texts)):
#             item = eval_items[idx] if eval_items is not None and idx < len(eval_items) else None
#             task_id = str(item.get("task_id", "unknown")) if item else "unknown"

#             if _normalize_text(pred_text) == _normalize_text(label_text):
#                 exact_match += 1

#             answer_format = str(item.get("answer_format") or "").lower() if item else ""
#             is_bfov = answer_format == "bfov_regression"
#             label_bfov = _extract_bfov(label_text) if is_bfov or not item else None

#             # -------------------------
#             # BFOV regression 分支
#             # -------------------------
#             if label_bfov is not None:
#                 bfov_total += 1
#                 pred_bfov = _extract_bfov(pred_text)
#                 iou = None

#                 if pred_bfov is not None:
#                     bfov_valid += 1
#                     iou = _bfov_iou(pred_bfov, label_bfov)
#                     if iou is not None:
#                         bfov_iou_sum += iou

#                 if task_id not in task_bfov_stats:
#                     task_bfov_stats[task_id] = {
#                         "count": 0.0,
#                         "valid_count": 0.0,
#                         "iou_sum": 0.0,
#                     }

#                 task_bfov_stats[task_id]["count"] += 1.0
#                 if pred_bfov is not None:
#                     task_bfov_stats[task_id]["valid_count"] += 1.0
#                 if iou is not None:
#                     task_bfov_stats[task_id]["iou_sum"] += float(iou)

#                 if print_predictions:
#                     item_id = item.get("item_id") if item else idx
#                     print(
#                         "[eval-sample] "
#                         f"item_id={item_id} "
#                         f"task_id={task_id} "
#                         f"type=bfov "
#                         f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                         f"pred_extracted={json.dumps(pred_bfov, ensure_ascii=False)} "
#                         f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                         f"gold_extracted={json.dumps(label_bfov, ensure_ascii=False)} "
#                         f"iou={json.dumps(iou, ensure_ascii=False)}",
#                         flush=True,
#                     )
#                 continue

#             # -------------------------
#             # Choice 分支
#             # -------------------------
#             sample_valid_keys = _get_item_valid_keys(item, valid_keys)
#             sample_option_text_to_key = (
#                 _get_item_option_text_to_key(item)
#                 if item is not None
#                 else (normalized_option_text_to_key or {})
#             )

#             pred_key = _extract_choice_key(
#                 pred_text,
#                 sample_valid_keys,
#                 sample_option_text_to_key,
#             )
#             label_key = _get_item_label_key(
#                 item,
#                 label_text,
#                 sample_valid_keys,
#                 sample_option_text_to_key,
#             )

#             if label_key is None:
#                 if print_predictions:
#                     item_id = item.get("item_id") if item else idx
#                     print(
#                         "[eval-sample] "
#                         f"item_id={item_id} "
#                         f"task_id={task_id} "
#                         f"type=choice "
#                         f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                         f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
#                         f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                         f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
#                         "correct=null "
#                         "note=\"gold key not parsable\"",
#                         flush=True,
#                     )
#                 continue

#             total += 1
#             if task_id not in task_choice_stats:
#                 task_choice_stats[task_id] = {"total": 0, "correct": 0}
#             task_choice_stats[task_id]["total"] += 1

#             is_correct = False
#             if pred_key is None:
#                 invalid_predictions += 1
#                 if print_predictions:
#                     item_id = item.get("item_id") if item else idx
#                     print(
#                         "[eval-sample] "
#                         f"item_id={item_id} "
#                         f"task_id={task_id} "
#                         f"type=choice "
#                         f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                         f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
#                         f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                         f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
#                         f"correct={is_correct}",
#                         flush=True,
#                     )
#                 continue

#             if pred_key == label_key:
#                 correct += 1
#                 task_choice_stats[task_id]["correct"] += 1
#                 is_correct = True

#             if print_predictions:
#                 item_id = item.get("item_id") if item else idx
#                 print(
#                     "[eval-sample] "
#                     f"item_id={item_id} "
#                     f"task_id={task_id} "
#                     f"type=choice "
#                     f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                     f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
#                     f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                     f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
#                     f"correct={is_correct}",
#                     flush=True,
#                 )

#         metrics = {
#             "exact_match": 0.0 if not label_texts else float(exact_match / len(label_texts)),
#             "choice_accuracy": 0.0 if total == 0 else float(correct / total),
#             "invalid_choice_rate": 0.0 if total == 0 else float(invalid_predictions / total),
#             "bfov_iou": 0.0 if bfov_total == 0 else float(bfov_iou_sum / bfov_total),
#             "bfov_valid_rate": 0.0 if bfov_total == 0 else float(bfov_valid / bfov_total),
#         }

#         # 每个 choice 任务的 accuracy
#         for task_id, stat in task_choice_stats.items():
#             safe_task_id = re.sub(r"[^a-zA-Z0-9_]+", "_", task_id)
#             task_total = stat["total"]
#             task_correct = stat["correct"]
#             metrics[f"task_acc_{safe_task_id}"] = (
#                 0.0 if task_total == 0 else float(task_correct / task_total)
#             )
#             metrics[f"task_count_{safe_task_id}"] = float(task_total)

#         # 每个 bfov 任务的 mean IoU
#         for task_id, stat in task_bfov_stats.items():
#             safe_task_id = re.sub(r"[^a-zA-Z0-9_]+", "_", task_id)
#             task_count = stat["count"]
#             task_valid_count = stat["valid_count"]
#             task_iou_sum = stat["iou_sum"]

#             metrics[f"task_bfov_iou_{safe_task_id}"] = (
#                 0.0 if task_count == 0 else float(task_iou_sum / task_count)
#             )
#             metrics[f"task_bfov_valid_rate_{safe_task_id}"] = (
#                 0.0 if task_count == 0 else float(task_valid_count / task_count)
#             )
#             metrics[f"task_bfov_count_{safe_task_id}"] = float(task_count)

#         return metrics

#     return compute_metrics

# def build_choice_accuracy_metrics(
#     tokenizer,
#     valid_keys: Optional[List[str]] = None,
#     option_text_to_key: Optional[Dict[str, str]] = None,
#     generation_mode: bool = False,
#     eval_items: Optional[List[Dict[str, Any]]] = None,
#     print_predictions: bool = False,
# ):
#     if not valid_keys:
#         valid_keys = ["A", "B", "C", "D", "E"]
#     valid_keys = [str(key).upper() for key in valid_keys]

#     normalized_option_text_to_key = None
#     if option_text_to_key:
#         normalized_option_text_to_key = {
#             _normalize_text(text): str(key).upper()
#             for text, key in option_text_to_key.items()
#             if text
#         }

#     def compute_metrics(eval_preds):
#         if generation_mode:
#             pred_texts, decoded_label_texts = _prepare_generation_eval_texts(tokenizer, eval_preds)
#         else:
#             pred_texts, decoded_label_texts = _prepare_eval_texts(tokenizer, eval_preds)

#         if eval_items is not None and len(eval_items) == len(pred_texts):
#             label_texts = [_get_item_target_text(item) for item in eval_items]
#         else:
#             label_texts = decoded_label_texts

#         total = 0
#         correct = 0
#         exact_match = 0
#         invalid_predictions = 0
#         bfov_total = 0
#         bfov_valid = 0
#         bfov_iou_sum = 0.0

#         for idx, (pred_text, label_text) in enumerate(zip(pred_texts, label_texts)):
#             item = eval_items[idx] if eval_items is not None and idx < len(eval_items) else None
#             if _normalize_text(pred_text) == _normalize_text(label_text):
#                 exact_match += 1

#             answer_format = str(item.get("answer_format") or "").lower() if item else ""
#             is_bfov = answer_format == "bfov_regression"
#             label_bfov = _extract_bfov(label_text) if is_bfov or not item else None
#             if label_bfov is not None:
#                 bfov_total += 1
#                 pred_bfov = _extract_bfov(pred_text)
#                 iou = None
#                 is_correct = False
#                 if pred_bfov is not None:
#                     bfov_valid += 1
#                     iou = _bfov_iou(pred_bfov, label_bfov)
#                     if iou is not None:
#                         bfov_iou_sum += iou
#                         is_correct = iou > 0.0
#                 if print_predictions:
#                     item_id = item.get("item_id") if item else idx
#                     print(
#                         "[eval-sample] "
#                         f"item_id={item_id} "
#                         f"type=bfov "
#                         f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                         f"pred_extracted={json.dumps(pred_bfov, ensure_ascii=False)} "
#                         f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                         f"gold_extracted={json.dumps(label_bfov, ensure_ascii=False)} "
#                         f"iou={json.dumps(iou, ensure_ascii=False)} "
#                         f"correct={is_correct}",
#                         flush=True,
#                     )
#                 continue

#             sample_valid_keys = _get_item_valid_keys(item, valid_keys)
#             sample_option_text_to_key = (
#                 _get_item_option_text_to_key(item)
#                 if item is not None
#                 else (normalized_option_text_to_key or {})
#             )
#             pred_key = _extract_choice_key(
#                 pred_text,
#                 sample_valid_keys,
#                 sample_option_text_to_key,
#             )
#             label_key = _get_item_label_key(
#                 item,
#                 label_text,
#                 sample_valid_keys,
#                 sample_option_text_to_key,
#             )
#             if label_key is None:
#                 if print_predictions:
#                     item_id = item.get("item_id") if item else idx
#                     print(
#                         "[eval-sample] "
#                         f"item_id={item_id} "
#                         f"type=choice "
#                         f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                         f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
#                         f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                         f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
#                         "correct=null "
#                         "note=\"gold key not parsable\"",
#                         flush=True,
#                     )
#                 continue

#             total += 1
#             is_correct = False
#             if pred_key is None:
#                 invalid_predictions += 1
#                 if print_predictions:
#                     item_id = item.get("item_id") if item else idx
#                     print(
#                         "[eval-sample] "
#                         f"item_id={item_id} "
#                         f"type=choice "
#                         f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                         f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
#                         f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                         f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
#                         f"correct={is_correct}",
#                         flush=True,
#                     )
#                 continue
#             if pred_key == label_key:
#                 correct += 1
#                 is_correct = True
#             if print_predictions:
#                 item_id = item.get("item_id") if item else idx
#                 print(
#                     "[eval-sample] "
#                     f"item_id={item_id} "
#                     f"type=choice "
#                     f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
#                     f"pred_extracted={json.dumps(pred_key, ensure_ascii=False)} "
#                     f"gold_raw={json.dumps(label_text, ensure_ascii=False)} "
#                     f"gold_extracted={json.dumps(label_key, ensure_ascii=False)} "
#                     f"correct={is_correct}",
#                     flush=True,
#                 )

#         return {
#             "exact_match": 0.0 if not label_texts else float(exact_match / len(label_texts)),
#             "choice_accuracy": 0.0 if total == 0 else float(correct / total),
#             "invalid_choice_rate": 0.0 if total == 0 else float(invalid_predictions / total),
#             "bfov_iou": 0.0 if bfov_total == 0 else float(bfov_iou_sum / bfov_total),
#             "bfov_valid_rate": 0.0 if bfov_total == 0 else float(bfov_valid / bfov_total),
#         }

#     return compute_metrics
