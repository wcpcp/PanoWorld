import copy
import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import build_prompt_and_target


def _content_has_text(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and str(item.get("text") or "").strip():
                    return True
                if item.get("type") not in {"text", "image"} and any(str(v).strip() for v in item.values() if v is not None):
                    return True
            elif str(item).strip():
                return True
        return False
    return bool(str(content).strip())

def _find_system_message_index(messages: List[Dict[str, Any]]) -> int:
    for idx, message in enumerate(messages):
        if message.get("role") == "system":
            return idx
    return -1

def _read_records(path: str) -> List[Dict[str, Any]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if isinstance(data.get("records"), list):
                return list(data["records"])
            return [data]
        if isinstance(data, list):
            return list(data)
        raise ValueError(f"Unsupported JSON payload in {path}")

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    raise ValueError(f"Only .json/.jsonl are supported, got: {path}")


def _collect_images(record: Dict[str, Any]) -> List[str]:
    images: List[str] = []
    for key in ("images", "image_paths"):
        value = record.get(key)
        if isinstance(value, list):
            images.extend(str(v) for v in value if v)
    for key in ("image", "image_path"):
        value = record.get(key)
        if value:
            images.append(str(value))
    deduped: List[str] = []
    seen = set()
    for image in images:
        if image in seen:
            continue
        seen.add(image)
        deduped.append(image)
    return deduped
    # return images


def _resolve_media_path(path: str, image_root: Optional[str], data_root: str) -> str:
    if os.path.isabs(path):
        if os.path.exists(path):
            return path

        abs_candidates = [path]
        if path.startswith("/workspace/USB_data2/"):
            abs_candidates.append(path.replace("/workspace/USB_data2/", "/workspace/data_dir/USB_data2/", 1))
        elif path.startswith("/workspace/data_dir/USB_data2/"):
            abs_candidates.append(path.replace("/workspace/data_dir/USB_data2/", "/workspace/USB_data2/", 1))

        for candidate in abs_candidates:
            if os.path.exists(candidate):
                return candidate
        return abs_candidates[-1]

    candidates: List[str] = []
    if image_root:
        candidates.append(os.path.abspath(os.path.join(image_root, path)))
    candidates.append(os.path.abspath(os.path.join(data_root, path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _format_options(record: Dict[str, Any]) -> str:
    options = record.get("options")
    if not isinstance(options, list) or not options:
        return ""

    lines = ["Options:"]
    for option in options:
        if isinstance(option, dict):
            key = option.get("key")
            text = option.get("text")
            if key and text:
                lines.append(f"{key}. {text}")
            elif text:
                lines.append(f"- {text}")
        else:
            lines.append(f"- {option}")
    return "\n".join(lines)


def _ensure_image_placeholders(content: Any, image_count: int) -> Any:
    if image_count <= 0:
        return content

    if isinstance(content, str):
        missing_images = max(image_count - content.count("<image>"), 0)
        if missing_images == 0:
            return content
        prefix = "<image>" * missing_images
        if not content:
            return prefix
        return f"{prefix}\n{content}"

    if isinstance(content, list):
        new_content = copy.deepcopy(content)
        existing_images = sum(
            1 for item in new_content if isinstance(item, dict) and item.get("type") == "image"
        )
        for _ in range(max(image_count - existing_images, 0)):
            new_content.append({"type": "image"})
        return new_content

    return content


def _build_user_text(record: Dict[str, Any]) -> str:
    if record.get("prompt") is not None:
        base = str(record["prompt"])
    elif record.get("question") is not None:
        base = str(record["question"])
    else:
        raise ValueError("Each record must provide `messages`, `prompt`, or `question`.")

    option_block = _format_options(record)
    if option_block:
        base = f"{base}\n\n{option_block}\n\nAnswer with the option key only."
    return base


def _infer_choice_keys(record: Dict[str, Any]) -> List[str]:
    options = record.get("options")
    if isinstance(options, list) and options:
        keys = [
            str(option.get("key")).upper()
            for option in options
            if isinstance(option, dict) and option.get("key") is not None
        ]
        if keys:
            return keys

    answer_format = str(record.get("answer_format") or "").lower()
    if answer_format in {"4_way_multiple_choice", "multiple_choice", "5_way_multiple_choice"}:
        return ["A", "B", "C", "D", "E"]

    answer = record.get("answer")
    if isinstance(answer, str):
        answer = answer.strip().upper()
        if answer in {"A", "B", "C", "D", "E"}:
            return ["A", "B", "C", "D", "E"]

    return []


def _build_messages_from_record(
    record: Dict[str, Any],
    *,
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    answer = None
    for key in ("answer", "response", "answer_text"):
        if record.get(key) is not None:
            answer = str(record[key])
            break
    if answer is None:
        raise ValueError("Each record needs `answer`, `response`, or `answer_text` when `messages` is absent.")

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    image_count = len(_collect_images(record))
    user_text = _build_user_text(record)
    if image_count > 0:
        user_content: Any = [{"type": "image"} for _ in range(image_count)]
        if user_text:
            user_content.append({"type": "text", "text": user_text})
    else:
        user_content = user_text
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": answer})
    return messages


# def _normalize_messages(
#     messages: List[Dict[str, Any]],
#     *,
#     record: Dict[str, Any],
#     system_prompt: Optional[str],
#     auto_insert_media_placeholders: bool,
# ) -> List[Dict[str, Any]]:
#     normalized = copy.deepcopy(messages)

#     if system_prompt and not any(message.get("role") == "system" for message in normalized):
#         normalized.insert(0, {"role": "system", "content": system_prompt})

#     image_count = len(_collect_images(record))
#     for message in normalized:
#         if message.get("role") != "user":
#             continue
#         if auto_insert_media_placeholders:
#             message["content"] = _ensure_image_placeholders(message.get("content"), image_count)
#         break

#     return normalized


def _normalize_messages(
    messages: List[Dict[str, Any]],
    *,
    record: Dict[str, Any],
    system_prompt: Optional[str],
    auto_insert_media_placeholders: bool,
) -> List[Dict[str, Any]]:
    normalized = copy.deepcopy(messages)

    system_idx = _find_system_message_index(normalized)
    if system_prompt:
        if system_idx < 0:
            normalized.insert(0, {"role": "system", "content": system_prompt})
        elif not _content_has_text(normalized[system_idx].get("content")):
            normalized[system_idx]["content"] = system_prompt
    if system_idx > 0:
        system_message = normalized.pop(system_idx)
        normalized.insert(0, system_message)
    image_count = len(_collect_images(record))
    for message in normalized:
        if message.get("role") != "user":
            continue
        if auto_insert_media_placeholders:
            message["content"] = _ensure_image_placeholders(message.get("content"), image_count)
        break

    return normalized


def _normalize_record(
    record: Dict[str, Any],
    *,
    system_prompt: Optional[str],
    auto_insert_media_placeholders: bool,
    image_root: Optional[str],
    data_root: str,
) -> Dict[str, Any]:
    output = copy.deepcopy(record)
    images = [
        _resolve_media_path(path, image_root=image_root, data_root=data_root)
        for path in _collect_images(record)
    ]

    if record.get("messages") is not None:
        output["messages"] = _normalize_messages(
            record["messages"],
            record=record,
            system_prompt=system_prompt,
            auto_insert_media_placeholders=auto_insert_media_placeholders,
        )
    else:
        output["messages"] = _build_messages_from_record(
            record,
            system_prompt=system_prompt,
        )

    if images:
        output["images"] = images
    else:
        output.pop("images", None)

    return output


class SupervisedDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        tokenizer,
        image_root: Optional[str],
        image_token: str,
        model_max_length: int,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        prompt_format: str = "chat_template",
        image_processor_cfg: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        auto_insert_media_placeholders: bool = True,
        strict_image_checks: bool = True,
        skip_missing_images: bool = True,
        generation_eval: bool = False,
        raw_generation_eval: bool = False,
        disable_thinking: bool = False,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.image_token = image_token
        self.model_max_length = model_max_length
        self.prompt_format = prompt_format
        self.image_processor_cfg = image_processor_cfg or {}
        self.strict_image_checks = strict_image_checks
        self.skip_missing_images = skip_missing_images
        self.generation_eval = generation_eval
        self.raw_generation_eval = raw_generation_eval
        self.disable_thinking = disable_thinking
        self.data_root = os.path.dirname(os.path.abspath(jsonl_path))

        self._configure_image_processor()

        items = _read_records(jsonl_path)
        items = [
            _normalize_record(
                item,
                system_prompt=system_prompt,
                auto_insert_media_placeholders=auto_insert_media_placeholders,
                image_root=image_root,
                data_root=self.data_root,
            )
            for item in items
        ]

        if self.strict_image_checks:
            items = self._validate_media_paths(items)

        if shuffle:
            import random

            random.shuffle(items)

        if max_samples is not None:
            items = items[:max_samples]

        self.items = items

    def _validate_media_paths(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        missing_images = []
        existing_paths = set()
        missing_path_set = set()
        kept_items: List[Dict[str, Any]] = []

        for item in items:
            image_paths = item.get("images", []) or []
            item_has_missing = False
            for path in image_paths:
                if path in existing_paths:
                    continue
                if path in missing_path_set:
                    item_has_missing = True
                    continue
                if os.path.exists(path):
                    existing_paths.add(path)
                else:
                    missing_path_set.add(path)
                    missing_images.append(path)
                    item_has_missing = True
            if item_has_missing and self.skip_missing_images:
                continue
            kept_items.append(item)

        if missing_images and not self.skip_missing_images:
            preview = "\n".join(missing_images[:20])
            extra = "" if len(missing_images) <= 20 else f"\n... and {len(missing_images) - 20} more"
            raise FileNotFoundError(f"Missing image file(s):\n{preview}{extra}")

        if not kept_items:
            preview = "\n".join(missing_images[:20])
            extra = "" if len(missing_images) <= 20 else f"\n... and {len(missing_images) - 20} more"
            raise FileNotFoundError(
                "All samples were filtered out because their image files are missing."
                + (f"\nMissing image file(s):\n{preview}{extra}" if missing_images else "")
            )

        if missing_images and self.skip_missing_images:
            print(
                f"[dataset] skipped {len(items) - len(kept_items)} samples with missing images; "
                f"first missing path: {missing_images[0]}",
                flush=True,
            )

        return kept_items

    def _configure_image_processor(self):
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            return

        for key, value in self.image_processor_cfg.items():
            if key == "size" and isinstance(value, dict):
                size = getattr(image_processor, "size", None)
                if isinstance(size, dict):
                    size.update({k: v for k, v in value.items() if v is not None})
                else:
                    setattr(image_processor, "size", {k: v for k, v in value.items() if v is not None})
            elif hasattr(image_processor, key):
                setattr(image_processor, key, value)

    def __len__(self):
        return len(self.items)

    def _load_images(self, image_list: List[str]) -> List[Image.Image]:
        images = []
        for path in image_list:
            if self.strict_image_checks and not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.items[idx]
        image_list = sample.get("images", [])
        messages = sample["messages"]

        prompt_and_target = build_prompt_and_target(
            messages,
            self.prompt_format,
            processor=self.processor,
            disable_thinking=self.disable_thinking,
        )
        prompt_text = prompt_and_target["prompt"]
        target_text = prompt_and_target["target"]

        if self.generation_eval and self.raw_generation_eval:
            return {
                "prompt_text": prompt_text,
                "target_text": target_text,
                "image_paths": list(image_list),
                "item_id": sample.get("item_id"),
                "options": copy.deepcopy(sample.get("options", [])),
                "answer_format": sample.get("answer_format"),
                "answer": copy.deepcopy(sample.get("answer")),
                "answer_text": sample.get("answer_text"),
            }

        eos = self.tokenizer.eos_token or ""
        target_with_eos = target_text + eos

        images = self._load_images(image_list) if image_list else None
        input_text = prompt_text if self.generation_eval else prompt_text + target_with_eos
        if images is not None:
            encoded = self.processor(
                text=input_text,
                images=images,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_max_length,
            )
        else:
            encoded = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_max_length,
            )

        target_ids = self.tokenizer(
            target_with_eos,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_length,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        else:
            attention_mask = torch.ones_like(input_ids)

        if self.generation_eval:
            labels = target_ids.squeeze(0)
        else:
            labels = input_ids.clone()
            target_len = min(target_ids.size(1), input_ids.size(0))
            labels[:-target_len] = -100

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if self.generation_eval:
            # Always attach choice_keys during mixed eval so the collator can
            # batch choice questions and non-choice questions together. The
            # trainer will use choice scoring for non-empty lists and fall back
            # to generation for empty lists (for example BFOV regression items).
            batch["choice_keys"] = _infer_choice_keys(sample)
        if "mm_token_type_ids" in encoded:
            batch["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)
        if "pixel_values" in encoded:
            batch["pixel_values"] = encoded["pixel_values"]
        if "image_grid_thw" in encoded:
            batch["image_grid_thw"] = encoded["image_grid_thw"]

        return batch
