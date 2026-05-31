from __future__ import annotations

import json
import re
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import Trainer

from utils import (
    _bfov_iou,
    _contains_circular_yaw,
    _contains_pitch,
    _extract_bfov,
    _extract_choice_key,
    _extract_yaw_pitch,
    _get_thinking360_intervals,
    _get_item_label_key,
    _get_item_option_text_to_key,
    _get_item_target_text,
    _get_item_valid_keys,
    _is_zero_pitch_interval,
    _normalize_signed_yaw,
)


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


class GenerationEvalTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_use_generation: bool = False,
        eval_method: str = "generation",
        eval_print_predictions: bool = False,
        generation_max_new_tokens: int = 16,
        generation_do_sample: bool = False,
        generation_num_beams: int = 1,
        generation_processor=None,
        **kwargs,
    ):
        self.eval_use_generation = eval_use_generation
        self.eval_method = str(eval_method or "generation").lower()
        self.eval_print_predictions = bool(eval_print_predictions)
        self.generation_max_new_tokens = generation_max_new_tokens
        self.generation_do_sample = generation_do_sample
        self.generation_num_beams = generation_num_beams
        self.generation_processor = generation_processor
        super().__init__(*args, **kwargs)

        if getattr(self.processing_class, "padding_side", None) is not None:
            self.processing_class.padding_side = "left"
        processor_tokenizer = getattr(self.generation_processor, "tokenizer", None)
        if processor_tokenizer is not None and getattr(processor_tokenizer, "padding_side", None) is not None:
            processor_tokenizer.padding_side = "left"

    @staticmethod
    def _model_device(model) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _slice_batch_value(value, indices):
        if isinstance(value, torch.Tensor):
            return value[indices]
        return value

    @contextmanager
    def _generation_mode(self, model):
        module = getattr(model, "module", model)
        gc_flags = []
        old_use_cache = None

        for candidate in (model, module):
            if candidate is None:
                continue
            has_gc = hasattr(candidate, "is_gradient_checkpointing")
            gc_enabled = bool(getattr(candidate, "is_gradient_checkpointing", False))
            gc_flags.append((candidate, has_gc, gc_enabled))
            if gc_enabled and hasattr(candidate, "gradient_checkpointing_disable"):
                candidate.gradient_checkpointing_disable()

        config = getattr(module, "config", None) or getattr(model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            old_use_cache = bool(config.use_cache)
            config.use_cache = True

        try:
            yield
        finally:
            if config is not None and old_use_cache is not None:
                config.use_cache = old_use_cache
            for candidate, has_gc, gc_enabled in gc_flags:
                if has_gc and gc_enabled and hasattr(candidate, "gradient_checkpointing_enable"):
                    candidate.gradient_checkpointing_enable()

    def _prepare_generation_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._prepare_inputs(inputs)

    def _run_generation(self, model, inputs: Dict[str, Any]) -> torch.Tensor:
        generation_inputs = self._prepare_generation_inputs(inputs)
        generation_inputs["max_new_tokens"] = int(self.generation_max_new_tokens)
        generation_inputs["do_sample"] = bool(self.generation_do_sample)
        generation_inputs["num_beams"] = int(self.generation_num_beams)

        try:
            with torch.inference_mode():
                with self._generation_mode(model):
                    generated_tokens = model.generate(**generation_inputs)

            max_prompt_len = generation_inputs["input_ids"].shape[-1]
            trimmed = []
            for row in generated_tokens:
                if row.shape[0] > max_prompt_len:
                    trimmed.append(row[max_prompt_len:])
                else:
                    trimmed.append(row.new_empty((0,), dtype=row.dtype))

            if trimmed:
                return torch.nn.utils.rnn.pad_sequence(
                    trimmed,
                    batch_first=True,
                    padding_value=int(self.processing_class.pad_token_id or self.processing_class.eos_token_id),
                )
            return generated_tokens[:, 0:0]

        except Exception as exc:
            input_ids = generation_inputs.get("input_ids", None)
            if not isinstance(input_ids, torch.Tensor) or input_ids.size(0) <= 1:
                raise

            rank = getattr(self.args, "process_index", 0)
            print(
                f"[rank{rank}][GenerationEvalTrainer] batched generation failed; "
                f"retrying per sample: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            per_sample_outputs: List[torch.Tensor] = []
            pad_id = int(self.processing_class.pad_token_id or self.processing_class.eos_token_id)

            for sample_idx in range(input_ids.size(0)):
                single_inputs: Dict[str, Any] = {}
                for key, value in generation_inputs.items():
                    if isinstance(value, torch.Tensor):
                        single_inputs[key] = value[sample_idx : sample_idx + 1]
                    elif isinstance(value, list):
                        single_inputs[key] = value[sample_idx : sample_idx + 1]
                    else:
                        single_inputs[key] = value

                try:
                    with torch.inference_mode():
                        with self._generation_mode(model):
                            single_generated = model.generate(**single_inputs)

                    single_prompt_len = single_inputs["input_ids"].shape[-1]
                    row = single_generated[0]
                    if row.shape[0] > single_prompt_len:
                        per_sample_outputs.append(row[single_prompt_len:])
                    else:
                        per_sample_outputs.append(row.new_empty((0,), dtype=row.dtype))
                except Exception as single_exc:
                    print(
                        f"[rank{rank}][GenerationEvalTrainer] sample {sample_idx} generation failed: {single_exc}",
                        flush=True,
                    )
                    per_sample_outputs.append(input_ids.new_full((1,), pad_id))

            return torch.nn.utils.rnn.pad_sequence(
                per_sample_outputs,
                batch_first=True,
                padding_value=pad_id,
            )

    def _encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = self.processing_class(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )["input_ids"]
        return encoded.to(self._model_device(self.model))

    def _load_images(self, image_paths_batch: List[List[str]]):
        images = []
        for image_paths in image_paths_batch:
            if not image_paths:
                images.append(None)
                continue
            if len(image_paths) != 1:
                raise ValueError("Only single-image eval samples are currently supported for batched generation.")
            with Image.open(image_paths[0]) as img:
                images.append(img.convert("RGB"))
        return images

    def _run_raw_generation(
        self,
        model,
        prompt_texts: List[str],
        image_paths_batch: List[List[str]],
    ) -> List[str]:
        images = self._load_images(image_paths_batch)
        has_any_image = any(image is not None for image in images)
        has_all_images = all(image is not None for image in images)

        if has_any_image and not has_all_images:
            outputs: List[str] = []
            for prompt_text, image_paths in zip(prompt_texts, image_paths_batch):
                outputs.extend(self._run_raw_generation(model, [prompt_text], [image_paths]))
            return outputs

        if has_any_image:
            if self.generation_processor is None:
                raise RuntimeError("generation_processor is required for multimodal generation eval")
            encoded = self.generation_processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            encoded = self.processing_class(
                prompt_texts,
                return_tensors="pt",
                padding=True,
            )

        generated_ids = self._run_generation(model, encoded)
        pred_texts = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)
        return [_strip_reasoning(text) for text in pred_texts]

    def _log_raw_eval_predictions(
        self,
        *,
        pred_texts: List[str],
        item_ids: List[Any],
        ids: List[Any],
        target_texts: List[str],
        options_batch: List[List[Dict[str, Any]]],
        answer_formats: List[Optional[str]],
        answers: List[Any],
        answer_texts: List[Optional[str]],
        task_ids: List[Any],
        target_yaw_intervals: List[Any],
        target_pitch_intervals: List[Any],
        metadata_batch: List[Any],
    ) -> None:
        if not self.eval_print_predictions:
            return

        rank = getattr(self.args, "process_index", 0)
        default_keys = ["A", "B", "C", "D", "E"]

        for idx, pred_text in enumerate(pred_texts):
            item = {
                "item_id": item_ids[idx] if idx < len(item_ids) else idx,
                "id": ids[idx] if idx < len(ids) else None,
                "options": options_batch[idx] if idx < len(options_batch) else [],
                "answer_format": answer_formats[idx] if idx < len(answer_formats) else None,
                "answer": answers[idx] if idx < len(answers) else None,
                "answer_text": answer_texts[idx] if idx < len(answer_texts) else None,
                "task_id": task_ids[idx] if idx < len(task_ids) else None,
                "target_yaw_interval": target_yaw_intervals[idx] if idx < len(target_yaw_intervals) else None,
                "target_pitch_interval": target_pitch_intervals[idx] if idx < len(target_pitch_intervals) else None,
                "metadata": metadata_batch[idx] if idx < len(metadata_batch) else None,
            }
            item_id = item.get("item_id") or item.get("id") or idx
            gold_raw = _get_item_target_text(item) or (target_texts[idx] if idx < len(target_texts) else "")
            answer_format = str(item.get("answer_format") or "").lower()

            intervals = _get_thinking360_intervals(item)
            if intervals is not None:
                pred_pair = _extract_yaw_pitch(pred_text)
                yaw_interval = intervals["yaw"]
                pitch_interval = intervals["pitch"]
                ignore_pitch = _is_zero_pitch_interval(pitch_interval)

                if pred_pair is None:
                    print(
                        "[eval-sample] "
                        f"rank={rank} "
                        f"item_id={item_id} "
                        "type=thinking360 "
                        f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                        f"yaw_interval={json.dumps(yaw_interval, ensure_ascii=False)} "
                        f"yaw_interval_signed={json.dumps([_normalize_signed_yaw(yaw_interval[0]), _normalize_signed_yaw(yaw_interval[1])], ensure_ascii=False)} "
                        f"pitch_interval={json.dumps(pitch_interval, ensure_ascii=False)} "
                        f"pitch_ignored={json.dumps(ignore_pitch)} "
                        "correct=false "
                        "note=\"prediction not parsable\"",
                        flush=True,
                    )
                    continue

                pred_yaw, pred_pitch = float(pred_pair[0]), float(pred_pair[1])
                yaw_ok = _contains_circular_yaw(pred_yaw, yaw_interval[0], yaw_interval[1])
                pitch_ok = True if ignore_pitch else _contains_pitch(pred_pitch, pitch_interval[0], pitch_interval[1])
                is_correct = bool(yaw_ok and pitch_ok)

                print(
                    "[eval-sample] "
                    f"rank={rank} "
                    f"item_id={item_id} "
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
                    f"correct={json.dumps(is_correct)}",
                    flush=True,
                )
                continue

            if answer_format == "bfov_regression":
                gold_extracted = _extract_bfov(gold_raw)
                pred_extracted = _extract_bfov(pred_text)
                iou = _bfov_iou(pred_extracted, gold_extracted) if pred_extracted is not None and gold_extracted is not None else None
                is_correct = bool(iou is not None and iou > 0.0)
                print(
                    "[eval-sample] "
                    f"rank={rank} "
                    f"item_id={item_id} "
                    "type=bfov "
                    f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                    f"pred_extracted={json.dumps(pred_extracted, ensure_ascii=False)} "
                    f"gold_raw={json.dumps(gold_raw, ensure_ascii=False)} "
                    f"gold_extracted={json.dumps(gold_extracted, ensure_ascii=False)} "
                    f"iou={json.dumps(iou, ensure_ascii=False)} "
                    f"correct={is_correct}",
                    flush=True,
                )
                continue

            sample_valid_keys = _get_item_valid_keys(item, default_keys)
            sample_option_text_to_key = _get_item_option_text_to_key(item)
            pred_extracted = _extract_choice_key(pred_text, sample_valid_keys, sample_option_text_to_key)
            gold_extracted = _get_item_label_key(item, gold_raw, sample_valid_keys, sample_option_text_to_key)
            is_correct = bool(pred_extracted is not None and gold_extracted is not None and pred_extracted == gold_extracted)
            print(
                "[eval-sample] "
                f"rank={rank} "
                f"item_id={item_id} "
                "type=choice "
                f"pred_raw={json.dumps(pred_text, ensure_ascii=False)} "
                f"pred_extracted={json.dumps(pred_extracted, ensure_ascii=False)} "
                f"gold_raw={json.dumps(gold_raw, ensure_ascii=False)} "
                f"gold_extracted={json.dumps(gold_extracted, ensure_ascii=False)} "
                f"correct={json.dumps(is_correct if gold_extracted is not None else None)}",
                flush=True,
            )

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        has_raw_generation = "prompt_text" in inputs and "target_text" in inputs and "image_paths" in inputs
        if has_raw_generation:
            prompt_texts = list(inputs["prompt_text"])
            target_texts = list(inputs["target_text"])
            image_paths_batch = list(inputs["image_paths"])
            item_ids = list(inputs.get("item_id", []))
            ids = list(inputs.get("id", []))
            options_batch = list(inputs.get("options", []))
            answer_formats = list(inputs.get("answer_format", []))
            answers = list(inputs.get("answer", []))
            answer_texts = list(inputs.get("answer_text", []))
            task_ids = list(inputs.get("task_id", []))
            target_yaw_intervals = list(inputs.get("target_yaw_interval", []))
            target_pitch_intervals = list(inputs.get("target_pitch_interval", []))
            metadata_batch = list(inputs.get("metadata", []))
            eos = self.processing_class.eos_token or ""

            if prediction_loss_only:
                return None, None, None

            pred_texts = self._run_raw_generation(model, prompt_texts, image_paths_batch)
            self._log_raw_eval_predictions(
                pred_texts=pred_texts,
                item_ids=item_ids,
                ids=ids,
                target_texts=target_texts,
                options_batch=options_batch,
                answer_formats=answer_formats,
                answers=answers,
                answer_texts=answer_texts,
                task_ids=task_ids,
                target_yaw_intervals=target_yaw_intervals,
                target_pitch_intervals=target_pitch_intervals,
                metadata_batch=metadata_batch,
            )
            label_texts = [text + eos for text in target_texts]

            predictions = self._encode_text_batch(pred_texts)
            labels = self._encode_text_batch(label_texts)
            return None, predictions, labels

        if not self.eval_use_generation and self.eval_method != "choice_scoring":
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels") if has_labels else None

        if prediction_loss_only:
            return None, None, None

        if self.eval_method == "choice_scoring":
            choice_keys = inputs.pop("choice_keys", None)
            if choice_keys is None:
                raise ValueError("`choice_keys` is required for choice_scoring evaluation.")

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            attention_mask = inputs["attention_mask"]
            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            next_token_logits = logits[batch_indices, last_indices]

            pred_token_seqs = [None] * len(choice_keys)
            pad_id = int(self.processing_class.pad_token_id or self.processing_class.eos_token_id)
            generation_indices = []
            for idx, (row_logits, sample_keys) in enumerate(zip(next_token_logits, choice_keys)):
                if sample_keys:
                    best_score = None
                    best_ids = None
                    for key in sample_keys:
                        candidate_ids = self.processing_class.encode(
                            str(key),
                            add_special_tokens=False,
                        )
                        if not candidate_ids:
                            continue
                        score = row_logits[int(candidate_ids[0])].item()
                        if best_score is None or score > best_score:
                            best_score = score
                            best_ids = candidate_ids
                    if best_ids is None:
                        best_ids = [pad_id]
                    pred_token_seqs[idx] = torch.tensor(best_ids, device=logits.device, dtype=torch.long)
                else:
                    generation_indices.append(idx)

            if generation_indices:
                sub_inputs = {
                    key: self._slice_batch_value(value, generation_indices)
                    for key, value in inputs.items()
                }
                generated_subset = self._run_generation(model, sub_inputs)
                for local_idx, global_idx in enumerate(generation_indices):
                    pred_token_seqs[global_idx] = generated_subset[local_idx]

            pred_token_seqs = [
                seq if seq is not None else torch.tensor([pad_id], device=logits.device, dtype=torch.long)
                for seq in pred_token_seqs
            ]

            predictions = torch.nn.utils.rnn.pad_sequence(
                pred_token_seqs,
                batch_first=True,
                padding_value=pad_id,
            )
            return None, predictions, labels

        generated_tokens = self._run_generation(model, inputs)
        return None, generated_tokens, labels
