from typing import Any, Dict, List, Optional

import torch


class MultiModalDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        has_raw_generation = [
            "prompt_text" in f and "target_text" in f and "image_paths" in f
            for f in features
        ]
        if any(has_raw_generation):
            if not all(has_raw_generation):
                raise ValueError("Mixed raw-generation and tensor features in the same batch are not supported.")
            return {
                "prompt_text": [f["prompt_text"] for f in features],
                "target_text": [f["target_text"] for f in features],
                "image_paths": [f["image_paths"] for f in features],
                "item_id": [f.get("item_id") for f in features],
                "id": [f.get("id") for f in features],
                "options": [f.get("options", []) for f in features],
                "answer_format": [f.get("answer_format") for f in features],
                "answer": [f.get("answer") for f in features],
                "answer_text": [f.get("answer_text") for f in features],
                "task_id": [f.get("task_id") for f in features],
                "target_yaw_interval": [f.get("target_yaw_interval") for f in features],
                "target_pitch_interval": [f.get("target_pitch_interval") for f in features],
                "metadata": [f.get("metadata") for f in features],
            }

        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        if all("attention_mask" in f for f in features):
            attention_mask = [f["attention_mask"] for f in features]
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            )
        else:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        has_choice_keys = ["choice_keys" in f for f in features]
        if any(has_choice_keys) and not all(has_choice_keys):
            raise ValueError("Missing `choice_keys` in part of the evaluation batch.")
        if all(has_choice_keys):
            batch["choice_keys"] = [f["choice_keys"] for f in features]

        has_mm_token_types = ["mm_token_type_ids" in f for f in features]
        if any(has_mm_token_types) and not all(has_mm_token_types):
            raise ValueError("Missing `mm_token_type_ids` in part of the multimodal batch.")
        if all(has_mm_token_types):
            mm_token_type_ids = [f["mm_token_type_ids"] for f in features]
            mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
                mm_token_type_ids,
                batch_first=True,
                padding_value=0,
            )
            batch["mm_token_type_ids"] = mm_token_type_ids

        has_pixel_values = ["pixel_values" in f for f in features]
        if any(has_pixel_values) and not all(has_pixel_values):
            raise ValueError("Mixed text-only and image samples in the same batch are not supported.")
        if all(has_pixel_values):
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)

        has_grid = ["image_grid_thw" in f for f in features]
        if any(has_grid) and not all(has_grid):
            raise ValueError("Missing `image_grid_thw` in part of the multimodal batch.")
        if all(has_grid):
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        return batch
