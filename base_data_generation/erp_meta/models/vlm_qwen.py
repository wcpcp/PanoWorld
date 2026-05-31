from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Sequence


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = text.find("{", start + 1)

    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    preview = text[:300].replace("\n", "\\n")
    raise ValueError(f"No JSON object found in model output: {preview}")


@dataclass
class Qwen3VLVlm:
    model_dir: str
    backend: str = "transformers"
    device: str = "cuda"
    dtype: str = "auto"
    max_new_tokens: int = 512
    batch_size: int = 4
    base_url: str = ""
    model_name: str = ""
    api_key: str = "EMPTY"

    _model: Any = None
    _processor: Any = None

    def _lazy_load(self) -> None:
        if self.backend not in ("transformers", "torch"):
            return
        if self._model is not None:
            return
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            model_cls = Qwen3VLForConditionalGeneration
        except Exception:
            model_cls = None

        if model_cls is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(self.model_dir, torch_dtype="auto", device_map="auto")
        else:
            model = model_cls.from_pretrained(self.model_dir, torch_dtype="auto", device_map="auto")

        processor = AutoProcessor.from_pretrained(self.model_dir)
        if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "padding_side", None) != "left":
            processor.tokenizer.padding_side = "left"
        self._model = model
        self._processor = processor

    def _build_chat_text(self, prompt: str, num_images: int = 1) -> str:
        if num_images <= 0:
            raise ValueError("num_images must be >= 1")
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _decode_generated_rows(self, inputs: dict[str, Any], output_ids) -> list[str]:
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        texts_out: list[str] = []
        for row_ids, prompt_length in zip(output_ids, prompt_lengths):
            gen_ids = row_ids[int(prompt_length) :]
            texts_out.append(self._processor.decode(gen_ids, skip_special_tokens=True))
        return texts_out

    def _run_transformers_batch(self, image_groups: Sequence[Sequence[Any]], prompts: list[str]) -> list[str]:
        self._lazy_load()
        import torch

        if len(image_groups) != len(prompts):
            raise ValueError("image_groups and prompts must have the same length")
        if not prompts:
            return []

        flat_images: list[Any] = []
        texts: list[str] = []
        for image_group, prompt in zip(image_groups, prompts):
            if not image_group:
                raise ValueError("Every prompt must have at least one image")
            texts.append(self._build_chat_text(prompt, num_images=len(image_group)))
            flat_images.extend(list(image_group))

        inputs = self._processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self._decode_generated_rows(inputs, output_ids)

    def chat(self, image_pil, prompt: str) -> str:
        return self.chat_multi_image([image_pil], prompt)

    def chat_batch(self, image_pils, prompts: list[str]) -> list[str]:
        return self.chat_multi_image_batch([[img] for img in image_pils], prompts)

    def chat_multi_image(self, image_pils: Sequence[Any], prompt: str) -> str:
        return self.chat_multi_image_batch([list(image_pils)], [prompt])[0]

    def chat_multi_image_batch(self, image_groups: Sequence[Sequence[Any]], prompts: list[str]) -> list[str]:
        if len(image_groups) != len(prompts):
            raise ValueError("image_groups and prompts must have the same length")
        if not prompts:
            return []
        if self.backend in ("openai_compatible", "openai", "vllm"):
            return [self._chat_openai_compatible_multi(group, prompt) for group, prompt in zip(image_groups, prompts)]
        return self._run_transformers_batch(image_groups, prompts)

    def _chat_single_transformers(self, image_pil, prompt: str) -> str:
        return self._chat_single_transformers_multi([image_pil], prompt)

    def _chat_single_transformers_multi(self, image_pils: Sequence[Any], prompt: str) -> str:
        return self._run_transformers_batch([list(image_pils)], [prompt])[0]

    def _chat_openai_compatible(self, image_pil, prompt: str) -> str:
        return self._chat_openai_compatible_multi([image_pil], prompt)

    def _chat_openai_compatible_multi(self, image_pils: Sequence[Any], prompt: str) -> str:
        if not self.base_url:
            raise ValueError("base_url is required for openai_compatible backend")
        content = [{"type": "text", "text": prompt}]
        for image_pil in image_pils:
            buf = io.BytesIO()
            image_pil.save(buf, format="JPEG", quality=95)
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        payload = {
            "model": self.model_name or self.model_dir,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
        }
        req = urllib.request.Request(
            url=self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        content = obj["choices"][0]["message"]["content"]
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)

    def entity_enrich(self, image_pil, hint_label: str) -> Dict[str, Any]:
        return self.entity_enrich_batch([image_pil], [hint_label])[0]

    def entity_enrich_batch(self, image_pils, hint_labels: list[str]) -> list[Dict[str, Any]]:
        if len(image_pils) != len(hint_labels):
            raise ValueError("image_pils and hint_labels must have the same length")
        if not hint_labels:
            return []
        prompts = []
        for hint_label in hint_labels:
            prompts.append(
                "You are building high-quality grounded metadata for an equirectangular panorama. "
                "Given the image crop of ONE object instance (masked or zoomed), output STRICT JSON only with keys: "
                "name_refined (string), semantic_type (string), attributes (object), caption_brief (string), caption_dense (string), "
                "affordances (array of strings), semantic_confidence (number from 0 to 1), "
                "local_ref_short (string), local_ref_full (string), local_cues (array of strings), salient_parts (array of strings). "
                "Keep captions concise, factual, local, and grounded in visible evidence only. "
                "Do not invent global spatial relations that are not clearly visible in the crop. "
                f"Open-vocab hint label: {hint_label}. Output JSON only."
            )
        texts = self.chat_batch(image_pils, prompts)
        parsed: list[Dict[str, Any]] = []
        for image_pil, prompt, txt in zip(image_pils, prompts, texts):
            try:
                parsed.append(_extract_json(txt))
                continue
            except Exception:
                pass

            if self.backend in ("transformers", "torch"):
                retry_text = self._chat_single_transformers(
                    image_pil,
                    prompt + " Return exactly one JSON object only. Do not add markdown fences or any extra text.",
                )
            else:
                retry_text = self._chat_openai_compatible(
                    image_pil,
                    prompt + " Return exactly one JSON object only. Do not add markdown fences or any extra text.",
                )
            parsed.append(_extract_json(retry_text))
        return parsed

    def reground_bbox(self, image_pil, query: str) -> Dict[str, Any]:
        prompt = (
            "Locate the object described by the query in the given image. "
            "Return STRICT JSON only with keys: bbox_xyxy (4 numbers in pixels), confidence (0-1). "
            f"Query: {query}"
        )
        txt = self.chat(image_pil, prompt)
        return _extract_json(txt)
