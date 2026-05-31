from __future__ import annotations

from typing import Any, Dict

from .models.detector import Detector, ExternalCommandDetector, InternalWeDetectDetector, MockDetector
from .models.segmenter import ExternalCommandSegmenter, MockSegmenter, Segmenter
from .models.vlm_qwen import Qwen3VLVlm


def build_detector(cfg: Dict[str, Any]) -> Detector:
    mcfg = cfg["models"]["wedetect"]
    backend = mcfg.get("backend", "external_command")
    if backend == "mock":
        return Detector(MockDetector())
    if backend == "external_command":
        return Detector(
            ExternalCommandDetector(
                command_template=mcfg["command"],
                command_batch_template=mcfg.get("command_batch"),
                weights=mcfg.get("weights"),
            )
        )
    if backend == "internal_wedetect":
        return Detector(
            InternalWeDetectDetector(
                weights=mcfg["weights"],
                score_thre=float(mcfg.get("score_thre", 0.0)),
                num_proposals=int(mcfg.get("num_proposals", 300)),
                num_prompts=int(mcfg.get("num_prompts", 256)),
                prompt_dim=int(mcfg.get("prompt_dim", 768)),
                device=mcfg.get("device", "cuda"),
                use_amp=bool(mcfg.get("use_amp", False)),
            )
        )
    raise ValueError(f"Unknown detector backend: {backend}")


def build_segmenter(cfg: Dict[str, Any]) -> Segmenter:
    mcfg = cfg["models"]["sam3"]
    backend = mcfg.get("backend", "external_command")
    if backend == "mock":
        return Segmenter(MockSegmenter())
    if backend == "external_command":
        return Segmenter(ExternalCommandSegmenter(command_template=mcfg["command"], model_dir=mcfg.get("model_dir")))
    raise ValueError(f"Unknown segmenter backend: {backend}")


def build_vlm(cfg: Dict[str, Any]) -> Qwen3VLVlm:
    mcfg = cfg["models"].get("semantic_vlm", cfg["models"]["qwen3_vl"])
    return Qwen3VLVlm(
        model_dir=mcfg["model_dir"],
        backend=mcfg.get("backend", "transformers"),
        device=mcfg.get("device", "cuda"),
        dtype=mcfg.get("dtype", "auto"),
        max_new_tokens=int(mcfg.get("max_new_tokens", 512)),
        batch_size=int(mcfg.get("batch_size", 4)),
        base_url=mcfg.get("base_url", ""),
        model_name=mcfg.get("model_name", ""),
        api_key=mcfg.get("api_key", "EMPTY"),
    )
