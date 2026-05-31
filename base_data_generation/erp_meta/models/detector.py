from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from ..types import DetBox


class DetectionBackend(Protocol):
    def detect(self, image_path: str, out_json: str, queries: Optional[list[str]] = None) -> List[DetBox]:
        ...

    def detect_batch(
        self, image_paths: list[str], out_jsons: list[str], queries: Optional[list[str]] = None
    ) -> None:
        ...


@dataclass
class ExternalCommandDetector:
    command_template: str
    weights: Optional[str] = None
    command_batch_template: Optional[str] = None

    def detect(self, image_path: str, out_json: str, queries: Optional[list[str]] = None) -> List[DetBox]:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        qpath = None
        if queries is not None:
            qpath = str(Path(out_json).with_suffix(".queries.json"))
            Path(qpath).write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")

        cmd = self.command_template.format(
            image=image_path,
            out_json=out_json,
            weights=(self.weights or ""),
            queries_json=(qpath or ""),
        )
        subprocess.run(cmd, shell=True, check=True)

        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        boxes: List[DetBox] = []
        for d in data:
            boxes.append(
                DetBox(
                    bbox_xyxy=tuple(map(float, d["bbox"])),
                    label=str(d.get("label", "")),
                    score=float(d.get("score", 0.0)),
                )
            )
        return boxes

    def detect_batch(self, image_paths: list[str], out_jsons: list[str], queries: Optional[list[str]] = None) -> None:
        if not self.command_batch_template:
            for image_path, out_json in zip(image_paths, out_jsons):
                self.detect(image_path=image_path, out_json=out_json, queries=queries)
            return

        out_dir = Path(out_jsons[0]).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        qpath = None
        if queries is not None:
            qpath = out_dir / f"_batch_queries_{os.getpid()}_{int(time.time()*1000)}.json"
            qpath.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")

        list_path = out_dir / f"_batch_list_{os.getpid()}_{int(time.time()*1000)}.json"
        items = [{"image": img, "out_json": out_json} for img, out_json in zip(image_paths, out_jsons)]
        list_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

        cmd = self.command_batch_template.format(
            image_list=str(list_path),
            out_dir=str(out_dir),
            weights=(self.weights or ""),
            queries_json=(str(qpath) if qpath else ""),
            batch_size=len(image_paths),
        )
        subprocess.run(cmd, shell=True, check=True)

        if qpath and qpath.exists():
            qpath.unlink()
        if list_path.exists():
            list_path.unlink()


@dataclass
class InternalWeDetectDetector:
    weights: str
    score_thre: float = 0.0
    num_proposals: int = 300
    num_prompts: int = 256
    prompt_dim: int = 768
    device: str = "cuda"
    use_amp: bool = False
    _model: Optional[Any] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from importlib import import_module
        import sys
        import torch

        root = Path(__file__).resolve().parents[3]
        wedetect_dir = root / "WeDetect"
        if str(wedetect_dir) not in sys.path:
            sys.path.insert(0, str(wedetect_dir))

        gp = import_module("generate_proposal")
        model_size = "base" if "base" in self.weights else "large"
        model = gp.SimpleYOLOWorldDetector(
            backbone_size=model_size,
            prompt_dim=self.prompt_dim,
            num_prompts=self.num_prompts,
            num_proposals=self.num_proposals,
        )
        checkpoint = torch.load(self.weights, map_location="cpu")

        keys = list(checkpoint.keys())
        for key in keys:
            if "backbone" in key:
                new_key = key.replace("backbone.image_model.model.", "backbone.")
                checkpoint[new_key] = checkpoint.pop(key)
        keys = list(checkpoint.keys())
        for key in keys:
            if "bbox_head" in key:
                new_key = key.replace("bbox_head.head_module.", "bbox_head.")
                new_key = new_key.replace("0.2.", "0.6.")
                new_key = new_key.replace("1.2.", "1.6.")
                new_key = new_key.replace("2.2.", "2.6.")
                new_key = new_key.replace("1.bn", "4")
                new_key = new_key.replace("1.conv", "3")
                new_key = new_key.replace("0.bn", "1")
                new_key = new_key.replace("0.conv", "0")
                checkpoint[new_key] = checkpoint.pop(key)

        model.load_state_dict(checkpoint, strict=False)
        self._model = model.to(self.device).eval()

    def _write_results(self, out_json: str, pred_bboxes, pred_scores) -> None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results = []
        for bbox, score in zip(pred_bboxes.tolist(), pred_scores.tolist()):
            results.append({"bbox": bbox, "label": "object", "score": float(score)})
        out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    def detect(self, image_path: str, out_json: str, queries: Optional[list[str]] = None) -> List[DetBox]:
        import torch

        self._load_model()
        assert self._model is not None

        with torch.no_grad():
            outputs = self._model([image_path])

        pred_bboxes = outputs[0]["bboxes"].float().cpu()
        pred_scores = outputs[0]["scores"].float().cpu()
        if self.score_thre > 0:
            mask = pred_scores > float(self.score_thre)
            pred_bboxes = pred_bboxes[mask]
            pred_scores = pred_scores[mask]

        self._write_results(out_json, pred_bboxes, pred_scores)
        boxes: List[DetBox] = []
        for bbox, score in zip(pred_bboxes.tolist(), pred_scores.tolist()):
            boxes.append(DetBox(bbox_xyxy=tuple(map(float, bbox)), label="object", score=float(score)))
        return boxes

    def detect_batch(self, image_paths: list[str], out_jsons: list[str], queries: Optional[list[str]] = None) -> None:
        import torch

        self._load_model()
        assert self._model is not None

        with torch.no_grad():
            outputs = self._model(image_paths)

        for out, out_json in zip(outputs, out_jsons):
            pred_bboxes = out["bboxes"].float().cpu()
            pred_scores = out["scores"].float().cpu()
            if self.score_thre > 0:
                mask = pred_scores > float(self.score_thre)
                pred_bboxes = pred_bboxes[mask]
                pred_scores = pred_scores[mask]
            self._write_results(out_json, pred_bboxes, pred_scores)


@dataclass
class MockDetector:
    """A tiny detector for dry-run/testing (no ML)."""

    def detect(self, image_path: str, out_json: str, queries: Optional[list[str]] = None) -> List[DetBox]:
        # Always return a center-ish box.
        from PIL import Image

        img = Image.open(image_path)
        w, h = img.size
        x1 = w * 0.25
        y1 = h * 0.25
        x2 = w * 0.75
        y2 = h * 0.75
        label = (queries[0] if queries else "object")
        out = [{"bbox": [x1, y1, x2, y2], "label": label, "score": 0.5}]
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return [DetBox(bbox_xyxy=(x1, y1, x2, y2), label=label, score=0.5)]


class Detector:
    def __init__(self, backend: DetectionBackend):
        self.backend = backend

    def detect_view(self, image_path: str, out_json: str, queries: Optional[list[str]] = None) -> List[DetBox]:
        return self.backend.detect(image_path=image_path, out_json=out_json, queries=queries)

    def detect_batch(self, image_paths: list[str], out_jsons: list[str], queries: Optional[list[str]] = None) -> None:
        if hasattr(self.backend, "detect_batch"):
            self.backend.detect_batch(image_paths=image_paths, out_jsons=out_jsons, queries=queries)
        else:
            for image_path, out_json in zip(image_paths, out_jsons):
                self.backend.detect(image_path=image_path, out_json=out_json, queries=queries)
