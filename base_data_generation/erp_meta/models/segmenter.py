from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

from ..types import DetBox, SegMask


class SegmentationBackend(Protocol):
    def segment(self, image_path: str, boxes: List[DetBox], out_json: str) -> List[SegMask]:
        ...


@dataclass
class ExternalCommandSegmenter:
    command_template: str
    model_dir: Optional[str] = None

    def segment(self, image_path: str, boxes: List[DetBox], out_json: str) -> List[SegMask]:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        boxes_json = str(Path(out_json).with_suffix(".boxes.json"))
        Path(boxes_json).write_text(
            json.dumps(
                [{"bbox": list(b.bbox_xyxy), "label": b.label, "score": b.score} for b in boxes],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        cmd = self.command_template.format(
            model_dir=(self.model_dir or ""),
            image=image_path,
            boxes_json=boxes_json,
            out_json=out_json,
        )
        subprocess.run(cmd, shell=True, check=True)

        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        masks: List[SegMask] = []
        for d in data:
            masks.append(
                SegMask(
                    rle=d["rle"],
                    bbox_xyxy=tuple(map(float, d["bbox"])),
                    score=float(d.get("score", 0.0)),
                    label=d.get("label"),
                )
            )
        return masks


@dataclass
class MockSegmenter:
    """Dry-run segmenter: turns each bbox into a filled rectangle mask RLE."""

    def segment(self, image_path: str, boxes: List[DetBox], out_json: str) -> List[SegMask]:
        from PIL import Image

        from ..rle import encode_binary_mask

        img = Image.open(image_path)
        w, h = img.size
        masks: List[SegMask] = []
        for b in boxes:
            x1, y1, x2, y2 = map(int, map(round, b.bbox_xyxy))
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            rle = encode_binary_mask(mask)
            masks.append(SegMask(rle=rle, bbox_xyxy=b.bbox_xyxy, score=b.score, label=b.label))

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps(
                [{"rle": m.rle, "bbox": list(m.bbox_xyxy), "score": m.score, "label": m.label} for m in masks],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return masks


class Segmenter:
    def __init__(self, backend: SegmentationBackend):
        self.backend = backend

    def segment_view(self, image_path: str, boxes: List[DetBox], out_json: str) -> List[SegMask]:
        return self.backend.segment(image_path=image_path, boxes=boxes, out_json=out_json)


# numpy is optional for external backend, but needed for mock
import numpy as np
