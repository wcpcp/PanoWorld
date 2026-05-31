#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


DEMO_PAIRS = [
    ("01.jpg", "000001.png"),
    ("03.jpg", "000002.png"),
    ("generated_00.jpg", "000003.png"),
]


def build_preview(input_dir: Path, depth_dir: Path, output_path: Path) -> None:
    rows = []

    for input_name, depth_name in DEMO_PAIRS:
        input_path = input_dir / input_name
        depth_path = depth_dir / depth_name
        if not input_path.exists() or not depth_path.exists():
            continue

        inp = Image.open(input_path).convert("RGB")
        dep = Image.open(depth_path).convert("RGB")
        target_h = 360
        inp = ImageOps.contain(inp, (9999, target_h))
        dep = ImageOps.contain(dep, (9999, target_h))

        canvas = Image.new("RGB", (inp.width + dep.width + 24, target_h + 50), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), f"Input: {input_name}", fill="black")
        draw.text((inp.width + 24, 12), f"Depth: {depth_name}", fill="black")
        canvas.paste(inp, (0, 50))
        canvas.paste(dep, (inp.width + 24, 50))
        rows.append(canvas)

    if not rows:
        raise FileNotFoundError("No valid input/depth pairs were found for preview generation.")

    max_w = max(im.width for im in rows)
    spacing = 20
    sheet_h = sum(im.height for im in rows) + spacing * (len(rows) - 1)
    sheet = Image.new("RGB", (max_w, sheet_h), "#f4f4f4")

    y = 0
    for im in rows:
        sheet.paste(im, ((max_w - im.width) // 2, y))
        y += im.height + spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_preview(Path(args.input_dir), Path(args.depth_dir), Path(args.output))


if __name__ == "__main__":
    main()
