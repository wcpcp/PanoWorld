#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

# Allow running as a script without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageFile

from erp_meta.dataset_realsee import iter_realsee_viewpoints
from erp_meta.io_utils import dump_json, load_json


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(value: str) -> str:
    value = (value or "").strip()
    value = _ID_RE.sub("_", value).strip("._")
    return value or "unknown"


def stem_from_value(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        return safe_id(Path(parsed.path).stem)
    return safe_id(Path(value).stem)


def detect_resize_backend() -> str | None:
    try:
        from PIL import Image as _PILImage  # noqa: F401
        return "pillow"
    except Exception:
        pass

    try:
        import cv2  # noqa: F401
        return "opencv"
    except Exception:
        pass

    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return None


def resize_image_in_place(path: str | Path, width: int, height: int) -> bool:
    path = Path(path)
    if not path.exists():
        print(f"[warn] image not found: {path}")
        return False

    backend = detect_resize_backend()
    if backend is None:
        raise RuntimeError("No resize backend found. Need Pillow / opencv-python / ffmpeg.")

    try:
        if backend == "pillow":
            with Image.open(path) as image:
                image.draft("RGB", (width, height))
                image = image.convert("RGB")
                if image.size == (width, height):
                    return True
                resized = image.resize((width, height), Image.Resampling.LANCZOS)
                save_kwargs = {}
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    save_kwargs["quality"] = 95
                resized.save(path, **save_kwargs)
            return True

        if backend == "opencv":
            import cv2

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"failed to read image: {path}")
            cur_h, cur_w = image.shape[:2]
            if (cur_w, cur_h) == (width, height):
                return True
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(path), resized):
                raise RuntimeError(f"failed to write resized image: {path}")
            return True

        temp_path = path.with_name(f"{path.stem}__tmp_resize{path.suffix}")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-vf",
                    f"scale={width}:{height}:flags=lanczos",
                    str(temp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            temp_path.replace(path)
            return True
        finally:
            if temp_path.exists():
                temp_path.unlink()

    except Exception as exc:
        print(f"[warn] resize failed for {path}: {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real_world_data", help="RealSee real_world_data root")
    ap.add_argument("--erp_json", default=None, help="Optional: JSON file with ERP image paths (for outdoor etc)")
    ap.add_argument("--out", default="results/00_scan_output.json", help="Output JSON path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of viewpoints (0=all)")
    ap.add_argument("--resize_width", type=int, default=2048, help="Resize width for erp_json branch")
    ap.add_argument("--resize_height", type=int, default=1024, help="Resize height for erp_json branch")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="If --out exists, load it and only append new viewpoints",
    )
    args = ap.parse_args()

    rows = []
    existing_keys: set[tuple[str, str]] = set()
    if args.skip_existing and Path(args.out).exists():
        try:
            prev = load_json(args.out)
            prev_rows = prev.get("viewpoints", []) if isinstance(prev, dict) else []
            for r in prev_rows:
                scene_id = str(r.get("scene_id", ""))
                viewpoint_id = str(r.get("viewpoint_id", ""))
                if scene_id and viewpoint_id:
                    existing_keys.add((scene_id, viewpoint_id))
            rows.extend(prev_rows)
            print(f"Loaded existing scan: {len(prev_rows)} viewpoints")
        except Exception as exc:
            print(f"[warn] Failed to load existing out: {args.out} ({exc})")

    if args.erp_json:
        erp_list = load_json(args.erp_json)
        if isinstance(erp_list, dict) and "images" in erp_list:
            erp_list = erp_list["images"]

        for i, item in enumerate(erp_list):
            pano_path = item["image_path"]
            source = item.get("source", "")

            # Resize can be enabled for plain ERP manifest inputs when needed.
            # resize_image_in_place(pano_path, args.resize_width, args.resize_height)

            scene_id = str(item.get("scene_id", "")).strip()
            viewpoint_id = str(item.get("viewpoint_id", "")).strip()

            if not scene_id:
                scene_id = stem_from_value(source) or Path(pano_path).stem
            if not viewpoint_id:
                viewpoint_id = Path(pano_path).stem

            key = (scene_id, viewpoint_id)
            if key in existing_keys:
                continue

            rows.append({
                "scene_id": scene_id,
                "viewpoint_id": viewpoint_id,
                "pano_path": pano_path,
                "source": source,
            })

            if args.limit and (len(rows) >= args.limit):
                break

        dump_json(args.out, {"erp_json": args.erp_json, "num_viewpoints": len(rows), "viewpoints": rows})
        print(f"Wrote {len(rows)} viewpoints -> {args.out}")
        return

    # 原有RealSee数据结构，不做 resize，不改逻辑
    for i, vp in enumerate(iter_realsee_viewpoints(args.root)):
        key = (vp.scene_id, vp.viewpoint_id)
        if key in existing_keys:
            continue
        rows.append(vp.__dict__)
        if args.limit and (i + 1) >= args.limit:
            break

    dump_json(args.out, {"root": args.root, "num_viewpoints": len(rows), "viewpoints": rows})
    print(f"Wrote {len(rows)} viewpoints -> {args.out}")


if __name__ == "__main__":
    main()
