from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
from PIL import Image

from .erp_projection import FaceName, cubemap_remap, remap_erp_to_view
from .io_utils import ensure_dir
from .perspective import perspective_remap, yaw_candidates, yaw_candidates_4


ViewType = Literal["ring", "seam", "cubemap", "persp", "persp_pair"]


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    view_type: ViewType
    image_path: str
    erp_path: str
    erp_w: int
    erp_h: int
    # ring/seam
    x0: Optional[int] = None
    w: Optional[int] = None
    # cubemap
    face: Optional[FaceName] = None
    size: Optional[int] = None
    # perspective
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    fov_deg: Optional[float] = None
    # perspective pair (stitched two adjacent perspective views horizontally)
    yaw2_deg: Optional[float] = None
    pitch2_deg: Optional[float] = None
    fov2_deg: Optional[float] = None
    # seam only
    left_w: Optional[int] = None
    right_w: Optional[int] = None

    # optional pano valid mask (ERP-space); used by downstream merge for filtering
    pano_mask_path: Optional[str] = None


def _load_erp_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _load_erp_mask01(path: str) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.array(m)
    return (arr > 0).astype(np.uint8)


def apply_pano_mask(erp_rgb: np.ndarray, pano_mask_path: Optional[str]) -> np.ndarray:
    if not pano_mask_path:
        return erp_rgb
    m = _load_erp_mask01(pano_mask_path)
    if m.shape[:2] != erp_rgb.shape[:2]:
        return erp_rgb
    return (erp_rgb * m[..., None]).astype(np.uint8)


def make_persp4_views(
    erp_path: str,
    out_dir: str,
    pano_mask_path: Optional[str] = None,
    out_size: int = 0,
    fov_deg: float = 100.0,
    add_top_bottom: bool = False,
) -> List[ViewSpec]:
    """Dense360-style view sampling: 4 lateral perspective views (optionally +up/down).

    - yaw: 0/90/180/270
    - pitch: 0 (lateral)
    - overlap controlled by fov_deg (>90 => overlap)
    - out_size default: ERP height (square)

    pano_mask_path: if provided, will be applied to ERP pixels before sampling.
    """
    erp = _load_erp_rgb(erp_path)
    erp_h, erp_w = erp.shape[:2]
    if out_size <= 0:
        out_size = erp_h

    erp_mask01: Optional[np.ndarray] = None
    if pano_mask_path:
        try:
            erp_mask01 = _load_erp_mask01(pano_mask_path)
        except Exception:
            erp_mask01 = None

    erp = apply_pano_mask(erp, pano_mask_path)

    out_dir_p = ensure_dir(out_dir)
    views: List[ViewSpec] = []

    for yaw in yaw_candidates_4():
        map_x, map_y = perspective_remap(out_size, fov_deg, yaw_deg=yaw, pitch_deg=0.0, erp_w=erp_w, erp_h=erp_h)
        view = remap_erp_to_view(erp, map_x, map_y)
        view_id = f"persp_y{int(yaw):03d}_p000"
        img_path = str(out_dir_p / f"{view_id}.jpg")
        Image.fromarray(view).save(img_path, quality=95)
        views.append(
            ViewSpec(
                view_id=view_id,
                view_type="persp",
                image_path=img_path,
                erp_path=erp_path,
                erp_w=erp_w,
                erp_h=erp_h,
                size=int(out_size),
                yaw_deg=float(yaw),
                pitch_deg=0.0,
                fov_deg=float(fov_deg),
                pano_mask_path=pano_mask_path,
            )
        )

    if add_top_bottom:
        tb_min_valid_frac = 0.01
        for pitch, tag in [(90.0, "up"), (-90.0, "down")]:
            map_x, map_y = perspective_remap(out_size, fov_deg, yaw_deg=0.0, pitch_deg=pitch, erp_w=erp_w, erp_h=erp_h)
            if erp_mask01 is not None and erp_mask01.shape[:2] == (erp_h, erp_w):
                ex = np.round(map_x).astype(np.int32) % erp_w
                ey = np.clip(np.round(map_y).astype(np.int32), 0, erp_h - 1)
                valid_frac = float(erp_mask01[ey, ex].mean())
                if valid_frac < tb_min_valid_frac:
                    continue
            view = remap_erp_to_view(erp, map_x, map_y)
            view_id = f"persp_{tag}"
            img_path = str(out_dir_p / f"{view_id}.jpg")
            Image.fromarray(view).save(img_path, quality=95)
            views.append(
                ViewSpec(
                    view_id=view_id,
                    view_type="persp",
                    image_path=img_path,
                    erp_path=erp_path,
                    erp_w=erp_w,
                    erp_h=erp_h,
                    size=int(out_size),
                    yaw_deg=0.0,
                    pitch_deg=float(pitch),
                    fov_deg=float(fov_deg),
                    pano_mask_path=pano_mask_path,
                )
            )

    return views


def make_persp_views(
    erp_path: str,
    out_dir: str,
    n_yaw: int = 8,
    pano_mask_path: Optional[str] = None,
    out_size: int = 0,
    fov_deg: float = 90.0,
    add_top_bottom: bool = False,
    pair_adjacent: bool = False,
) -> List[ViewSpec]:
    """General perspective multi-face sampling.

    - lateral faces: pitch=0, yaw evenly spaced by 360/n_yaw
    - overlap is determined by (fov_deg - 360/n_yaw)
    - optional top/bottom (pitch=+90/-90) when add_top_bottom=True
    - optional adjacent-pair stitched views when pair_adjacent=True
      (NOT a 180-degree single perspective; it is two perspective images concatenated)
    """
    erp = _load_erp_rgb(erp_path)
    erp_h, erp_w = erp.shape[:2]
    if out_size <= 0:
        out_size = erp_h

    erp_mask01: Optional[np.ndarray] = None
    if pano_mask_path:
        try:
            erp_mask01 = _load_erp_mask01(pano_mask_path)
        except Exception:
            erp_mask01 = None

    erp = apply_pano_mask(erp, pano_mask_path)
    out_dir_p = ensure_dir(out_dir)
    views: List[ViewSpec] = []

    yaws = yaw_candidates(n_yaw)
    # 1) single lateral faces
    for idx, yaw in enumerate(yaws):
        map_x, map_y = perspective_remap(out_size, fov_deg, yaw_deg=yaw, pitch_deg=0.0, erp_w=erp_w, erp_h=erp_h)
        view = remap_erp_to_view(erp, map_x, map_y)
        view_id = f"persp_{n_yaw}f_{idx:02d}_y{int(round(yaw))%360:03d}_p000"
        img_path = str(out_dir_p / f"{view_id}.jpg")
        Image.fromarray(view).save(img_path, quality=95)
        views.append(
            ViewSpec(
                view_id=view_id,
                view_type="persp",
                image_path=img_path,
                erp_path=erp_path,
                erp_w=erp_w,
                erp_h=erp_h,
                size=int(out_size),
                yaw_deg=float(yaw),
                pitch_deg=0.0,
                fov_deg=float(fov_deg),
                pano_mask_path=pano_mask_path,
            )
        )

    # 2) optional stitched adjacent pairs for detection/verification
    if pair_adjacent and len(yaws) >= 2:
        for i in range(len(yaws)):
            yaw1 = float(yaws[i])
            yaw2 = float(yaws[(i + 1) % len(yaws)])
            map_x1, map_y1 = perspective_remap(out_size, fov_deg, yaw_deg=yaw1, pitch_deg=0.0, erp_w=erp_w, erp_h=erp_h)
            map_x2, map_y2 = perspective_remap(out_size, fov_deg, yaw_deg=yaw2, pitch_deg=0.0, erp_w=erp_w, erp_h=erp_h)
            v1 = remap_erp_to_view(erp, map_x1, map_y1)
            v2 = remap_erp_to_view(erp, map_x2, map_y2)
            stitched = np.concatenate([v1, v2], axis=1)
            view_id = f"persp_pair_{n_yaw}f_{i:02d}_y{int(round(yaw1))%360:03d}_y{int(round(yaw2))%360:03d}_p000"
            img_path = str(out_dir_p / f"{view_id}.jpg")
            Image.fromarray(stitched).save(img_path, quality=95)
            views.append(
                ViewSpec(
                    view_id=view_id,
                    view_type="persp_pair",
                    image_path=img_path,
                    erp_path=erp_path,
                    erp_w=erp_w,
                    erp_h=erp_h,
                    size=int(out_size),
                    yaw_deg=yaw1,
                    pitch_deg=0.0,
                    fov_deg=float(fov_deg),
                    yaw2_deg=yaw2,
                    pitch2_deg=0.0,
                    fov2_deg=float(fov_deg),
                    pano_mask_path=pano_mask_path,
                )
            )

    # 3) optional top/bottom (only keep if pano mask supports it)
    if add_top_bottom:
        tb_min_valid_frac = 0.01
        for pitch, tag in [(90.0, "up"), (-90.0, "down")]:
            map_x, map_y = perspective_remap(out_size, fov_deg, yaw_deg=0.0, pitch_deg=pitch, erp_w=erp_w, erp_h=erp_h)
            if erp_mask01 is not None and erp_mask01.shape[:2] == (erp_h, erp_w):
                ex = np.round(map_x).astype(np.int32) % erp_w
                ey = np.clip(np.round(map_y).astype(np.int32), 0, erp_h - 1)
                valid_frac = float(erp_mask01[ey, ex].mean())
                if valid_frac < tb_min_valid_frac:
                    continue
            view = remap_erp_to_view(erp, map_x, map_y)
            view_id = f"persp_{n_yaw}f_{tag}"
            img_path = str(out_dir_p / f"{view_id}.jpg")
            Image.fromarray(view).save(img_path, quality=95)
            views.append(
                ViewSpec(
                    view_id=view_id,
                    view_type="persp",
                    image_path=img_path,
                    erp_path=erp_path,
                    erp_w=erp_w,
                    erp_h=erp_h,
                    size=int(out_size),
                    yaw_deg=0.0,
                    pitch_deg=float(pitch),
                    fov_deg=float(fov_deg),
                    pano_mask_path=pano_mask_path,
                )
            )

    return views


def make_ring_views(
    erp_path: str,
    out_dir: str,
    tile_w: int,
    overlap: float = 0.5,
    include_seam: bool = True,
) -> List[ViewSpec]:
    erp = _load_erp_rgb(erp_path)
    erp_h, erp_w = erp.shape[:2]
    out_dir_p = ensure_dir(out_dir)

    stride = max(1, int(tile_w * (1 - overlap)))
    n_tiles = max(1, math.ceil((erp_w - tile_w) / stride) + 1)

    views: List[ViewSpec] = []
    for i in range(n_tiles):
        x0 = min(i * stride, erp_w - tile_w)
        tile = erp[:, x0 : x0 + tile_w]
        view_id = f"ring_{i:03d}"
        img_path = str(out_dir_p / f"{view_id}.jpg")
        Image.fromarray(tile).save(img_path, quality=95)
        views.append(
            ViewSpec(
                view_id=view_id,
                view_type="ring",
                image_path=img_path,
                erp_path=erp_path,
                erp_w=erp_w,
                erp_h=erp_h,
                x0=int(x0),
                w=int(tile_w),
            )
        )

    if include_seam and tile_w < erp_w:
        right_w = tile_w // 2
        left_w = tile_w - right_w
        tile = np.concatenate([erp[:, erp_w - right_w :], erp[:, :left_w]], axis=1)
        view_id = "seam_000"
        img_path = str(out_dir_p / f"{view_id}.jpg")
        Image.fromarray(tile).save(img_path, quality=95)
        views.append(
            ViewSpec(
                view_id=view_id,
                view_type="seam",
                image_path=img_path,
                erp_path=erp_path,
                erp_w=erp_w,
                erp_h=erp_h,
                left_w=int(left_w),
                right_w=int(right_w),
                w=int(tile_w),
            )
        )

    return views


def make_cubemap_views(erp_path: str, out_dir: str, face_size: int) -> List[ViewSpec]:
    erp = _load_erp_rgb(erp_path)
    erp_h, erp_w = erp.shape[:2]
    out_dir_p = ensure_dir(out_dir)

    faces: List[FaceName] = ["front", "right", "back", "left", "up", "down"]
    views: List[ViewSpec] = []
    for face in faces:
        map_x, map_y = cubemap_remap(face, face_size, erp_w, erp_h)
        view = remap_erp_to_view(erp, map_x, map_y)
        view_id = f"cube_{face}"
        img_path = str(out_dir_p / f"{view_id}.jpg")
        Image.fromarray(view).save(img_path, quality=95)
        views.append(
            ViewSpec(
                view_id=view_id,
                view_type="cubemap",
                image_path=img_path,
                erp_path=erp_path,
                erp_w=erp_w,
                erp_h=erp_h,
                face=face,
                size=int(face_size),
            )
        )

    return views


def view_to_erp_maps(view: ViewSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (map_x, map_y) mapping from view pixels to ERP pixels (float32).

    Used for backprojecting masks / boxes.
    """
    if view.view_type == "ring":
        assert view.x0 is not None and view.w is not None
        h = view.erp_h
        w = view.w
        jj, ii = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (jj + view.x0).astype(np.float32)
        map_y = ii.astype(np.float32)
        return map_x, map_y

    if view.view_type == "seam":
        assert view.left_w is not None and view.right_w is not None and view.w is not None
        h = view.erp_h
        w = view.w
        jj, ii = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = np.empty_like(jj)
        # left part comes from right side of ERP
        map_x[:, : view.right_w] = (view.erp_w - view.right_w) + jj[:, : view.right_w]
        # right part comes from left side of ERP
        map_x[:, view.right_w :] = jj[:, view.right_w :] - view.right_w
        map_y = ii.astype(np.float32)
        return map_x.astype(np.float32), map_y

    if view.view_type == "cubemap":
        assert view.face is not None and view.size is not None
        return cubemap_remap(view.face, view.size, view.erp_w, view.erp_h)

    if view.view_type == "persp":
        assert view.size is not None and view.fov_deg is not None and view.yaw_deg is not None and view.pitch_deg is not None
        return perspective_remap(view.size, view.fov_deg, view.yaw_deg, view.pitch_deg, view.erp_w, view.erp_h)

    if view.view_type == "persp_pair":
        assert (
            view.size is not None
            and view.fov_deg is not None
            and view.yaw_deg is not None
            and view.pitch_deg is not None
            and view.yaw2_deg is not None
            and view.pitch2_deg is not None
        )
        fov2 = view.fov2_deg if view.fov2_deg is not None else view.fov_deg
        map_x1, map_y1 = perspective_remap(view.size, view.fov_deg, view.yaw_deg, view.pitch_deg, view.erp_w, view.erp_h)
        map_x2, map_y2 = perspective_remap(view.size, fov2, view.yaw2_deg, view.pitch2_deg, view.erp_w, view.erp_h)
        map_x = np.concatenate([map_x1, map_x2], axis=1)
        map_y = np.concatenate([map_y1, map_y2], axis=1)
        return map_x.astype(np.float32), map_y.astype(np.float32)

    raise ValueError(f"Unknown view_type: {view.view_type}")
