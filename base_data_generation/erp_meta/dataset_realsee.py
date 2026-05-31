from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class RealSeeViewpoint:
    scene_id: str
    viewpoint_id: str
    root_dir: str
    viewpoint_dir: str
    pano_path: str
    pano_mask_path: Optional[str]
    depth_path: Optional[str]
    depth_scale_path: Optional[str]
    extrinsics_path: Optional[str]
    floor_path: Optional[str]


def iter_realsee_viewpoints(dataset_root: str | Path) -> Iterator[RealSeeViewpoint]:
    """Iterate RealSee viewpoints.

    Expected layout (observed):
      real_world_data/scene_xxxxx/viewpoints/<timestamp>/
        panoImage_*.jpg
        pano_mask.png (optional)
        depth_image.png (optional)
        depth_scale.txt (optional)
        extrinsics.txt (optional)
        floor.txt (optional)

    This function is permissive: it will yield only when a pano image exists.
    """
    dataset_root = Path(dataset_root)
    scene_dirs = list(dataset_root.glob("scene_*"))
    # Support synthetic data layout (synthetic_scene_XXXX)
    scene_dirs += list(dataset_root.glob("synthetic_scene_*"))
    # De-duplicate while preserving sort order
    scene_dirs = sorted({p.resolve() for p in scene_dirs})
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        vp_root = scene_dir / "viewpoints"
        if not vp_root.exists():
            continue
        for vp_dir in sorted(vp_root.iterdir()):
            if not vp_dir.is_dir():
                continue
            viewpoint_id = vp_dir.name
            pano_candidates = list(vp_dir.glob("panoImage_*.jpg")) + list(vp_dir.glob("panoImage_*.png"))
            if not pano_candidates:
                continue
            pano_path = str(sorted(pano_candidates)[0])
            pano_mask = vp_dir / "pano_mask.png"
            depth = vp_dir / "depth_image.png"
            depth_scale = vp_dir / "depth_scale.txt"
            extr = vp_dir / "extrinsics.txt"
            floor = vp_dir / "floor.txt"
            yield RealSeeViewpoint(
                scene_id=scene_id,
                viewpoint_id=viewpoint_id,
                root_dir=str(dataset_root),
                viewpoint_dir=str(vp_dir),
                pano_path=pano_path,
                pano_mask_path=str(pano_mask) if pano_mask.exists() else None,
                depth_path=str(depth) if depth.exists() else None,
                depth_scale_path=str(depth_scale) if depth_scale.exists() else None,
                extrinsics_path=str(extr) if extr.exists() else None,
                floor_path=str(floor) if floor.exists() else None,
            )
