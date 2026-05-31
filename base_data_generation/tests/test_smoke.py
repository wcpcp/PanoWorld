from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running tests without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_meta.dataset_realsee import iter_realsee_viewpoints
from erp_meta.erp_projection import backproject_mask_to_erp
from erp_meta.merge_entities import ProjectedInstance, merge_projected_instances
from erp_meta.overlap_verify import footprint_erp, overlap_pair_metrics
from erp_meta.rle import decode_binary_mask
from erp_meta.view_sampling import ViewSpec, make_cubemap_views, make_persp_views, make_ring_views, view_to_erp_maps
from erp_meta.models.detector import MockDetector
from erp_meta.models.segmenter import MockSegmenter
from erp_meta.types import DetBox


class TestSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(__file__).resolve().parent / "_tmp"
        shutil.rmtree(self.tmp, ignore_errors=True)
        (self.tmp / "real_world_data" / "scene_00001" / "viewpoints" / "1753781394").mkdir(parents=True)
        self.vp_dir = self.tmp / "real_world_data" / "scene_00001" / "viewpoints" / "1753781394"

        # create synthetic ERP 1600x800 to match RealSee example name
        w, h = 320, 160
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        Image.fromarray(img).save(self.vp_dir / "panoImage_1600.jpg", quality=90)

        # create a pano_mask that removes top/bottom
        m = np.zeros((h, w), dtype=np.uint8)
        m[h // 4 : 3 * h // 4, :] = 255
        Image.fromarray(m).save(self.vp_dir / "pano_mask.png")

    def test_scan_and_views_and_merge(self) -> None:
        vps = list(iter_realsee_viewpoints(self.tmp / "real_world_data"))
        self.assertEqual(len(vps), 1)
        pano_path = vps[0].pano_path

        out_views = self.tmp / "views"
        ring = make_ring_views(pano_path, str(out_views / "ring"), tile_w=128, overlap=0.5, include_seam=True)
        cube = make_cubemap_views(pano_path, str(out_views / "cube"), face_size=64)
        self.assertGreaterEqual(len(ring), 2)
        self.assertEqual(len(cube), 6)

        # use a single ring view for end-to-end mock detect->seg->backproject->merge
        view = ring[0]
        det = MockDetector()
        seg = MockSegmenter()

        dets = det.detect(view.image_path, str(self.tmp / "det.json"), queries=["object"])
        masks = seg.segment(view.image_path, dets, str(self.tmp / "seg.json"))
        self.assertEqual(len(masks), 1)

        map_x, map_y = view_to_erp_maps(view)
        m_view = decode_binary_mask(masks[0].rle)
        erp_mask = backproject_mask_to_erp(m_view, map_x, map_y, view.erp_w, view.erp_h)

        entities = merge_projected_instances([ProjectedInstance(view_id=view.view_id, label="object", score=0.5, mask_erp=erp_mask)])
        self.assertEqual(len(entities), 1)
        self.assertIn("mask_rle", entities[0])

    def test_persp8_pair_maps(self) -> None:
        vps = list(iter_realsee_viewpoints(self.tmp / "real_world_data"))
        pano_path = vps[0].pano_path
        pano_mask_path = vps[0].pano_mask_path

        out_views = self.tmp / "views_persp"
        views = make_persp_views(
            erp_path=pano_path,
            out_dir=str(out_views / "persp"),
            n_yaw=8,
            pano_mask_path=pano_mask_path,
            out_size=64,
            fov_deg=90.0,
            add_top_bottom=False,
            pair_adjacent=True,
        )
        self.assertGreaterEqual(len(views), 8)
        self.assertTrue(any(v.view_type == "persp_pair" for v in views))

        pair = next(v for v in views if v.view_type == "persp_pair")
        map_x, map_y = view_to_erp_maps(pair)
        self.assertEqual(map_x.shape, (64, 128))
        self.assertEqual(map_y.shape, (64, 128))

    def test_overlap_footprint_adjacent(self) -> None:
        vps = list(iter_realsee_viewpoints(self.tmp / "real_world_data"))
        pano_path = vps[0].pano_path
        pano_mask_path = vps[0].pano_mask_path

        out_views = self.tmp / "views_overlap"
        views = make_persp_views(
            erp_path=pano_path,
            out_dir=str(out_views / "persp"),
            n_yaw=8,
            pano_mask_path=pano_mask_path,
            out_size=64,
            fov_deg=90.0,
            pair_adjacent=False,
        )
        lateral = [v for v in views if v.view_type == "persp" and abs(float(v.pitch_deg or 0.0)) < 1e-3]
        self.assertGreaterEqual(len(lateral), 8)

        a = lateral[0]
        b = lateral[1]
        fa = footprint_erp(a)
        fb = footprint_erp(b)
        overlap = (fa & fb).astype(np.uint8)
        self.assertGreater(int(overlap.sum()), 0)

        # With empty foreground masks, agreement is perfect by definition.
        empty = np.zeros_like(overlap, dtype=np.uint8)
        m = overlap_pair_metrics(empty, empty, overlap, a.view_id, b.view_id)
        self.assertGreater(m.overlap_area, 0)
        self.assertEqual(m.fg_union, 0)
        self.assertAlmostEqual(m.fg_iou, 1.0)


if __name__ == "__main__":
    unittest.main()
