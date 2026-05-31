"""ERP panorama metadata pipeline (RealSee).

This package is intentionally model-agnostic: detectors/segmenters/VLMs are pluggable backends.
"""

from .dataset_realsee import RealSeeViewpoint, iter_realsee_viewpoints
