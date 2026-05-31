# ERP Metadata and Data Generation Pipeline

This module builds object-level metadata and training/evaluation data from equirectangular panorama (ERP) images. It keeps ERP as the global coordinate system and supports object detection, view projection, instance voting, semantic enrichment, local re-grounding, depth-aware spatial attributes, and QA export.

## Main Capabilities

- Scan RealSee-style panorama folders or a plain ERP image manifest.
- Generate perspective views from ERP images for detector/VLM processing.
- Run open-vocabulary detection on projected views.
- Merge projected detections back into ERP coordinates.
- Enrich objects with semantic captions, attributes, and re-grounding queries.
- Attach depth and 3D spatial statistics when depth maps are available.
- Export metadata and generate SFT/benchmark QA samples.

The pipeline is designed around these model roles:

- WeDetect / WeDetect-Uni: open-vocabulary detection.
- WeDetect-Ref: local re-grounding verification.
- Qwen3-VL: semantic enrichment and captioning.
- SAM-style segmentation: optional mask refinement.

## Expected Input Formats

### RealSee Layout

```text
real_world_data/
  scene_00001/
    viewpoints/
      1753781394/
        panoImage_1600.jpg
        pano_mask.png          # optional
        depth_image.png        # optional
        depth_scale.txt        # optional
```

### ERP Manifest

For custom or outdoor datasets, provide a JSON array:

```json
[
  {
    "image_path": "/path/to/erp_0001.jpg",
    "source": "erp_0001.jpg"
  }
]
```

## Configuration

Start from `configs/default.json` and update local model/data paths before running the full pipeline. The most important entries are detector weights, VLM model directories, optional segmenter paths, and output roots.

Generated experiment outputs should go under a local `results/` directory. They are ignored by git and should not be committed.

## Pipeline

Run commands from `base_data_generation/`.

### 1. Scan Input Panoramas

RealSee-style data:

```bash
python scripts/00_scan_realsee.py \
  --root /path/to/real_world_data \
  --out results/00_scan_output.json
```

Plain ERP manifest:

```bash
python scripts/00_scan_realsee.py \
  --erp_json /path/to/image_manifest.json \
  --out results/00_scan_output.json
```

### 2. Generate Perspective Views

```bash
python scripts/01_make_views.py \
  --scan_json results/00_scan_output.json \
  --out_dir results/01_make_views_output \
  --mode persp6 \
  --persp_fov 120 \
  --num_workers 32
```

If views already exist, rebuild only the index:

```bash
python scripts/01_make_views.py \
  --scan_json results/00_scan_output.json \
  --out_dir results/01_make_views_output \
  --rebuild_index
```

### 3. Detect Objects on Views

```bash
python scripts/02_detect.py \
  --cfg configs/default.json \
  --index_views results/01_make_views_output/index_views.json \
  --out_dir results/02_detect_output \
  --num_gpus 8 \
  --batch_size 16
```

### 4. Merge Detections into ERP Instances

```bash
python scripts/04c_instance_vote.py \
  --cfg configs/default.json \
  --index_views results/01_make_views_output/index_views.json \
  --det_root results/02_detect_output \
  --out_dir results/04c_instance_vote_output \
  --num_workers 32 \
  --save_views filtered
```

Add visualization flags only for debugging:

```bash
python scripts/04c_instance_vote.py \
  --cfg configs/default.json \
  --index_views results/01_make_views_output/index_views.json \
  --det_root results/02_detect_output \
  --out_dir results/04c_instance_vote_output \
  --viz_dir results/04c_instance_vote_viz \
  --viz_erp_final \
  --viz_overlay
```

### 5. Semantic Enrichment

Single panorama:

```bash
python scripts/05_region_semantic.py \
  --cfg configs/default.json \
  --instance_vote_json results/04c_instance_vote_output/scene_00001/1753781394/instance_vote.json \
  --views_json results/01_make_views_output/scene_00001/1753781394/views.json \
  --out_json results/05_semantic_output/scene_00001/1753781394/entities_enriched.json \
  --semantic_batch_size 8
```

Batch mode:

```bash
python scripts/05_region_semantic.py \
  --cfg configs/default.json \
  --index_views results/01_make_views_output/index_views.json \
  --instance_vote_root results/04c_instance_vote_output \
  --out_root results/05_semantic_output \
  --semantic_batch_size 16 \
  --num_gpus 8 \
  --skip_existing
```

### 6. Optional Re-grounding and Depth Attributes

Use local re-grounding when you need a verification pass over object regions:

```bash
python scripts/05b_local_reground.py \
  --cfg configs/default.json \
  --index_views results/01_make_views_output/index_views.json \
  --semantic_root results/05_semantic_output \
  --out_root results/05b_reground_output
```

Use depth attributes when each ERP record has a compatible depth map:

```bash
python scripts/05c_depth_spatial.py \
  --index_views results/01_make_views_output/index_views.json \
  --semantic_root results/05_semantic_output \
  --out_root results/05c_depth_spatial_output
```

### 7. Export Metadata and QA

```bash
python scripts/07_export_metadata.py \
  --index_views results/01_make_views_output/index_views.json \
  --semantic_root results/05_semantic_output \
  --out_json results/metadata_export.json
```

```bash
python scripts/06_generate_sft_qa.py \
  --metadata_json results/metadata_export.json \
  --out_json results/sft_qa.json
```

## Tests

The smoke tests use small mocked inputs and do not require committing generated artifacts:

```bash
python -m pytest tests
```

## Notes

- The repository keeps source code and lightweight configuration only.
- Do not commit `results/`, `_smoke/`, temporary test outputs, downloaded model weights, or Python caches.
- Some scripts keep path defaults for the original training environment; override them on the command line or in `configs/default.json`.
