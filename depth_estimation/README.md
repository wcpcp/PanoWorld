# Panoramic Depth Estimation

This module wraps the official DAP codebase for panoramic depth inference and adds PanoWorld-specific batch utilities for local demos, server inference, multi-GPU sharding, and JSON manifest updates.

## Contents

| Path | Description |
| --- | --- |
| `DAP/` | Vendored DAP source code used by the inference wrappers. |
| `inputs/official_examples/` | Lightweight demo panoramas. |
| `models/hf_model/` | Expected local location for downloaded DAP weights. The weight file is not committed. |
| `outputs/` | Runtime inference outputs. Ignored by git. |
| `run_demo.sh` | Minimal demo entry point. |
| `server_infer_worker.py` | Single-worker inference process used by the launcher. |
| `multi_gpu_infer.py` | Multi-GPU sharding launcher. |
| `json_multi_gpu_infer.py` | Reads a JSON image manifest, writes depth files, and updates the manifest with `depth_path`. |
| `run_server_multi_gpu.sh` | Shell entry point for server batch inference. |
| `requirements_server.txt` | Minimal dependency list for inference deployments. |

## Setup

Create an environment with Python 3.12, then install PyTorch for your CUDA version and the remaining dependencies:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1
pip install -r requirements_server.txt
```

Download the DAP checkpoint from Hugging Face and place it here:

```text
depth_estimation/models/hf_model/model.pth
```

Official weights: <https://huggingface.co/Insta360-Research/DAP-weights>

The demo script creates a symlink from `DAP/checkpoints/model.pth` to this local weight path. The checkpoint itself is intentionally excluded from git.

## Run the Demo

From `depth_estimation/`:

```bash
./run_demo.sh
```

By default, results are written to:

```text
outputs/demo/
```

Typical outputs include:

- `depth_npy/*.npy`: raw depth arrays.
- `depth_vis_gray_100m/*.png`: grayscale depth visualizations.
- `depth_vis_color_100m/*.png`: color depth visualizations.
- `demo_contact_sheet.png`: side-by-side preview.

## Run on a Custom Image List

Create a text file with one ERP image path per line, then run the DAP inference script:

```bash
cd DAP
../.venv/bin/python test/infer.py \
  --config config/infer_local.yaml \
  --txt /path/to/pano_paths.txt \
  --output /path/to/output_dir \
  --device auto
```

`--device auto` prefers CUDA when available and falls back to MPS or CPU.

## Multi-GPU Batch Inference

For a directory of panorama images:

```bash
./run_server_multi_gpu.sh \
  --input-dir /data/panos \
  --output-dir /data/dap_outputs \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --metric-scale 100
```

For an existing text list:

```bash
./run_server_multi_gpu.sh \
  --input-list /data/pano_paths.txt \
  --output-dir /data/dap_outputs \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --metric-scale 100
```

The launcher starts one worker process per GPU by default. This is usually more reliable for large panorama batches than a single `DataParallel` process.

## JSON Manifest Mode

If your dataset is represented as a JSON array:

```json
[
  {
    "image_path": "/data/images/Sphere360/demo_scene/frame_00.jpg",
    "source": "/data/videos/demo_scene.mp4",
    "scene_id": "demo_scene",
    "viewpoint_id": "frame_00"
  }
]
```

Run:

```bash
.venv/bin/python json_multi_gpu_infer.py \
  --json-path /data/pano_records.json \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --metric-scale 100 \
  --batch-size 1
```

The script reads `image_path`, writes depth maps next to the image tree under an `images_dep` directory, and updates each JSON record with `depth_path` and `depth_scale`.

By default, JSON mode writes a compact 16-bit PNG depth representation. Use `--output-format npy` when you need NumPy arrays:

```bash
.venv/bin/python json_multi_gpu_infer.py \
  --json-path /data/pano_records.json \
  --gpu-ids 0,1,2,3 \
  --output-format npy
```

Use `--backup-json` to keep a copy of the original manifest before it is updated.

## Optional Outputs

Add these flags when debugging or analyzing results:

- `--save-raw`: save raw model depth before metric scaling.
- `--save-mask`: save the model valid-region mask.
- `--save-vis`: save grayscale and color visualizations.
- `--amp`: enable CUDA autocast for faster inference on supported GPUs.

## Official DAP References

- GitHub: <https://github.com/Insta360-Research-Team/DAP>
- Hugging Face weights: <https://huggingface.co/Insta360-Research/DAP-weights>
- Hugging Face demo: <https://huggingface.co/spaces/Insta360-Research/DAP>
