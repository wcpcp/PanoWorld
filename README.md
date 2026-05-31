# PanoWorld: Towards Spatial Supersensing in 360 Panorama World

**Pano-native multimodal learning for full-surround 360-degree spatial reasoning, holistic sensing, and panoramic navigation.**

[![Project Page](https://img.shields.io/badge/Project-Page-2d776f?style=for-the-badge&logo=githubpages&logoColor=white)](https://wcpcp.github.io/PanoWorld/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.13169-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2605.13169)
[![HF Paper](https://img.shields.io/badge/HuggingFace-Paper-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111)](https://huggingface.co/papers/2605.13169)

## Overview

PanoWorld studies **pano-native supersensing**: VLMs perceive and reason directly over complete equirectangular panorama (ERP) observations instead of relying only on fragmented perspective crops.

This repository contains the public code release for the main engineering components used in the project:

| Directory | Purpose |
| --- | --- |
| `base_data_generation/` | ERP metadata, object grounding, spatial relation, and SFT/benchmark data generation pipeline. |
| `depth_estimation/` | DAP-based panoramic depth inference utilities, including single-machine and multi-GPU batch entry points. |
| `train_copy/` | PanoWorld base-model training code with ERP spherical geometry adapters for Qwen3.5-VL. |
| `train_copy_hstar/` | H* / Thinking-in-360 training and evaluation variant built on the same ERP adapter stack. |

Large model weights, generated outputs, caches, local virtual environments, and experiment logs are intentionally not committed. See each subdirectory README for the expected local paths and download instructions.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/wcpcp/PanoWorld.git
cd PanoWorld
```

Read the module-level README that matches your task:

- Dataset and metadata generation: [`base_data_generation/README.md`](base_data_generation/README.md)
- Panoramic depth inference: [`depth_estimation/README.md`](depth_estimation/README.md)
- ERP base-model training: [`train_copy/README.md`](train_copy/README.md)
- H* / Thinking-in-360 training: [`train_copy_hstar/README.md`](train_copy_hstar/README.md)

## Repository Status

- Code: released in this repository.
- Project page and paper: available through the badges above.
- Datasets, checkpoints, and benchmark files: released separately or prepared for staged release.

## Citation

If you find PanoWorld useful for your research, please cite:

```bibtex
@article{panoworld2026,
  title   = {PanoWorld: Towards Spatial Supersensing in 360 Panorama World},
  author  = {Wang, Changpeng and Lin, Xin and Liu, Junhan and Liu, Yuheng and Wang, Zhen and Qi, Donglian and Yan, Yunfeng and Chen, Xi},
  journal = {arXiv preprint arXiv:2605.13169},
  year    = {2026}
}
```

## Contact

For questions, please open a GitHub issue or contact the authors listed in the paper.
