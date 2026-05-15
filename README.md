<div align="center">

# PanoWorld: Towards Spatial Supersensing in 360° Panorama World

**Pano-native multimodal learning for full-surround 360° spatial reasoning, holistic sensing, and panoramic navigation.**

[![Project Page](https://img.shields.io/badge/Project-Page-2d776f?style=for-the-badge&logo=githubpages&logoColor=white)](https://wcpcp.github.io/PanoWorld/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.13169-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2605.13169)
[![HF Paper](https://img.shields.io/badge/HuggingFace-Paper-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=111)](https://huggingface.co/papers/2605.13169)
[![Dataset](https://img.shields.io/badge/Dataset-Coming%20Soon-8c8c8c?style=for-the-badge)](#release-todo)
[![Model](https://img.shields.io/badge/Model-Coming%20Soon-8c8c8c?style=for-the-badge)](#release-todo)

</div>

<p align="center">
  <img src="./docs/assets/teaser.png" alt="PanoWorld teaser" width="92%">
</p>

## What Is PanoWorld?

Existing MLLMs often reason over fragmented perspective crops, making it difficult to associate spatial cues across the full 360° field of view. **PanoWorld** introduces **pano-native supersensing**, where VLMs perceive and reason directly over complete equirectangular panorama (ERP) observations as continuous observer-centered worlds.

This enables a unified full-surround representation for downstream tasks such as human-centric visual search, omnidirectional 3D spatial reasoning, and panoramic navigation.

## Highlights

| Icon | Component | Description |
| --- | --- | --- |
| 🌐 | **Pano-native supersensing** | Learns from complete 360° ERP panoramas instead of stitching together narrow perspective views. |
| 🧭 | **Observer-centered spatial reasoning** | Models directions, reference-frame changes, seam continuity, and panoramic topology. |
| 🧠 | **PanoSpace-Bench** | Diagnostic benchmark for ERP-native spatial localization, 3D relations, BFOV grounding, and reorientation. |
| 🏗️ | **PanoWorld architecture** | Injects spherical geometry into the visual stream through Spherical Spatial Cross-Attention. |
| 🚶 | **Embodied transfer** | Transfers panoramic understanding to navigation settings such as R2R-CE Val-Unseen. |

## Release TODO

We are preparing the public release. Code, model weights, and data will be uploaded in stages.

- [x] 🌍 Project page
- [x] 📄 arXiv paper link
- [x] 🤗 Hugging Face paper page
- [ ] 📦 Training and evaluation code
- [ ] 🧠 PanoWorld model checkpoints
- [ ] 🗂️ PanoWorld training data
- [ ] 🧪 PanoSpace-Bench benchmark files
- [ ] 📊 Evaluation scripts and result reproduction guide
- [ ] 🧭 Navigation transfer setup

## Repository Status

| Resource | Status | Link |
| --- | --- | --- |
| Project page | ✅ Available | [wcpcp.github.io/PanoWorld](https://wcpcp.github.io/PanoWorld/) |
| Paper | ✅ Available | [arXiv PDF](https://arxiv.org/pdf/2605.13169) |
| Hugging Face paper page | ✅ Available | [HF Papers](https://huggingface.co/papers/2605.13169) |
| Code | 🚧 Coming soon | To be released |
| Dataset | 🚧 Coming soon | To be released |
| Model | 🚧 Coming soon | To be released |

## Main Tasks

| Task | Goal |
| --- | --- |
| 🌐 **PanoSpace-Bench spatial reasoning** | Evaluate whether models understand spherical localization, ERP geometry, object reorientation, and 3D relations in a panorama-native setting. |
| 🔎 **H\*Bench holistic sensing** | Test whether full-surround panoramic observations support human-centric object and position sensing. |
| 🚶 **Panoramic navigation transfer** | Use pano-native visual representations to improve embodied navigation in unseen environments. |

## News

- **2026-05**: Project page, arXiv PDF, and Hugging Face paper page are available.
- **Coming soon**: Code, checkpoints, dataset, and benchmark release.

## Citation

If you find PanoWorld useful for your research, please cite:

```bibtex
@article{panoworld2026,
  title   = {PanoWorld: Towards Spatial Supersensing in 360° Panorama World},
  author  = {Wang, Changpeng and Lin, Xin and Liu, Junhan and Liu, Yuheng and Wang, Zhen and Qi, Donglian and Yan, Yunfeng and Chen, Xi},
  journal = {arXiv preprint arXiv:2605.13169},
  year    = {2026}
}
```

## Contact

For questions about the project, please open an issue or contact the authors listed in the paper.
