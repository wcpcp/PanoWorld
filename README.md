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

| Component | Description |
| --- | --- |
| 🌐 **Pano-native supersensing** | Learns from complete 360° ERP panoramas instead of stitching together narrow perspective views. |
| 🧠 **PanoSpace-Bench** | Diagnostic benchmark for ERP-native spatial localization, 3D relations, BFOV grounding, and reorientation. |
| 🏗️ **PanoWorld** | Injects spherical geometry into the visual stream through Spherical Spatial Cross-Attention. |
| 🚶 **Embodied transfer** | Transfers panoramic understanding to navigation settings such as R2R-CE Val-Unseen. |

## Release TODO

We are preparing the public release. Code, checkpoints, datasets, and benchmark files will be uploaded in stages.

- [x] 📄 Paper
- [ ] 📦 Code
- [ ] 🗂️ Dataset
- [ ] 🧪 Benchmark
- [ ] 🧠 Checkpoints

## Visual Examples

<p align="center">
  <img src="./docs/assets/demo_3d_relation.png" alt="PanoSpace-Bench 3D relation reasoning" width="30%">
  <img src="./docs/assets/demo_camera_rotation.png" alt="PanoSpace-Bench camera rotation reasoning" width="30%">
  <img src="./docs/assets/demo_object_reorientation.png" alt="PanoSpace-Bench object reorientation reasoning" width="30%">
</p>

<p align="center">
  <em>PanoSpace-Bench examples cover 3D relation reasoning, reference-frame transformation, and object reorientation in full 360° ERP panoramas.</em>
</p>

<p align="center">
  <img src="./docs/assets/demo_hstar_comparison.png" alt="H*Bench holistic sensing comparison" width="82%">
</p>

<p align="center">
  <em>H*Bench examples show how pano-native reasoning avoids fragmented perspective-view search and supports holistic object and position sensing.</em>
</p>

<p align="center">
  <a href="./docs/assets/demo_vln1.mp4">
    <img src="./docs/assets/demo_vln1_preview.png" alt="Navigation transfer demo preview" width="82%">
  </a>
</p>

<p align="center">
  <em>Navigation transfer example: full-surround ERP observations expose global layout cues and reduce blind spots compared with narrow RGB perspective-view navigation. Click the preview to open the MP4 demo.</em>
</p>

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
