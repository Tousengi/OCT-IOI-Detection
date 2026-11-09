# OCT IOI Detection

Code release for:

> **Quantitative Analysis of Vitreous Cells and Vitreoretinal Interface Irregularities in Optical Coherence Tomography for Brolucizumab-related Intraocular Inflammation Detection in Neovascular Age-related Macular Degeneration**
>
> Qianyi Deng, Yuhei Iga, Masahiro Akada, Junki Hoshino, Hidetaka Matsumoto, Hideo Akiyama, Kazumasa Kishimoto, Osamu Sugiyama, Akitaka Tsujikawa, Hiroshi Tamura, Masayuki Hata

This repository contains the model and training code only. The implementation
modifies the Ultralytics YOLO11 source tree, so the files here are the
modified components and the configurations used; they are not a standalone
package and must be placed in a compatible Ultralytics checkout.

No data, trained weights or per-patient results are distributed here. All
reported results, including the per-fold, class-wise, patient-level and
case-level tables, are in the Multimedia Appendices of the paper.

## Contents

| Path | Description |
| --- | --- |
| `cpm_module.py` | Channel purification module, adapted for detection from ADNet (Xu and Wang, *J Mar Sci Eng* 2024;12(6):852; see `NOTICE`). `CPM` is the softmax-gated form; `CPMSigmoid` is the sigmoid-gated form used by the proposed model. |
| `configs/yolo11_proposed.yaml` | Proposed model: stride-4 (P2) detection branch added, deepest (P5) stage removed, CPM after the P2 and P3 stages. |
| `configs/yolo11_no_cpm.yaml` | Same architecture with the CPM replaced by `nn.Identity`. Layer indices are identical, so the CPM is the only variable. |
| `configs/yolo11_reference.yaml` | Unmodified YOLO11n reference model. |
| `configs/dataset.yaml` | Dataset configuration template. |
| `train.py` | Patient-level 10-fold cross-validation loop. |
| `test.py`, `binary.py` | Inference on the external cohort and the baseline-versus-post comparison. |
| `loss.py`, `metrics.py` | Modified Ultralytics components. |

## Settings

| Item | Setting |
| --- | --- |
| Framework | Ultralytics 8.3.200 |
| Python / PyTorch / CUDA | 3.11.14 / 2.9.0 / 12.8 |
| Hardware | 3 x NVIDIA RTX A6000 (48 GB each) |
| Epochs | 500 |
| Input size | 1280 x 1280 |
| Batch size | 24 (8 per device) |
| Early stopping | Disabled (patience 0) |
| Mixed precision | Disabled |
| Optimizer | `auto`, resolves to AdamW |
| Learning rate / momentum | 0.001667 / 0.9 |
| Initialisation | Random, trained from scratch |
| Random seed | 42 |
| Deterministic operations | Enabled |
| Box loss | Complete IoU (framework default) |
| Augmentation | Framework defaults, applied after the patient-level split |

Inference uses a confidence threshold of 0.25 and a non-maximum-suppression
IoU threshold of 0.7 for counting, and a confidence threshold of 0.001 for
mAP.

## Licence

AGPL-3.0-only, following the Ultralytics source this work modifies. See
`LICENSE` and `NOTICE`.
