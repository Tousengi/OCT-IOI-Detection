# OCT IOI Detection

Code release for:

> **Quantitative Analysis of Vitreous Cells and Vitreoretinal Interface Irregularities in Optical Coherence Tomography for Brolucizumab-related Intraocular Inflammation Detection in Neovascular Age-related Macular Degeneration**
>
> Qianyi Deng, Yuhei Iga, Masahiro Akada, Junki Hoshino, Hidetaka Matsumoto, Hideo Akiyama, Kazumasa Kishimoto, Osamu Sugiyama, Akitaka Tsujikawa, Hiroshi Tamura, Masayuki Hata

The study detects two intraocular inflammation (IOI) related features on OCT
B-scans — vitreous cells (VC) and vitreoretinal interface irregularities
(VRII) — counts them per examination, and compares the counts between the
baseline and the post-injection visit.

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

## Corrections relative to the earlier release

Two defects in the first public release were found while preparing the
revision, and all results reported in the paper were produced after they were
corrected.

1. **The CPM output did not reach the detection neck.** In the earlier
   configuration the neck took its skip connections from the layers *before*
   the CPM (`[[-1, 5]]` and `[[-1, 2]]`), so the purified features flowed only
   into the following downsampling convolution. The high-resolution detection
   branch therefore never received them. The connections are now taken after
   the CPM (`[[-1, 6]]` and `[[-1, 3]]`). The three configurations here all use the
   corrected wiring.
2. **`cpm_module.py` could not be imported.** The forward method was written
   as `def forward(self.x)`. Any attempt to build the model raised a
   `SyntaxError`.

A third change is a design decision rather than a defect. A channel-wise
softmax normalises to a sum of one, so each gate is of the order of the
reciprocal of the channel count; at the 64 and 128 channels used here the
softmax-gated module changes its input by only about 0.2% to 0.4% in relative
L2 norm. The proposed model therefore uses sigmoid gating, for which the
measured change is approximately 20%. Both forms are provided.

## Reproducing the experiments

### Data

The OCT data are not public. Images came from two hospitals and cannot be
redistributed under the approved protocols. `configs/dataset.yaml` shows the
expected layout; `nc` is 2, with class 0 = vitreous cells and class 1 =
vitreoretinal interface irregularities.

Development set: 304 images from 11 patients with IOI (Kyoto University
Hospital). 129 images contain at least one annotated feature and 175 contain
none; the latter are kept as negative background examples. 483 features were
annotated in total (298 VC, 185 VRII).

External cohort: 30 patients (Gunma University Hospital), 20 with IOI and 10
without. Two orthogonal B-scans, one horizontal and one vertical, were
acquired per visit, so the same cell is not counted twice.

### Cross-validation

`train.py` runs the 10-fold loop. Each of the 10 patients contributing 30
images defines one validation fold; the remaining patient (4 images) is in the
training set of every fold. Each fold therefore has 274 training and 30
validation images. Cross-validation is used only to characterise detection
performance and to compare model variants.

### External validation

A single model is trained once on all 11 development patients with the same
settings and applied once to the external cohort. The checkpoint from the
final epoch is used, not a checkpoint selected on a validation fold, so no
selection is made on validation data. Selecting on the validation fold instead
changes mAP50-95 by 0.10 to 0.13, which is an order of magnitude larger than
the differences between the models compared in the paper.

### Settings

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

### Decision rule

Per patient and per visit the detector produces a vitreous cell score (VCS)
and a vitreoretinal interface irregularity score (VRIIS), each the number of
detections. A patient is classified as at risk of IOI when either score is
higher after injection than at baseline. The rule was fixed before the
experiments. The sensitivity analysis over nine threshold combinations, the
per-fold and class-wise results, the patient-level statistics and the
case-level predictions are reported in Multimedia Appendix 2 of the paper.

## Source files

The source files contain code only. The architecture, the corrections, the
training and inference settings and the decision rule are documented in this
README and in the Multimedia Appendices of the paper.

## Licence

AGPL-3.0-only, following the Ultralytics source this work modifies. See
`LICENSE` and `NOTICE`.
