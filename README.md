# OCT IOI Detection

Official code release for the paper:

> **Quantitative Analysis of Vitreous Cells and Vitreoretinal Interface Irregularities in Optical Coherence Tomography for Intraocular Inflammation Detection**
>
> Deng Qianyi, Kazumasa Kishimoto, Osamu Sugiyama, Yuhei Iga, Masahiro Akada, Junki Hoshino, Hidetaka Matsumoto, Hideo Akiyama, Hiroshi Tamura, Masayuki Hata.

This repository provides the principal method components and training protocol used in the study. The implementation is based on modified Ultralytics YOLO11 source code.

The study uses a non-public OCT dataset. Consequently, this repository is intended as a module-level reference rather than a standalone, directly executable package. The released files must be integrated into a compatible Ultralytics codebase together with appropriately prepared data.

## Code overview

| File | Description |
| --- | --- |
| `cpm_module.py` | CPM module definitions used by the modified network. |
| `binary.py` | Image binarization code. |
| `train.py` | Training procedure and experiment parameters. |
| `test.py` | Code used with the independent test set. |
| `configs/train.yaml` | YOLO11 model configuration with CPM blocks and P2-P4 detection outputs. |
| `configs/dataset.yaml` | Dataset configuration template for the training and validation splits. |

The module files mirror modifications made within the Ultralytics source tree. They may rely on imports, registrations, and surrounding classes from the corresponding Ultralytics modules.

## Dataset

The OCT dataset used in this study is not publicly available and is not included in this repository.

Each training configuration contains only `train` and `val` splits:

```yaml
path: /path/to/datasets
train: images/train
val: images/val
nc: 2
```

An independent test set is maintained separately from the training and validation data. It is not referenced by the training dataset configurations and is not distributed with this code release.

## Training configuration

The training settings below follow `train.py`:

| Parameter | Value |
| --- | --- |
| Model configuration | `train.yaml` |
| Dataset configurations | `dataset_i.yaml`|
| Number of runs | 10 |
| Epochs | 500 |
| Image size | 1280 |
| Batch size | 32 |
| Patience | 0 |

The dataset configuration files and paths used for the experiments are environment-specific. `configs/dataset.yaml` is provided only as a structural template.

## Citation

If this work is useful in your research, please cite the associated paper. The complete publication record and BibTeX entry will be added when available. Citation metadata is also provided in `CITATION.cff`.

## License

This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE` and `NOTICE` for details.

The AGPL license is retained because this release contains modifications and excerpts derived from the AGPL-licensed Ultralytics YOLO source code.

## Acknowledgements

This work builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Users must also comply with the applicable Ultralytics license and citation requirements.
