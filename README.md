# OCT IOI Detection

Official implementation for the paper:

> **Quantitative Analysis of Vitreous Cells and Vitreoretinal Interface Irregularities in Optical Coherence Tomography for Intraocular Inflammation Detection**
>
> Deng Qianyi

This repository will contain the CPM module and YOLO11 training code used in the study. The implementation files currently contain explicit placeholders and will be replaced with the final research code before release.

## Repository structure

```text
OCT-IOI-Detection/
|-- configs/
|   `-- data.yaml          # Example Ultralytics dataset configuration
|-- cpm_module.py          # Placeholder for the CPM module
|-- train.py               # Placeholder for the YOLO11 training entry point
|-- requirements.txt
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- NOTICE
`-- LICENSE
```

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/Tousengi/OCT-IOI-Detection.git
cd OCT-IOI-Detection
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset preparation

Organize the dataset in the standard Ultralytics detection format:

```text
datasets/oct_ioi/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

Update `configs/data.yaml` with the local dataset path and the final class names. The dataset is not distributed in this repository. Users are responsible for obtaining the data and following all applicable ethics, privacy, and licensing requirements.

## Training

After replacing the placeholders with the final CPM module and training implementation, training will follow the standard Ultralytics YOLO11 workflow:

```bash
python train.py \
  --model yolo11n.pt \
  --data configs/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

The exact model variant, hyperparameters, random seeds, and source modifications used for the paper will be documented with the final code release.

## Inference

Inference instructions and pretrained weights will be added when the final implementation is released.

## Citation

If this work is useful in your research, please cite the paper. The BibTeX entry will be added after publication. Citation metadata is also available in `CITATION.cff`.

## License

This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE` and `NOTICE` for details.

The AGPL license is used because the planned release includes modifications derived from the AGPL-licensed Ultralytics YOLO source. If the final release is covered by an Ultralytics Enterprise License or no longer distributes derived source, the licensing arrangement should be reviewed before publication.

## Acknowledgements

This project builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Please also follow its license and citation requirements.
