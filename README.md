# Where Should Excitations Meet? Deep Fusion Strategies for Multi-Excitation Hyperspectral Imaging of Ablated Atrial Tissue

Code for the IEEE WHISPERS 2026 paper by A. Hunanyan, N. Ter-Petrosyan, F. Villarruel, N. Sarvazyan, A. Butavyan and V. Yeghiazaryan.

We compare where excitation-specific information should be fused (**feature level** or **logit level**) and how (**AVG, SUM, FC, Gate, CNN, WAVG, ATT**), using MLP and LeViT backbones on multi-excitation hyperspectral images of ablated bovine atrial tissue (4 excitations, 31 emission bands). The baseline is **soft majority voting (Soft MV)**: the logits of four independently trained single-excitation models are averaged at inference.

## Structure

| File | Purpose |
|---|---|
| `main.py` | Train and evaluate one model on one image and split |
| `models.py` | Backbones, fusion operators, training and inference |
| `levit.py` | LeViT backbone adapted to 15 × 15 patches |
| `datasets.py` | Data loading and patch dataset |
| `utils.py` | Metrics (OA, AA, κ) and helpers |
| `soft_majority_voting.py` | Soft MV baseline |
| `run_fusion.sh` | All fusion models |
| `run_single_excitation.sh` | Single-excitation models, then Soft MV |

## Installation

```bash
git clone https://github.com/ArpiHunanyan/me-hsi-excitation-fusion.git
cd me-hsi-excitation-fusion
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt
```

## Data

The dataset is not included. Expected layout:

```
<data_root>/4D/<id>_norm_<H>x<W>x31x4_4d.mat       multi-excitation (H, W, 31, 4)
<data_root>/Separate/<id>_norm_<H>x<W>x31_<e>.mat  single excitation (H, W, 31), e = 2..5
<gt_root>/gt_circle/   training clicks, one file per split
<gt_root>/gt_test/     test labels
<gt_root>/gt/          tissue mask
```

Labels: 0 background, 1 ablated, 2 unablated, 3 unsure (not used for training or testing).

## Usage

One run:

```bash
python main.py --model levit_logit_wavg \
    --dataset 241127_1_norm_696x520x31x4_4d \
    --gt_train 241127_1_696x520_gt_circle_1_percentage_rcount5_split0 \
    --data_root ./data --gt_root ./data/gt --cuda 0
```

Model names are `{levit|mlp}_{feat|logit}_{avg|sum|fc|gate|cnn|wavg|att}`, plus `levit_single` and `mlp_single` for Soft MV. Results (`logits.npy`, `metrics.txt`, `prediction.png`) are saved in `results/<dataset>/<model>/<gt_train>/`.

All experiments:

```bash
DATA_ROOT=./data GT_ROOT=./data/gt CUDA=0 ./run_fusion.sh
DATA_ROOT=./data GT_ROOT=./data/gt CUDA=0 ./run_single_excitation.sh
```

Training: Adam (lr 1e-4), 100 epochs, batch size 100, cross-entropy loss.

## Citation

```bibtex
@inproceedings{hunanyan2026excitations,
  title     = {Where Should Excitations Meet? Deep Fusion Strategies for Multi-Excitation Hyperspectral Imaging of Ablated Atrial Tissue},
  author    = {Hunanyan, Arpi and Ter-Petrosyan, Nazeli and Villarruel, Fernando and Sarvazyan, Narine and Butavyan, Aram and Yeghiazaryan, Varduhi},
  booktitle = {IEEE Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS)},
  year      = {2026}
}
```

## Acknowledgments

Funded by the Afeyan Family Foundation, RA HESC (24PostDoc/2-2G002) and the EU (ERA-CHAIR NAS-SAR project). Parts of the code are adapted from [DeepHyperX](https://github.com/nshaud/DeepHyperX) (GPLv3) and [vit-pytorch](https://github.com/lucidrains/vit-pytorch) (MIT).

