# Wildfire Spread Prediction — ML for Climate Change

**Course:** ML for Climate Change, Imperial College London
**Dataset:** [WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) (10% subset) · [Zenodo](https://doi.org/10.5281/zenodo.8006177)
**Presentation:** [`docs/mlClimate_presentation.pdf`](docs/mlClimate_presentation.pdf)

---

## Project Overview

This project investigates deep learning approaches to **next-day wildfire spread prediction** using the WildfireSpreadTS benchmark dataset. Given T=5 consecutive days of multi-modal satellite imagery (23 channels: fire masks, weather, topography, vegetation), models predict the binary active fire mask for the following day.

Four architectures were explored across three independent workstreams, with evaluation via Average Precision (AP) under 3-fold cross-validation on a 10% subset of the dataset.

---

## Team & Contributions

| Contributor | Branch | Model / Workstream |
|-------------|--------|--------------------|
| **Ahmed Zekry** | `WSTS-(ResNet18+UNet)` | ResNet18-UNet baseline & ablation study — reproduced and extended the WSTS+ paper (Lahrichi et al., 2026) |
| **Paolo Salvetti** | `model/vit-unet` | ViT-UNet (MiT-B2 encoder + UNet decoder) architecture comparison; benchmarked against UTAE and ConvLSTM |
| **Naimeh Vaezi** | `model/temporal-fusion-unet` | Temporal Fusion U-Net with multi-head attention; systematic hyperparameter search |
| **xhibix** | `HH` | Custom experiment notebooks; exploratory ViT-UNet and feature engineering work |

---

## Repository Structure

```
.
├── README.md                          # This file
├── docs/
│   └── mlClimate_presentation.pdf    # Project presentation slides
│
└── models/
    ├── resnet-unet/                   # ResNet18-UNet — full training codebase
    │   ├── README.md                  # Model details & full benchmark results table
    │   ├── RUNME.md                   # Step-by-step reproduction instructions
    │   ├── requirements.txt
    │   ├── load_trained_model.py      # Utility to load pretrained weights
    │   ├── notebooks/
    │   │   └── WildfireTS_Vit_Unet.ipynb
    │   ├── src/                       # Core training library
    │   │   ├── train.py
    │   │   ├── dataloader/            # FireSpreadDataset, DataModule
    │   │   ├── preprocess/            # HDF5 dataset creation
    │   │   └── models/                # ResUNet, SwinUNet, SegFormer, UTAE, ConvLSTM, ...
    │   ├── cfgs/                      # Hydra YAML configs per model & experiment
    │   └── experiments/               # Standalone ablation scripts + saved results
    │       ├── train_baseline.py
    │       ├── train_improved.py
    │       ├── train_ablation_*.py
    │       └── results/               # JSON results + figures
    │
    ├── vit-unet/                      # ViT-UNet — MiT-B2 encoder + UNet decoder
    │   ├── README.md                  # Architecture details & 3-fold CV results
    │   └── notebooks/
    │       └── WildfireTS_ViT_UNet.ipynb   # Training, ablation, evaluation
    │
    ├── temporal-fusion-unet/          # Temporal Fusion U-Net
    │   ├── README.md                  # Architecture & hyperparameter search results
    │   └── notebooks/
    │       ├── temporal_fusion_unet.ipynb  # Model dev + initial 3-fold CV
    │       ├── tfunet_final.ipynb          # 12-experiment hyperparameter sweep (fold 0)
    │       └── tfunet_3fold_all.ipynb      # 3-fold CV for top-4 configs + plots
    │
    └── hh-experiments/                # Exploratory experiments
        ├── plot_results.py
        ├── Figure_1.png
        └── notebooks/
            ├── wildfire_experiments.ipynb
            ├── wildfirespread_v3_new.ipynb
            └── ViT_Unet.ipynb
```

---

## Model Architectures

### 1. ResNet18-UNet (Baseline & Extension)
**Location:** [`models/resnet-unet/`](models/resnet-unet/)

The standard encoder-decoder segmentation architecture using a ResNet-18 backbone pre-trained on ImageNet. Temporal inputs are handled by either stacking T days along the channel axis (data-level fusion) or using a temporal attention module (feature-level fusion).

This workstream reproduces and extends the WSTS+ paper (Lahrichi et al., 2026), testing multiple backbones (ResNet-18, ResNet-50, SwinUNet, SegFormer) and temporal fusion strategies (UTAE, HTAE, ConvLSTM).

**Ablation study** isolates three proposed improvements over the Gerard et al. (2023) baseline:
- **FireDiff**: adding a fire-spread differential channel
- **Wind**: adding explicit wind direction channels
- **Boundary**: adding a fire perimeter boundary channel

See [`models/resnet-unet/RUNME.md`](models/resnet-unet/RUNME.md) for full reproduction instructions.

---

### 2. ViT-UNet (MiT-B2)
**Location:** [`models/vit-unet/`](models/vit-unet/)

Replaces the ResNet encoder with a **MixVisionTransformer (MiT-B2)** from the SegFormer family — a hierarchical Vision Transformer that produces multi-scale feature maps (H/4, H/8, H/16, H/32). These are fed into a standard UNet decoder via skip connections.

Temporal inputs (T=5 days × 23 channels = 115 channels) are flattened into the channel dimension before the encoder. Three architectures were compared via ablation on fold 0 before running 3-fold CV on the best:

| Model | Temporal Mechanism | Params |
|-------|--------------------|--------|
| **ViT-UNet** | MiT-B2 encoder, flatten T into channels | ~25 M |
| **ConvLSTM** | Single ConvLSTM cell across T days | ~1 M |
| **UTAE** | U-Net with temporal attention at bottleneck | ~1.1 M |

---

### 3. Temporal Fusion U-Net (TF-UNet)
**Location:** [`models/temporal-fusion-unet/`](models/temporal-fusion-unet/)

A custom U-Net where each input day is processed independently by a **shared encoder**, then temporal features are fused via **multi-head attention** before decoding. Two attention modes were tested:

- **All levels**: temporal attention at every encoder level + bottleneck (33.8M params)
- **Bottleneck only**: temporal attention only at the bottleneck, averaging elsewhere (33.1M params)

A systematic 12-experiment hyperparameter search (experiments A–L) explored attention mode, loss functions, learning rate, scheduler, model size, dropout, and input window size.

---

### 4. HH Exploratory Experiments
**Location:** [`models/hh-experiments/`](models/hh-experiments/)

Exploratory notebooks covering data loading, feature engineering, and ViT-UNet experiments on the WildfireSpreadTS dataset.

---

## Results Summary

All results use **Average Precision (AP)** as the primary metric (area under precision-recall curve), evaluated on the 10% subset of WildfireSpreadTS with 3-fold cross-validation.

### Architecture Comparison — Ablation on Fold 0

| Model | Test AP | Notes |
|-------|--------:|-------|
| Persistence baseline | ~0.19 | Prior day fire mask |
| ResNet18-UNet (Gerard 2023) | 0.328 | Original benchmark |
| **ResNet18-UNet (improved)** | **0.455** | +FireDiff, +Wind, +Boundary improvements |
| ViT-UNet (MiT-B2) | 0.267 | MiT-B2 encoder, channels-stacked temporal |
| ConvLSTM | 0.302 | Temporal recurrence |
| UTAE | 0.340 | Temporal attention at bottleneck |
| TF-UNet (bottleneck + dropout) | 0.352 | Best stable TF-UNet config |
| TF-UNet (mono-temporal) | 0.466 | Unstable; highest single AP |

### 3-Fold Cross-Validation Results

**ViT-UNet / UTAE (Paolo)**

| Model | Fold 0 | Fold 1 | Fold 2 | Mean ± Std |
|-------|-------:|-------:|-------:|-----------|
| UTAE | 0.3305 | 0.1365 | 0.3987 | 0.289 ± 0.136 |

**Temporal Fusion U-Net (Naimeh)**

| Config | Fold 0 | Fold 3 | Fold 10 | Mean ± Std |
|--------|-------:|-------:|--------:|-----------|
| All levels, Dice | 0.315 | 0.064 | 0.085 | 0.155 ± 0.114 |

**ResNet18-UNet (Ahmed) — full benchmark across feature sets**

| Model | T | Veg AP | Multi AP | All AP |
|-------|---|-------:|---------:|-------:|
| ResNet18-UNet (improved) | 1 | 0.455±0.090 | 0.468±0.087 | 0.460±0.084 |
| ResNet18-UNet (improved) | 5 | 0.472±0.083 | 0.469±0.087 | 0.460±0.084 |
| UTAE (Res18) | 5 | **0.478±0.085** | **0.477±0.089** | **0.475±0.091** |

---

## Dataset

**WildfireSpreadTS** — 59 wildfire events, daily VIIRS satellite composites across the continental US (2018–2021).

| Split | Years | Fires | Timesteps |
|-------|-------|------:|----------:|
| Train | 2018, 2019 | 24 | 525 |
| Val | 2020 | 20 | 381 |
| Test | 2021 | 15 | 307 |

3-fold cross-validation rotates year assignments across folds 0–2. Each HDF5 file stores one fire event:
- **`x`**: `(T, 23, 128, 128)` — 23 geophysical channels over T days
- **`y`**: `(128, 128)` — binary next-day fire mask

The 23 channels span fire history, weather (observed + forecast), topography, and vegetation. After preprocessing (land-cover one-hot, normalisation), each timestep expands to **40 channels** (41 with fire-diff).

---

## References

- Gerard et al. (2023). *WildfireSpreadTS: A dataset for multi-modal wildfire spread prediction*. NeurIPS.
- Lahrichi, Bova, Johnson, Malof (2026). *Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark*. WACV 2026. [arXiv:2502.12003](https://arxiv.org/abs/2502.12003)
- Xie et al. (2021). *SegFormer: Simple and efficient design for semantic segmentation with Transformers*. NeurIPS.
- Garnot & Landrieu (2021). *Panoptic segmentation of satellite image time series with convolutional temporal attention networks*. ICCV. (UTAE)
