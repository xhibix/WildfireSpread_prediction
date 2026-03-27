# Wildfire Spread Prediction — ML for Climate Change

**Course:** ML for Climate Change, Imperial College London

**Dataset:** [WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) (10% subset) · [Zenodo](https://doi.org/10.5281/zenodo.8006177)

**Presentation:** [`docs/mlClimate_presentation.pdf`](docs/mlClimate_presentation.pdf)

**Peer Assessment Form** (https://docs.google.com/spreadsheets/d/1m32XOlF3WH_khP9DvAulmOF2T6Qka2bCJ9-M0E9MpHM/edit?usp=sharing)

---

## Project Overview

This project investigates deep learning approaches to **next-day wildfire spread prediction** using the WildfireSpreadTS benchmark dataset. Given T=5 consecutive days of multi-modal satellite imagery (23 channels: fire masks, weather, topography, vegetation), models predict the binary active fire mask for the following day.

Six architectures were proposed and evaluated across four independent workstreams, with evaluation via Average Precision (AP) under 3-fold cross-validation on a 10% subset of the dataset.

---

## Team & Contributions

| Contributor | Folder | Model / Workstream |
|-------------|--------|--------------------|
| **Ahmed Zekry** | [`models/resnet-unet/`](models/resnet-unet/) | ResNet18-UNet baseline & ablation study — reproduced and extended the WSTS+ paper (Lahrichi et al., 2026) |
| **Paolo Salvetti** | [`models/vit-unet/`](models/vit-unet/) | ViT-UNet (MiT-B2 encoder + UNet decoder) architecture comparison; benchmarked against UTAE and ConvLSTM |
| **Naimeh Vaezi** | [`models/temporal-fusion-unet/`](models/temporal-fusion-unet/) | Temporal Fusion U-Net with multi-head attention; systematic hyperparameter search |
| **xhibix** | [`models/hh-experiments/`](models/hh-experiments/) | **Static-Conditioned FNO-Transformer** (FiLM conditioning + Fourier Neural Operator encoder + multi-scale temporal attention + probabilistic head) & **VQ-VAE + Latent Diffusion Model** (discrete latent space + DDPM/DDIM generative wildfire spread prediction) |

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
    └── hh-experiments/                # Static-Conditioned FNO-Transformer & VQ-VAE + LDM
        ├── README.md                  # Full architecture & method documentation
        ├── plot_results.py
        ├── Figure_1.png
        └── notebooks/
            ├── wildfire_experiments.ipynb   # Arch A (FNO-Transformer) + Arch B (VQ-VAE + LDM)
            ├── wildfirespread_v3_new.ipynb  # UTAE baseline + UTAE with pretrained CNN
            └── ViT_Unet.ipynb               # Initial ViT-UNet baseline
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

### 4. Static-Conditioned FNO/ CNN-Transformer (Architecture A)
**Location:** [`models/hh-experiments/`](models/hh-experiments/)

A fully custom discriminative model that explicitly separates the 23 input channels into **static** (time-invariant: vegetation, topography, landcover) and **dynamic** (per-day: weather, fire history, VIIRS bands) groups. The dynamic encoder is conditioned on static features at every scale via **FiLM (Feature-wise Linear Modulation)**, replacing the standard unconditional encoder.

Two encoder variants are provided: a **ResNet18-style CNN** with residual blocks and FiLM at each stage, and a **Fourier Neural Operator (FNO)** encoder that operates in the frequency domain via learnable complex-valued spectral kernels — enabling long-range spatial dependencies without stacking convolution layers.

Temporal fusion is handled by the **MultiScaleTemporalTransformer**: a learnable-query cross-attention module that collapses T=5 frames into a single representation applied at **every encoder scale** (not just the bottleneck as in UTAE). The model outputs both a fire probability map and a per-pixel **aleatoric uncertainty estimate** via a heteroscedastic probabilistic head.

| Component | Design choice |
|-----------|--------------|
| Static encoder | Dual-path (continuous + categorical) → fused 64-ch map |
| Dynamic encoder | ResNet18+FiLM or FNO+FiLM; 4 scales [64,128,256,512] |
| Temporal fusion | Learnable-query attention at all 4 encoder scales |
| Output head | Probabilistic: fire probability + aleatoric uncertainty |
| Loss | Focal-heteroscedastic / focal-BCE / soft-Dice |

---

### 5. VQ-VAE + Latent Diffusion Model (Architecture B)
**Location:** [`models/hh-experiments/`](models/hh-experiments/)

A two-stage **generative** approach that reframes wildfire spread prediction as conditional image generation in a learned discrete latent space — the only generative architecture in this project.

**Stage 1 — WildfireVQVAE:** A convolutional VQ-VAE compresses each 40-channel preprocessed frame (128×128) into a discrete 32×32 grid of 256-dim tokens drawn from a 512-entry codebook (cosine-similarity matching, EMA codebook updates). The reconstruction loss up-weights the fire and active-fire channels (×10 and ×5) to preserve the signal most relevant to the prediction task.

**Stage 2 — ConditioningNetwork + LatentDiffusionModel:** The T=5 input frames are encoded and quantized by the frozen VQ-VAE into a sequence of latent grids. A lightweight **ConditioningNetwork** (3-layer Transformer encoder with per-spatial-location attention) summarises the T−1 context frames into a single conditioning latent `z_cond`. The **LatentDiffusionModel** then runs DDPM (1000 steps, linear β schedule) in the 32×32 latent space: a small U-Net denoiser (channels [256,384,512], GroupNorm, sinusoidal time embeddings) takes `[z_noisy ∥ z_cond]` and predicts noise. At inference, **DDIM sampling** (50 steps) generates the predicted next-day latent, which the frozen decoder maps back to fire probability.

Three experiment variants are evaluated under 3-fold CV:

| Variant | Description |
|---------|-------------|
| **B1** | VQ-VAE + ConditioningNetwork only — deterministic latent forecast (MSE in latent space) |
| **B2** | VQ-VAE + ConditioningNetwork + DDPM — stochastic refinement via diffusion |
| **E2** | Architecture A with CNN encoder weights transferred from pretrained VQ-VAE encoder |

See [`models/hh-experiments/README.md`](models/hh-experiments/README.md) for full architecture and training details.

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
- van den Oord et al. (2017). *Neural Discrete Representation Learning*. NeurIPS. (VQ-VAE)
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. (DDPM)
- Song et al. (2021). *Denoising Diffusion Implicit Models*. ICLR. (DDIM)
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. (LDM)
- Li et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR. (FNO)
- Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI.
