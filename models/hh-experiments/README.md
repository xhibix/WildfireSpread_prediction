# HH Experiments — WildfireSpreadTS

**Contributor:** xhibix
**Task:** Next-day wildfire spread prediction (binary segmentation)
**Dataset:** WildfireSpreadTS (10% subset) — 128×128 patches, 23 input channels, T=5 days

---

## Overview

Three progressively refined experiments, moving from a simple ViT channel-stacking
baseline toward a fully custom temporal transformer (Architecture A) and a
generative latent diffusion pipeline (Architecture B).

| Notebook | Experiment |
|----------|------------|
| `ViT_Unet.ipynb` | Baseline — MiT-B2 ViT-UNet, T days flattened into channels |
| `wildfirespread_v3_new.ipynb` | UTAE baseline; UTAE with pretrained CNN encoder |
| `wildfire_experiments.ipynb` | Architecture A (WildfireTransformer) + Architecture B (VQ-VAE + LDM) |

---

## Experiment 1 — ViT-UNet Baseline (`ViT_Unet.ipynb`)

The starting point. A MiT-B2 (SegFormer) encoder paired with a standard UNet decoder,
using `segmentation-models-pytorch`.

**Temporal handling:** T=5 days are stacked along the channel axis
(5 × 23 = 115 channels) before the encoder — no explicit temporal module.

**Limitations addressed in later work:**
- `ModelCheckpoint` monitored `val_f1` instead of `val_loss`
- Single-fold evaluation only (not 3-fold CV)
- No temporal modelling — time axis collapsed into channels

---

## Experiment 2 — UTAE Baselines (`wildfirespread_v3_new.ipynb`)

Introduces proper temporal modelling using the reference UTAE implementation
from the WildfireSpreadTS repo (Garnot & Landrieu, ICCV 2021).

### Setup
- **Data:** `FireSpreadDataModule` from the reference repo; loaded from local SSD
  (avoids Google Drive FUSE I/O bottleneck on Colab)
- **Loss:** BCE with positive class weight ~960–1000 (addresses severe class imbalance)
- **Evaluation:** Average Precision (AP), 3-fold CV

### Variants

**UTAE_baseline** — `UTAELightning` wrapped in a custom `UTAEWrapper` class.
Standard temporal attention U-Net operating on the 5-day sequence.

**UTAE_pretrained_CNN** — `UTAELightning` with a pretrained CNN encoder and
day-of-year (DOY) positional encoding injected into the temporal attention module.

---

## Experiment 3 — Architecture A & B (`wildfire_experiments.ipynb`)

The main notebook. Implements two architectures:

- **Architecture A — WildfireTransformer:** Custom discriminative model with
  static conditioning, FNO encoding, and multi-scale temporal attention
- **Architecture B — VQ-VAE + Latent Diffusion:** Generative approach that
  first learns a discrete latent space, then trains a diffusion model within it

### Shared Configuration

```python
N_DAYS       = 5       # input sequence length T
BATCH_SIZE   = 32
CROP_SIZE    = 128
MAX_EPOCHS   = 20
LR           = 1e-3
NUM_WORKERS  = 8
```

### Cross-Validation Folds

| Fold | data_fold_id | Train years   | Val year | Test year |
|------|-------------|---------------|----------|-----------|
| A    | 0           | 2018, 2019    | 2020     | 2021      |
| B    | 3           | 2018, 2020    | 2021     | 2019      |
| C    | 10          | 2020, 2021    | 2018     | 2019      |

---

## Architecture A — WildfireTransformer

A discriminative encoder-decoder that separates the 23 raw input channels into
**static** (time-invariant) and **dynamic** (per-day) groups, then conditions
the dynamic encoder on the static branch at every scale via FiLM.

### Feature Split

| Group | Size | Channels |
|-------|------|----------|
| **Static** | 22 ch | NDVI, EVI2, topography (slope, aspect, elevation, PDSI), landcover (one-hot) |
| **Dynamic** | 18 ch × T | VIIRS bands (M11, I2, I1), weather observed + forecast, active fire mask |

> After landcover one-hot expansion the full preprocessed tensor has 40 channels.

### Architecture Config

```python
D_STATIC     = 64               # static embedding dimension
ENC_CHANNELS = [64, 128, 256, 512]
DEC_CHANNELS = [256, 128, 64, 32]
N_HEADS      = 8
FNO_MODES    = [32, 16, 8, 4]  # spectral cutoff per encoder stage
```

### Components

#### StaticBranch
Encodes the 22 static channels into a spatial conditioning map `(B, 64, H, W)`.
Two parallel sub-encoders — continuous (NDVI, EVI2, topography) and categorical
(landcover one-hot) — fused by a 1×1 convolution.

#### FiLM — Feature-wise Linear Modulation
Conditions the dynamic encoder on static context at each encoder scale:
```
output = γ(static) · x + β(static)
```
A small MLP maps the static embedding to per-channel scale (γ) and shift (β).
Applied at **every encoder stage**, so static information propagates through
the full feature hierarchy.

#### CNNEncoder
ResNet18-style encoder processing one dynamic timestep `(B, 18, H, W)`.
- Stem: 7×7 conv, stride 2
- 4 residual stages, each followed by a FiLM layer
- Multi-scale outputs: `[64, 128, 256, 512]` channels at H/2, H/4, H/8, H/16

#### SpectralConv2d — Fourier Neural Operator layer
Learnable complex-valued kernels in the frequency domain:
1. Apply 2D FFT to the feature map
2. Multiply low-frequency modes by learned complex weights via `einsum`
3. Inverse FFT back to spatial domain
4. Add a residual 1×1 conv path

`FNO_MODES` controls the spectral cutoff per encoder stage.

#### FNOEncoder
Multi-scale encoder where each stage is a `SpectralConv2d` block.
Static features are concatenated to the input before each stage.
Used as an alternative to `CNNEncoder`.

#### MultiScaleTemporalTransformer
Collapses the T=5 temporal frames into a single fused representation using
**learnable-query cross-attention**, applied independently at each of the four
encoder scales. Each scale has its own learnable positional embeddings for the T
input frames and a single learnable query token that attends over the T embeddings.
Also outputs per-scale attention maps for interpretability.

> Unlike UTAE (which fuses time only at the bottleneck), this module fuses at
> all four scales.

#### UNetDecoder
Standard UNet decoder: progressive 2× upsampling with skip connections from the
encoder, channel concatenation at each level.

#### ProbabilisticHead
Dual-output head predicting:
- **Fire probability** — sigmoid-activated mean
- **Aleatoric uncertainty** — softplus-activated variance (heteroscedastic model)

### Loss Functions

| Name | Description |
|------|-------------|
| `focal_heteroscedastic_loss` | Focal loss weighted by predicted variance |
| `focal_bce_loss` | Standard focal loss for class imbalance |
| `soft_dice_loss` | Differentiable Dice coefficient |

### Optimiser

- AdamW, β=(0.9, 0.999), weight_decay=0.01
- CosineAnnealingLR, η_min = LR × 0.01
- ModelCheckpoint monitors `val_loss`; EarlyStopping enabled

---

## Architecture B — VQ-VAE + Latent Diffusion Model

A two-stage generative pipeline:
1. **VQ-VAE** — learns a discrete compressed latent space from single-frame inputs
2. **Latent Diffusion Model (DDPM)** — learns to denoise in that latent space,
   conditioned on a summary of the T=5 input sequence

The VQ-VAE is pretrained once (self-supervised, reconstruction only) and then
frozen for all diffusion experiments.

### Stage 1 — WildfireVQVAE

Encodes a full 40-channel preprocessed frame `(B, 40, 128, 128)` into a discrete
32×32 latent grid.

**Architecture:**

```
Encoder:
  Conv2d(40 → 64, stride=2)  →  ResBlock(64)
  Conv2d(64 → 128, stride=2) →  ResBlock(128)
  Conv2d(128 → 256, 1×1)
  → z_e: (B, 256, 32, 32)

VectorQuantize:
  codebook_size = 512
  dim = 256
  decay = 0.99
  commitment_weight = 0.25
  cosine similarity matching
  → z_q: (B, 256, 32, 32)

Decoder (symmetric):
  Conv(256 → 128, 1×1) → ResBlock(128)
  ConvTranspose(128 → 64, stride=2) → ResBlock(64)
  ConvTranspose(64 → 40, stride=2)
  → x̂: (B, 40, 128, 128)
```

**Training:**
- Reconstruction loss: weighted MSE; fire channel (index 39) weighted ×10,
  active-fire channel (index 38) weighted ×5
- Total loss: `recon_loss + vq_loss`
- Optimiser: Adam, lr=1e-4
- Pretraining uses fold 0 train split (2018+2019); checkpoint saved and reused

### Stage 2a — ConditioningNetwork

A lightweight temporal transformer that summarises the T latent frames into a
single conditioning latent `z_cond`.

```
Input:   z_seq  (B, T, 256, 32, 32)
         ↓ add learnable positional embeddings per timestep
         ↓ reshape to (B×32×32, T, 256)  — per-spatial-location sequences
         ↓ TransformerEncoder (8 heads, 3 layers, 4× feedforward, dropout=0.1)
         ↓ take last token → reshape to (B, 256, 32, 32)
         ↓ Conv2d(256→256, 3×3) + BN + GELU + Conv2d(256→256, 1×1)
Output:  z_cond (B, 256, 32, 32)
```

### Stage 2b — LatentDiffusionModel (DDPM + DDIM)

A U-Net denoiser operating entirely in the 32×32 latent space.

**Noise schedule:** linear β from 1e-4 to 0.02 over T_DIFF=1000 steps

**LatentUNet denoiser:**

```
Input:  [z_noisy ∥ z_cond]  (B, 512, 32, 32)   ← concatenated on channel axis

Encoder:
  ConvBlock(512 → 256)  →  AvgPool(2)
  ConvBlock(256 → 384)  →  AvgPool(2)
  ConvBlock(384 → 512)

Bottleneck:
  ConvBlock(512 → 512)

Decoder (with skip connections):
  ConvTranspose(512→512) + ConvBlock(512+384 → 384)
  ConvTranspose(384→384) + ConvBlock(384+256 → 256)

Head: Conv2d(256 → 256, 1×1)
Output: ε̂  (B, 256, 32, 32)   ← predicted noise
```

Each `ConvBlock` consists of two (Conv2d → GroupNorm(8) → SiLU) layers plus a
time-conditioning term: `output += Linear(t_emb)[:, :, None, None]`.

**Time embedding:** sinusoidal (dim=128) passed through a two-layer MLP
(Linear → SiLU → Linear).

**Training objective:** MSE between predicted and actual noise (`ε̂` vs `ε`)

**DDIM sampling** (50 steps):
```python
z_T ~ N(0, I)
for t in linspace(T-1, 0, 50):
    ε̂ = UNet([z_t ∥ z_cond], t)
    z0_hat = (z_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t        # clamp to [-3, 3]
    z_{t-1} = √ᾱ_{t-1}·z0_hat + √(1-ᾱ_{t-1})·ε̂
return z_0
```

### Full Inference Pipeline (Architecture B)

```
x_seq (B, T, 40, 128, 128)
  ↓  VQ-VAE encoder + quantize  [frozen]
z_seq (B, T, 256, 32, 32)
  ↓  z_input = z_seq[:, :-1]   # first T-1 frames as context
  ↓  ConditioningNetwork
z_cond (B, 256, 32, 32)
  ↓  DDIM sample (50 steps, conditioned on z_cond)
z_pred (B, 256, 32, 32)
  ↓  VQ-VAE decoder  [frozen]
x_pred (B, 40, 128, 128)
  ↓  sigmoid(x_pred[:, -1])    # fire channel
fire_prob (B, 128, 128)
```

### Experiment Variants

| Name | Description |
|------|-------------|
| **B1 — `cond_only`** | VQ-VAE + ConditioningNetwork only; deterministic latent forecast via MSE in latent space |
| **B2 — `full`** | VQ-VAE + ConditioningNetwork + DDPM; stochastic refinement via diffusion |
| **E2 — `CNN_pretrained`** | Architecture A (CNN encoder) with encoder weights transferred from the pretrained VQ-VAE encoder; partial weight transfer (matching layers only, skipping first conv due to channel mismatch 40 vs 18) |

### Diffusion Training Setup

- **Frozen:** VQ-VAE encoder and decoder throughout all diffusion training
- **Optimised:** ConditioningNetwork (both B1 and B2); LatentDiffusionModel (B2 only)
- **Optimiser:** AdamW, lr=1e-4, weight_decay=0.01
- **Checkpoint:** monitors `val_loss`, saves best model per fold
- **Precision:** fp16 mixed on GPU, fp32 on CPU

---

## References

- Garnot & Landrieu (2021). *Panoptic segmentation of satellite image time series
  with convolutional temporal attention networks.* ICCV. (UTAE)
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. (DDPM)
- Song et al. (2021). *Denoising Diffusion Implicit Models.* ICLR. (DDIM)
- Li et al. (2021). *Fourier Neural Operator for parametric PDEs.* ICLR.
- Perez et al. (2018). *FiLM: Visual reasoning with a general conditioning layer.* AAAI.
- van den Oord et al. (2017). *Neural Discrete Representation Learning.* NeurIPS. (VQ-VAE)
- Rombach et al. (2022). *High-resolution image synthesis with latent diffusion models.* CVPR.
- Xie et al. (2021). *SegFormer.* NeurIPS. (MiT-B2 encoder)
