# Wildfire Spread Prediction — ViT-UNet & Temporal Baselines

**Course:** ML for Climate Change — Imperial College London
**Dataset:** [WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) (10% subset) · [Zenodo](https://doi.org/10.5281/zenodo.8006177)
**Notebook:** [`notebooks/WildfireTS_ViT_UNet.ipynb`](notebooks/WildfireTS_ViT_UNet.ipynb)

---

## Overview

We benchmark three temporal deep learning architectures for **next-day wildfire spread prediction** on a 10% subset of the WildfireSpreadTS dataset. Each model receives T=5 days of multi-modal satellite imagery and predicts the next-day active fire mask as a binary segmentation task.

---

## Dataset

**WildfireSpreadTS** — 59 fire events, daily satellite composites across the US (2018–2021).

| Split | Years | Fires | Timesteps |
|-------|-------|------:|----------:|
| Train | 2018, 2019 | 24 | 525 |
| Val | 2020 | 20 | 381 |
| Test | 2021 | 15 | 307 |

3-fold cross-validation rotates the year assignment across folds 0–2.

**Class imbalance:** fire pixels represent ~0.1% of all pixels (pos_class_weight ≈ 954), addressed via weighted BCE + Dice loss.

### Input Features (23 channels × T=5 days = 115 channels total)

| Group | Features |
|-------|----------|
| Spectral | VIIRS M11, I2, I1 |
| Vegetation | NDVI, EVI2 |
| Weather | Total precipitation, Wind speed, Min/Max temperature, Specific humidity, Energy Release Component |
| Topography | Elevation, PDSI, Slope, Aspect |
| Landcover | One-hot encoded classes (8 channels) |
| Fire history | Active fire mask (previous T days) |

---

## Architectures

Three architectures are compared via ablation on fold 0 before running 3-fold cross-validation on the best.

| # | Model | Temporal Mechanism | Params |
|---|-------|--------------------|--------|
| 1 | **ViT-UNet** | MiT-B2 encoder + UNet decoder · flatten T timesteps into channels | ~25 M |
| 2 | **ConvLSTM** | Single ConvLSTM cell propagates hidden state across T days | ~1 M |
| 3 | **UTAE** | U-Net with temporal attention at the bottleneck (Gerard et al. 2023) | ~1.1 M |

**ViT-UNet** uses a MixVisionTransformer (MiT-B2) encoder from the SegFormer family — a hierarchical Vision Transformer that produces multi-scale feature maps, making it naturally compatible with a UNet decoder (`segmentation-models-pytorch`).

---

## Results

### Ablation — Architecture Comparison (Fold 0, 20 epochs)

| Model | Test AP | Test F1 | Test IoU | Precision | Recall |
|-------|--------:|--------:|---------:|----------:|-------:|
| ViT-UNet | 0.2672 | 0.0258 | 0.0131 | 0.0131 | 0.9213 |
| ConvLSTM | 0.3024 | 0.0912 | 0.0478 | 0.0484 | 0.8003 |
| **UTAE** | **0.3402** | 0.0849 | 0.0443 | 0.0445 | **0.9341** |

**Best architecture: UTAE** (Test AP = 0.3402)

### 3-Fold Cross-Validation — UTAE

| Fold | Test AP | Test F1 | Test IoU | Precision | Recall |
|------|--------:|--------:|---------:|----------:|-------:|
| 0 | 0.3305 | 0.0858 | 0.0448 | 0.0451 | 0.8909 |
| 1 | 0.1365 | 0.2599 | 0.1493 | 0.1817 | 0.4558 |
| 2 | **0.3987** | **0.2585** | **0.1484** | 0.1529 | 0.8344 |
| **Mean** | 0.2886 | 0.2014 | 0.1142 | 0.1266 | 0.7270 |
| **Std** | 0.1360 | 0.1001 | 0.0601 | 0.0720 | 0.2366 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Input days (T) | 5 |
| Features per timestep | 40 (after duplicate removal) |
| Epochs | 20 |
| Batch size | 128 |
| Crop size | 128 × 128 |
| Loss | BCE + Dice |
| Optimiser | Adam |
| Seed | 42 |
| Hardware | NVIDIA A100-SXM4-40GB |

---

## Dependencies

```bash
pip install segmentation-models-pytorch pytorch-lightning==2.0.1 \
    torchmetrics einops jsonargparse[signatures]==4.20.1 wandb h5py
```

---

## Citation

```bibtex
@inproceedings{
    gerard2023wildfirespreadts,
    title={WildfireSpread{TS}: A dataset of multi-modal time series for wildfire spread prediction},
    author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=RgdGkPRQ03}
}
```
