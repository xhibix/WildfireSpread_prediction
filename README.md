# WildfireSpread Prediction — MSc Group Project

**Course:** ML for Climate Change — Imperial College London
**Dataset:** [WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) (10% subset) · [Zenodo](https://doi.org/10.5281/zenodo.8006177)
**Reference paper:** [Gerard et al., NeurIPS 2023](https://openreview.net/forum?id=RgdGkPRQ03)

We benchmark deep learning architectures for **next-day wildfire spread prediction** using multi-temporal satellite imagery, reproducing and extending the WildfireSpreadTS benchmark on a 10% data subset.

---

## Models

Each model lives on its own branch under `notebooks/`. Branches follow the naming convention `model/<name>`.

| Branch | Model | Temporal Handling | Author |
|--------|-------|-------------------|--------|
| [`model/vit-unet`](../../tree/model/vit-unet) | ViT-UNet (MiT-B2 + UNet decoder) | Flatten T days → channels / ConvLSTM / UTAE | Paolo Salvetti |

> Add your row here when you push your branch.

---

## ViT-UNet — Architecture & Results

### Architecture

Three architectures are benchmarked and compared via ablation before selecting the best for 3-fold cross-validation:

| # | Name | Temporal Mechanism | Params |
|---|------|--------------------|--------|
| 1 | **ViT-UNet** | MiT-B2 encoder + UNet decoder · flatten T timesteps as extra channels | ~25 M |
| 2 | **ConvLSTM** | Recurrent cell propagates hidden state across T days | ~1 M |
| 3 | **UTAE** | Temporal attention at the bottleneck (U-TAE from Gerard et al.) | ~1.1 M |

The MiT-B2 encoder belongs to the SegFormer family — a hierarchical Vision Transformer that produces multi-scale feature maps, making it naturally compatible with a UNet decoder.

### Input Features (23 channels)

| Group | Features |
|-------|----------|
| Spectral | VIIRS M11, I2, I1 |
| Vegetation | NDVI, EVI2 |
| Weather | Total precipitation, Wind speed, Min/Max temperature, Specific humidity, Energy Release Component |
| Topography | Elevation, PDSI, Slope, Aspect |
| Landcover | One-hot encoded classes (8 channels) |
| Fire history | Active fire mask (previous T days) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input days (T) | 5 |
| Epochs | 20 |
| Batch size | 128 |
| Seed | 42 |
| Crop size | 128 × 128 |
| Loss | BCE + Dice |
| Optimiser | Adam |

### Ablation — Architecture Comparison (Fold 0)

| Model | Test AP | Test F1 | Test IoU |
|-------|---------|---------|----------|
| ViT-UNet | — | — | — |
| ConvLSTM | — | — | — |
| UTAE | — | — | — |

*Results to be filled in after training. Raw logs saved to `experiments/`.*

### 3-Fold Cross-Validation (Best Architecture)

| Fold | Test AP | Test F1 | Test IoU | Test Loss |
|------|---------|---------|----------|-----------|
| 0 | — | — | — | — |
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| **Mean** | — | — | — | — |
| **Std** | — | — | — | — |

*Results to be filled in after training.*

---

## Dataset

**WildfireSpreadTS (WSTS)** — daily satellite composites of US wildfires.
We use the **10% subset** (Colab-friendly) with the standard year-based fold assignment:

| Fold | Train | Val | Test |
|------|-------|-----|------|
| 0 | 2018 | 2019 | 2020 |
| 1 | 2019 | 2020 | 2018 |
| 2 | 2020 | 2018 | 2019 |

Download: [Zenodo](https://doi.org/10.5281/zenodo.8006177) — CC-BY-4.0

---

## Repository Structure

```
├── cfgs/           # YAML configuration files
├── experiments/    # Results, ablation tables, plots
├── notebooks/      # Model notebooks (one per branch)
├── src/            # Shared utilities and dataloaders
├── .gitignore
└── README.md
```

---

## Getting Started

All notebooks are designed to run on **Google Colab** with data on Google Drive.

```bash
# Clone and switch to a model branch
git clone https://github.com/xhibix/WildfireSpread_prediction.git
cd WildfireSpread_prediction
git checkout model/vit-unet   # or your own branch
```

Open the notebook in Colab and follow the path setup in the first cell.

### Dependencies

```bash
pip install segmentation-models-pytorch pytorch-lightning==2.0.1 \
    torchmetrics einops jsonargparse[signatures]==4.20.1 wandb h5py
```

---

## Workflow

Each team member contributes on their own branch:

1. `git checkout -b model/<your-model-name>` from `main`
2. Add your notebook to `notebooks/`
3. Push and add your row to the Models table above
4. We merge all branches into `main` once everyone is done

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
