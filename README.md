# WildfireSpread Prediction

MSc group project — Imperial College London, ML for Climate Change.

We benchmark deep learning architectures for **next-day wildfire spread prediction** on the [WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS) dataset (10 % subset).

---

## Dataset

**WildfireSpreadTS** — daily satellite composites of US wildfires (2018–2020).
Each sample is a multi-temporal stack of **23 input features** (VIIRS bands, NDVI/EVI2, weather, topography, landcover) and a binary active-fire mask as the prediction target.

| Split | Years |
|-------|-------|
| Train | 2018 |
| Val   | 2019 |
| Test  | 2020 |

3-fold cross-validation is applied by rotating the year assignment across folds 0–2.

---

## Models

Each model lives on its own branch and is submitted as a Jupyter notebook under `notebooks/`.
Branches follow the convention `model/<name>`.

| Branch | Model | Author |
|--------|-------|--------|
| `model/vit-unet` | ViT-UNet (MiT-B2 encoder + UNet decoder) | Paolo Salvetti |

> Add your row here when you open your branch.

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

## Workflow

Each team member works on their own branch:

```bash
# Clone the repo
git clone https://github.com/xhibix/WildfireSpread_prediction.git
cd WildfireSpread_prediction

# Switch to your model branch (or create one)
git checkout model/<your-model-name>
```

Notebooks are designed to run on **Google Colab** with data stored on Google Drive.
See the notebook header for dataset path setup instructions.

**When everyone has uploaded their model**, we merge all branches into `main`.

---

## Dependencies

```bash
pip install segmentation-models-pytorch pytorch-lightning==2.0.1 \
    torchmetrics einops jsonargparse[signatures]==4.20.1 wandb h5py
```
