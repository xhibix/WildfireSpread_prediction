# RUNME — Wildfire Spread Prediction Experiments

This document explains how to reproduce the baseline and improved model results on
the `WildfireSpreadTS_10pct` dataset, including a full ablation study isolating each
proposed improvement.

---

## 1. The Dataset — WildfireSpreadTS_10pct

### What the dataset contains

`WildfireSpreadTS_10pct` is a 10%-sample subset of the **WildfireSpreadTS** dataset
(Rufin et al., 2023), a multi-temporal satellite-derived dataset of wildfires in the
continental United States.  Each data point is a daily snapshot of an active fire event,
cropped to a 128×128 pixel patch (each pixel ≈ 375 m resolution).

Each HDF5 file stores one fire event and contains:
- **`x`**: shape `(T, 23, H, W)` — 23 geophysical input channels over T consecutive days
- **`y`**: shape `(H, W)` — binary fire mask for the *next* day (the prediction target)

The 23 input channels cover:
| Group | Channels | Description |
|---|---|---|
| Fire | 0 | Active fire mask (VIIRS) |
| Weather (observed) | 1–7 | Temperature, humidity, wind speed, wind direction, precipitation, ... |
| Weather (forecast) | 8–19 | ECMWF next-day forecast for the same variables |
| Topography | 20–21 | Elevation, slope |
| Vegetation | 22 | NDVI / fuel load proxy |

> **Note:** The codebase internally expands each day's 23 raw channels into ~40 channels
> via one-hot encoding of land cover, binary fire mask normalisation, and standardisation
> per training statistics.  After preprocessing, each timestep has **40 channels**
> (or **41** with the fire-diff improvement).

### Pre-split layout

The dataset arrives **already split** into train / val / test sub-directories:

```
WildfireSpreadTS_10pct/
├── train/
│   ├── 2018/   *.hdf5   (24 fire events)
│   └── 2019/   *.hdf5
├── val/
│   └── 2020/   *.hdf5   (20 fire events)
└── test/
    └── 2021/   *.hdf5   (14 fire events)
```

### Why the test set is always 2021

Fire climatology has strong inter-annual structure (drought years, La Niña, etc.).
Holding out 2021 as the test set gives a completely unseen year for final evaluation,
preventing any temporal leakage.  All models use the same test set so comparisons are fair.

### 3-fold cross-validation

Because the dataset is small (≈58 fire events in 2018–2020), we use 3-fold
cross-validation to estimate generalisation variance without touching the test set.
The three folds re-arrange which years are used for training vs. validation:

| Fold | Train data | Val data | Test data | Stats computed from |
|---|---|---|---|---|
| 0 | train/2018 + train/2019 | val/2020 | test/2021 | 2018 + 2019 |
| 1 | train/2018 + val/2020 | train/2019 | test/2021 | 2018 + 2020 |
| 2 | train/2019 + val/2020 | train/2018 | test/2021 | 2019 + 2020 |

**Important:** per-channel normalisation statistics (mean, std) are computed
*only* from the training years of each fold, not from the full dataset.  This
prevents data leakage from the validation year into the standardisation step.

### Evaluation metric

We use **Average Precision (AP)**, the area under the Precision–Recall curve.
AP is preferred over AUC-ROC for this heavily class-imbalanced task (fire pixels
are rare, < 0.5% of all pixels).  A higher AP = better fire-spread prediction.

---

## 2. Models

### 2.1 Architecture — Res18-UNet

All experiments use the same backbone: a **U-Net with a ResNet-18 encoder**
(from the `segmentation_models_pytorch` library, pretrained on ImageNet).

- Input: `[B, N_OBS × C_per_step, H, W]` — temporal observations are flattened
  along the channel dimension (`flatten_temporal_dimension=True`).
- Output: `[B, 1, H, W]` — per-pixel logits (sigmoid → probability map).
- Loss: **Focal loss** (α set from class imbalance, γ=2).
- Optimiser: **AdamW** (lr=1e-3, weight_decay=1e-4).
- Checkpoint selection: best **val AP** across 20 epochs.

### 2.2 Baseline

**Script:** `experiments/train_baseline.py`
**Input:** T=5 days × 40 ch/step = **200 channels**
**Improvements:** none (paper's model, unchanged)

The baseline is the Res18-UNet exactly as used in the original WildfireSpreadTS paper.
It receives 5 consecutive days of the 40 pre-processed channels and uses the standard
Focal loss with no spatial weighting.  This is our performance floor.

### 2.3 Ablation — Wind u/v re-encoding only

**Script:** `experiments/train_ablation_wind.py`
**Input:** T=5 × 40 ch/step = **200 channels**
**Improvements:** wind re-encoding [ON], fire-diff [OFF], boundary-loss [OFF]

Raw wind data is stored as (speed, direction-degrees).  A direction is a circular
quantity: 359° and 1° are almost the same wind direction but numerically far apart.
Feeding degrees directly into a linear model is sub-optimal.

This improvement converts wind to **Cartesian u/v components**:
```
u = -speed × sin(direction_rad)   # westward/eastward component
v = -speed × cos(direction_rad)   # southward/northward component
```
Applied to observed wind (channels 6, 7) and forecast wind (channels 18, 19).
The u/v representation is continuous, rotation-equivariant, and physically meaningful.
The channel count stays at 40 (replacing speed+dir with u+v, not adding channels).

### 2.4 Ablation — Temporal fire diff channel only

**Script:** `experiments/train_ablation_firediff.py`
**Input:** T=5 × 41 ch/step = **205 channels**
**Improvements:** wind re-encoding [OFF], fire-diff [ON], boundary-loss [OFF]

Fire spread is a dynamic process: the *change* in fire extent over time encodes
spread direction and speed more directly than absolute fire masks.  For each timestep t:
```
fire_diff[t] = binary_fire_mask[t] - binary_fire_mask[t-1]   (zero for t=0)
```
This extra channel is appended to each timestep's feature vector, increasing the
channel count from 40 to 41 per step (200 → 205 total input channels).

Positive values indicate pixels that newly caught fire; negative values indicate
pixels that were recorded as extinguished (rare).  The model can use this to
identify the active frontier of the fire.

### 2.5 Ablation — Boundary-weighted focal loss only

**Script:** `experiments/train_ablation_boundary.py`
**Input:** T=5 × 40 ch/step = **200 channels**
**Improvements:** wind re-encoding [OFF], fire-diff [OFF], boundary-loss [ON]

Standard Focal loss treats all pixels equally (after the α class-weight adjustment).
However, fire-spread prediction errors at the **fire frontier** matter most:
predicting the current burnt interior correctly is easy, but predicting which
unburnt pixels will catch fire tomorrow is the hard and high-stakes part.

This improvement identifies the boundary ring of the fire at the last input timestep:
```
dilated   = max_pool2d(prev_fire, kernel=11, padding=5)  # 5-pixel dilation
boundary  = dilated > 0.5 AND NOT prev_fire > 0.5        # ring just outside the fire
weight_map[boundary] = 5.0                                # 5x loss weight there
```
The weighted focal loss is computed as `mean(focal_element_loss × weight_map)`.
This focuses the model's attention on the advancing fire edge.

### 2.6 Improved (all three)

**Script:** `experiments/train_improved.py`
**Input:** T=5 × 41 ch/step = **205 channels**
**Improvements:** wind re-encoding [ON], fire-diff [ON], boundary-loss [ON]

Combines all three improvements simultaneously.  This is our proposed model;
the ablation scripts tell us which improvements actually help and which are redundant
or harmful in combination.

---

## 3. Running the experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Ensure the dataset is at `WildfireSpreadTS_10pct/` relative to the project root,
or pass `--data_dir /path/to/WildfireSpreadTS_10pct` to each script.

**GPU (recommended):** RTX 5060 or any CUDA GPU.  For PyTorch 2.11+ with CUDA 12.8:
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Run all five models (in order)

```bash
# 1. Baseline (paper's model, no improvements)
python experiments/train_baseline.py --epochs 20 --batch_size 16 --device cuda

# 2. Ablation: wind u/v re-encoding only
python experiments/train_ablation_wind.py --epochs 20 --batch_size 16 --device cuda

# 3. Ablation: temporal fire diff only
python experiments/train_ablation_firediff.py --epochs 20 --batch_size 16 --device cuda

# 4. Ablation: boundary-weighted focal loss only
python experiments/train_ablation_boundary.py --epochs 20 --batch_size 16 --device cuda

# 5. Improved model (all three improvements together)
python experiments/train_improved.py --epochs 20 --batch_size 16 --device cuda
```

Each script trains **3 folds × 20 epochs** and saves results to
`experiments/results/<name>_results.json`.

### Generate plots

After all training is complete:

```bash
python experiments/plot_results.py --data_dir WildfireSpreadTS_10pct --device cuda
```

This generates three figures in `experiments/results/figures/`:
- **fig1_learning_curves.png** — loss and val AP per epoch for all configs
- **fig2_test_ap_comparison.png** — bar chart comparing all 5 models across folds
- **fig3_predictions_\*.png** — one file per config/fold: prev-day fire | GT | prob | binary

### Flags common to all training scripts

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `20` | Training epochs per fold |
| `--batch_size` | `16` | Mini-batch size (reduce to 4–8 if GPU OOM) |
| `--num_workers` | `0` | DataLoader workers (keep 0 on Windows) |
| `--device` | `auto` | `cuda` or `cpu` (auto-detects) |
| `--data_dir` | `WildfireSpreadTS_10pct` | Path to dataset root |

### Extra flags for boundary scripts

| Flag | Default | Meaning |
|---|---|---|
| `--boundary_weight` | `5.0` | Loss multiplier on frontier pixels |
| `--dilation_radius` | `5` | Pixel radius of boundary ring |

---

## 4. Output files

```
experiments/results/
├── baseline_results.json
├── ablation_wind_results.json
├── ablation_firediff_results.json
├── ablation_boundary_results.json
├── improved_results.json
├── logs/
│   ├── baseline_T5_fold0/version_0/metrics.csv
│   ├── ablation_wind_T5_fold0/...
│   ├── ablation_firediff_T5_fold0/...
│   ├── ablation_boundary_T5_fold0/...
│   └── improved_T5_fold0/...
└── figures/
    ├── fig1_learning_curves.png
    ├── fig2_test_ap_comparison.png
    └── fig3_predictions_*.png
```

Each JSON result file contains one entry per fold:
```json
{
  "fold": 0,
  "n_obs": 5,
  "train_years": [2018, 2019],
  "val_years": [2020],
  "test_years": [2021],
  "best_val_AP": 0.XXXX,
  "test_AP": 0.XXXX,
  "test_F1": 0.XXXX
}
```

---

## 5. Interpreting results

The ablation study isolates each improvement's contribution:

| Model | Wind u/v | Fire diff | Boundary loss | Expected effect |
|---|:---:|:---:|:---:|---|
| Baseline | - | - | - | Performance floor |
| Wind only | ON | - | - | Marginal gain from better wind representation |
| Fire-diff only | - | ON | - | Moderate gain from explicit spread signal |
| Boundary only | - | - | ON | Gain from focusing loss on frontier |
| Improved (all) | ON | ON | ON | Combined gain; best expected AP |

**How to read the results:**
- Compare each ablation vs. the baseline to see the individual contribution.
- Compare improved (all) vs. each ablation to see if interactions between improvements matter.
- Mean AP across folds is more reliable than any single fold on this small dataset.
- If improved (all) > sum of individual gains, the improvements interact synergistically.

---

## 6. Troubleshooting

**Out of memory:** Reduce `--batch_size` (try 4 or 8).

**Windows DataLoader errors:** Keep `--num_workers 0` (the default).

**First epoch is very slow on Blackwell GPU (RTX 5060):** Expected — PyTorch JIT-compiles
CUDA kernels for sm_120 on first use. Subsequent epochs run at normal speed.

**`ModuleNotFoundError: dataloader`:** Run scripts from the **project root**, not from `experiments/`:
```bash
# Correct
python experiments/train_baseline.py

# Wrong
cd experiments && python train_baseline.py
```

**Quick sanity check (2 epochs, CPU):**
```bash
python experiments/train_baseline.py --epochs 2 --batch_size 4 --device cpu
```
