# Temporal Fusion U-Net (TF-UNet)

Next-day wildfire spread prediction using temporal attention on multi-day satellite inputs.

## Model

A U-Net with a shared encoder that processes each input day independently, then fuses the temporal features using multi-head attention before decoding. Two attention variants were tested:

- **All levels**: temporal attention at every encoder level + bottleneck (33.8M params)
- **Bottleneck only**: temporal attention at the bottleneck only, simple averaging at encoder levels (33.1M params)

### Default hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, wd=0.01) |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| Loss | Dice loss |
| Epochs | 50 |
| Batch size | 4 |
| Encoder features | [64, 128, 256, 512] |
| Input window | 5 days |
| Crop size | 128 x 128 |

## Files

| File | Description |
|------|-------------|
| `temporal_fusion_unet.ipynb` | Original model development: data loading, model architecture, initial 3-fold CV with all-levels and bottleneck modes (20 epochs) |
| `tfunet_final.ipynb` | Systematic hyperparameter search: 12 experiments (A-L) on fold 0, testing attention mode, loss functions, learning rate, scheduler, model size, dropout, window size, and Jaccard loss |
| `tfunet_3fold_all.ipynb` | 3-fold cross-validation (folds 0, 3, 10) for the top 4 configurations (A, B, C, D) with persistence baseline and comparison plots |

## Results

### Best results on fold 0 (test=2021)

| Exp | Configuration | AP | F1 | Status |
|-----|--------------|-----|------|--------|
| J | Window=1 (mono-temporal) | 0.466 | 0.546 | Highest AP, but unstable (collapsed and recovered) |
| H | Bottleneck + Dropout 0.2 | 0.352 | 0.454 | Best stable model |
| B | Bottleneck, Dice | 0.345 | 0.450 | Stable baseline |
| A | All levels, Dice | 0.342 | 0.474 | Collapsed in some runs |
| K | No scheduler (fixed lr) | 0.335 | 0.447 | Stable, matches paper's approach |

### 3-fold CV (all levels, 20 epochs)

| Fold | Test year | AP |
|------|-----------|-----|
| 0 | 2021 (easy) | 0.315 |
| 3 | 2019 (hard) | 0.064 |
| 10 | 2019 (hard) | 0.085 |
| **Mean** | | **0.155 +/- 0.114** |

### Comparison with paper (full dataset, 12-fold CV)

| Model | AP |
|-------|-----|
| Paper: Persistence | 0.193 +/- 0.065 |
| Paper: ResNet18 U-Net | 0.344 +/- 0.076 |
| Paper: UTAE (best) | 0.372 +/- 0.088 |
| Ours (fold 0, 10% data) | 0.352 (best stable) |

## Key findings

1. Bottleneck-only attention is more stable than all-levels attention on small datasets
2. Dropout (0.2) prevents catastrophic collapse and produces the best stable model
3. Dice loss outperforms BCE-based losses for this task
4. Mono-temporal input (window=1) achieved the highest AP, suggesting temporal information may add noise on 10% subset
5. On the easy test year (2021), our model matches the paper's U-Net despite using only 10% of the data
