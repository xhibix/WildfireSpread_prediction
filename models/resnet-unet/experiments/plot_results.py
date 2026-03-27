"""
plot_results.py
===============
Generates all figures after training is complete:

  Figure 1 - Learning curves (loss + val AP per epoch, per fold)
  Figure 2 - Test AP bar chart with parameter counts
  Figure 3 - Prediction maps per config/fold (prob heatmap style)
  Figure 4 - Baseline vs Improved overlay comparison (TP/FP/FN colour coding)

Usage:
  python experiments/plot_results.py
  python experiments/plot_results.py --data_dir WildfireSpreadTS_10pct --device cuda

Flags:
  --results_dir   PATH   default: experiments/results
  --data_dir      PATH   default: WildfireSpreadTS_10pct
  --n_samples     INT    test samples to visualise per config/fold  (default: 4)
  --threshold     FLOAT  prob threshold for binary prediction  (default: 0.5)
  --device        auto|cpu|cuda
"""

import os, sys, json, argparse, warnings, tempfile
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.utils.data import ConcatDataset

from dataloader.FireSpreadDataset import FireSpreadDataset
from dataset_improved import ImprovedFireSpreadDataset
from train_ablation_firediff import FireDiffOnlyDataset
from train_ablation_wind import WindOnlyDataset
from models.SMPModel import SMPModel
from data_utils import FOLDS, DATA_DIR, compute_pos_class_weight


# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

CMAP_FIRE = ListedColormap(["#1a1a2e", "#e94560"])   # binary: no-fire / fire
CMAP_PROB = "YlOrRd"                                   # continuous probability

# Overlay colour map: 0=TN(grey), 1=TP(yellow), 2=FP(green), 3=FN(red)
OVERLAY_COLORS = ["#4a4a4a", "#f5e642", "#4ecb57", "#e94560"]
CMAP_OVERLAY   = ListedColormap(OVERLAY_COLORS)
NORM_OVERLAY   = BoundaryNorm([0, 1, 2, 3, 4], len(OVERLAY_COLORS))

CONFIG_COLORS = {
    "Baseline":        "#4878d0",
    "Wind u/v only":   "#ee854a",
    "Fire-diff only":  "#6acc65",
    "Boundary only":   "#d65f5f",
    "Improved (all)":  "#b47cc7",
}

# Parameter counts read from training logs
PARAM_COUNTS = {
    "Baseline":        "14.9 M",
    "Wind u/v only":   "14.9 M",
    "Fire-diff only":  "15.0 M",
    "Boundary only":   "14.9 M",
    "Improved (all)":  "15.0 M",
}

# (result_json, display_name, log_prefix, n_channels, is_improved, DatasetClass)
# DatasetClass=None -> resolved at runtime (FireSpreadDataset or ImprovedFireSpreadDataset)
CONFIGS = [
    ("baseline_results.json",         "Baseline",        "baseline_T5",          200, False, None),
    ("ablation_wind_results.json",     "Wind u/v only",   "ablation_wind_T5",     200, False, "WindOnlyDataset"),
    ("ablation_firediff_results.json", "Fire-diff only",  "ablation_firediff_T5", 205, False, "FireDiffOnlyDataset"),
    ("ablation_boundary_results.json", "Boundary only",   "ablation_boundary_T5", 200, False, None),
    ("improved_results.json",          "Improved (all)",  "improved_T5",          205, True,  None),
]

# Map string name -> class (resolved after imports)
_DATASET_CLASS_MAP = {
    "WindOnlyDataset":      lambda: WindOnlyDataset,
    "FireDiffOnlyDataset":  lambda: FireDiffOnlyDataset,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, filename: str):
    p = results_dir / filename
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_csv_metrics(logs_dir: Path, run_prefix: str):
    records = []
    for version_dir in sorted(logs_dir.glob(f"{run_prefix}*/version_*/metrics.csv")):
        try:
            df = pd.read_csv(version_dir)
        except Exception:
            continue
        parts = version_dir.parent.parent.name.split("_")
        fold_str = [p for p in parts if p.startswith("fold")]
        if not fold_str:
            continue
        fold_id = int(fold_str[0].replace("fold", ""))
        df["fold"] = fold_id
        records.append(df)
    if not records:
        return None
    return pd.concat(records, ignore_index=True)


def find_best_checkpoint(run_name: str):
    ckpt_dir = Path(tempfile.gettempdir()) / "wfire_ckpts" / run_name
    # pick the most recently modified checkpoint (avoids stale best.ckpt from old runs)
    ckpts = list(ckpt_dir.glob("best*.ckpt"))
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: p.stat().st_mtime))


def build_test_dataset(data_root: Path, fold_def: dict, is_improved: bool, dataset_class_name=None):
    if dataset_class_name and dataset_class_name in _DATASET_CLASS_MAP:
        DatasetClass = _DATASET_CLASS_MAP[dataset_class_name]()
    elif is_improved:
        DatasetClass = ImprovedFireSpreadDataset
    else:
        DatasetClass = FireSpreadDataset
    datasets = []
    for split_subdir, year in fold_def["test"]:
        ds = DatasetClass(
            data_dir=str(data_root / split_subdir),
            included_fire_years=[year],
            n_leading_observations=5,
            crop_side_length=128,
            load_from_hdf5=True,
            is_train=False,
            remove_duplicate_features=False,
            stats_years=fold_def["stats_years"],
            features_to_keep=None,
            return_doy=False,
        )
        datasets.append(ds)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def load_model(ckpt_path: str, is_improved: bool, n_channels: int, pos_weight: float):
    if is_improved:
        from train_improved import ImprovedSMPModel
        model = ImprovedSMPModel(
            encoder_name="resnet18", encoder_weights=None,
            n_channels=n_channels, flatten_temporal_dimension=True,
            pos_class_weight=pos_weight, loss_function="Focal",
        )
    else:
        class _M(SMPModel):
            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters())
        model = _M(
            encoder_name="resnet18", encoder_weights=None,
            n_channels=n_channels, flatten_temporal_dimension=True,
            pos_class_weight=pos_weight, loss_function="Focal",
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    return model


def pick_samples(test_ds, n_samples, seed=42):
    """Return indices of samples with fire pixels in ground truth."""
    rng  = np.random.default_rng(seed)
    idxs = []
    for _ in range(3000):
        if len(idxs) >= n_samples:
            break
        i = int(rng.integers(0, len(test_ds)))
        try:
            _, y_i = test_ds[i]
        except Exception:
            continue
        if y_i.sum() > 0:
            idxs.append(i)
    if not idxs:
        idxs = list(range(min(n_samples, len(test_ds))))
    return idxs


def run_inference(model, test_ds, idxs, device, threshold):
    """
    Returns list of (prev_fire, gt, prob, binary, prob_vmax) numpy arrays.

    Binary threshold is adaptive: uses the top-0.5% percentile of the
    probability map so predictions are always visible even when raw probs
    are small (fire is ~0.1% of pixels so sigmoid outputs are tiny).
    prob_vmax is the 99.9th percentile of prob, used to scale the colormap
    so real variation in the heatmap is visible.
    """
    results = []
    with torch.no_grad():
        for idx in idxs:
            x_i, y_i = test_ds[idx]
            prev_fire = x_i[-1, 39, :, :].numpy()
            logits = model(x_i.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            prob   = torch.sigmoid(torch.tensor(logits)).numpy()

            # Adaptive threshold: flag top 0.5% of pixels as fire
            # (matches expected fire prevalence ~0.1-0.5% per patch)
            adaptive_thr = float(np.percentile(prob, 99.5))
            binary = (prob >= max(adaptive_thr, 1e-6)).astype(np.uint8)

            # Colour scale: stretch to actual signal range
            prob_vmax = float(np.percentile(prob, 99.9))
            prob_vmax = max(prob_vmax, 1e-6)

            gt = y_i.numpy().astype(np.uint8)
            results.append((prev_fire, gt, prob, binary, prob_vmax))
    return results


# ---------------------------------------------------------------------------
# Figure 1 -- Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(logs_dir: Path, out_dir: Path):
    available = []
    for _, display, prefix, _, _, _ in CONFIGS:
        df = load_csv_metrics(logs_dir, prefix)
        if df is not None:
            available.append((display, prefix, df))

    if not available:
        print("  No log data found -- skipping learning curves.")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
    fig.suptitle("Learning Curves -- Train Loss (top) and Val AP (bottom)", fontsize=13)

    for col, (label, prefix, df) in enumerate(available):
        color = CONFIG_COLORS.get(label, f"C{col}")
        ax_loss, ax_ap = axes[0, col], axes[1, col]

        for fold_id, gdf in df.groupby("fold"):
            loss_rows = gdf.dropna(subset=["train_loss_epoch"])
            if not loss_rows.empty:
                ax_loss.plot(loss_rows["epoch"], loss_rows["train_loss_epoch"],
                             alpha=0.85, label=f"Fold {fold_id}")
            ap_rows = gdf.dropna(subset=["val_avg_precision"])
            if not ap_rows.empty:
                ax_ap.plot(ap_rows["epoch"], ap_rows["val_avg_precision"],
                           alpha=0.85, label=f"Fold {fold_id}")

        ax_loss.set_title(label, fontsize=10, color=color, fontweight="bold")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Train Loss")
        ax_loss.legend(fontsize=8); ax_loss.grid(True, alpha=0.3)
        ax_ap.set_xlabel("Epoch"); ax_ap.set_ylabel("Val AP")
        ax_ap.legend(fontsize=8); ax_ap.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig1_learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 -- Test AP bar chart with parameter counts
# ---------------------------------------------------------------------------

def plot_ap_comparison(results_dir: Path, out_dir: Path):
    datasets = {}
    for filename, display, _, _, _, _ in CONFIGS:
        rows = load_results(results_dir, filename)
        if rows:
            datasets[display] = sorted(rows, key=lambda r: r["fold"])

    if not datasets:
        print("  No results found -- skipping AP comparison plot.")
        return

    n_configs = len(datasets)
    n_folds   = 3
    x         = np.arange(n_folds + 1)
    width     = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_configs), 5))

    for i, (name, rows) in enumerate(datasets.items()):
        aps     = [r["test_AP"] for r in rows]
        mean_ap = np.mean(aps)
        values  = aps + [mean_ap]
        offset  = (i - n_configs / 2 + 0.5) * width
        color   = CONFIG_COLORS.get(name, f"C{i}")
        params  = PARAM_COUNTS.get(name, "?")
        label   = f"{name}\n({params} params)"
        bars    = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=6.5)

    ax.axvline(n_folds - 0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(n_folds)] + ["Mean"])
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Test AP -- Baseline vs Ablations vs Improved  (test year = 2021, WildfireSpreadTS_10pct)")
    ax.legend(fontsize=8, loc="upper left", ncol=n_configs)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.18))

    plt.tight_layout()
    out_path = out_dir / "fig2_test_ap_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 -- Probability heatmap predictions (one file per config x fold)
# ---------------------------------------------------------------------------

def plot_predictions_for_config(cfg_name, run_prefix, n_channels, is_improved,
                                 data_root, fold_id, fold_def, out_dir,
                                 n_samples=4, threshold=0.5, device="cpu",
                                 dataset_class_name=None):
    ckpt = find_best_checkpoint(f"{run_prefix}_fold{fold_id}")
    if ckpt is None:
        print(f"  No checkpoint for {run_prefix}_fold{fold_id} -- skipping.")
        return None

    pos_weight = compute_pos_class_weight(fold_def["stats_years"])
    try:
        test_ds = build_test_dataset(data_root, fold_def, is_improved, dataset_class_name)
        model   = load_model(ckpt, is_improved, n_channels, pos_weight).to(device)
    except Exception as e:
        print(f"  Error ({cfg_name} fold {fold_id}): {e}")
        return None

    idxs    = pick_samples(test_ds, n_samples, seed=42 + fold_id)
    samples = run_inference(model, test_ds, idxs, device, threshold)

    n_rows = len(samples)
    gs  = gridspec.GridSpec(n_rows, 5, width_ratios=[1, 1, 1, 1, 0.06],
                            wspace=0.04, hspace=0.15)
    fig = plt.figure(figsize=(15, 3.5 * n_rows))
    fig.suptitle(f"{cfg_name}  |  Fold {fold_id}  (test=2021)",
                 fontsize=12, fontweight="bold", y=1.01)

    col_titles = ["Prev-day fire mask", "Ground truth (next day)",
                  "Predicted probability\n(scaled to signal range)", "Predicted fire\n(top 0.5% threshold)"]
    last_im = None

    for row, (prev_fire, gt, prob, binary, prob_vmax) in enumerate(samples):
        panels = [prev_fire, gt, prob, binary]
        cmaps  = [CMAP_FIRE, CMAP_FIRE, CMAP_PROB, CMAP_FIRE]
        vmaxs  = [1, 1, prob_vmax, 1]   # stretch prob to actual signal range

        for col, (panel, cmap, vmax) in enumerate(zip(panels, cmaps, vmaxs)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(panel, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Sample {idxs[row]}", fontsize=8)
            if col == 2:
                last_im = im

    if last_im is not None:
        plt.colorbar(last_im, cax=fig.add_subplot(gs[:, 4]), label="Probability")

    patch_no  = mpatches.Patch(color="#1a1a2e", label="No fire")
    patch_yes = mpatches.Patch(color="#e94560", label="Fire")
    fig.legend(handles=[patch_no, patch_yes], loc="lower center",
               ncol=2, bbox_to_anchor=(0.42, -0.03), fontsize=9)

    safe = (cfg_name + f"_fold{fold_id}").replace(" ", "_").replace("(","").replace(")","").replace("/","")
    out_path = out_dir / f"fig3_predictions_{safe}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")
    return idxs   # return indices so Fig4 can reuse them


# ---------------------------------------------------------------------------
# Figure 4 -- Baseline vs Improved overlay comparison (TP/FP/FN)
# ---------------------------------------------------------------------------

def make_overlay(gt: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """
    Encode prediction quality as integer map:
      0 = TN  (grey)
      1 = TP  (yellow)
      2 = FP  (green)
      3 = FN  (red)
    """
    overlay = np.zeros_like(gt, dtype=np.uint8)
    overlay[(gt == 1) & (binary == 1)] = 1   # TP
    overlay[(gt == 0) & (binary == 1)] = 2   # FP
    overlay[(gt == 1) & (binary == 0)] = 3   # FN
    return overlay


def plot_baseline_vs_improved(data_root, results_dir, out_dir,
                               n_samples=4, threshold=0.5, device="cpu"):
    # Check both result files exist
    bl_rows  = load_results(results_dir, "baseline_results.json")
    imp_rows = load_results(results_dir, "improved_results.json")
    if not bl_rows or not imp_rows:
        print("  Missing baseline or improved results -- skipping comparison figure.")
        return

    for fold_id, fold_def in enumerate(FOLDS):
        bl_ckpt  = find_best_checkpoint(f"baseline_T5_fold{fold_id}")
        imp_ckpt = find_best_checkpoint(f"improved_T5_fold{fold_id}")
        if bl_ckpt is None or imp_ckpt is None:
            print(f"  Missing checkpoint for fold {fold_id} -- skipping.")
            continue

        print(f"  Baseline vs Improved overlay: fold {fold_id}")
        pos_weight_bl  = compute_pos_class_weight(fold_def["stats_years"])
        pos_weight_imp = compute_pos_class_weight(fold_def["stats_years"])

        try:
            test_ds_bl  = build_test_dataset(data_root, fold_def, is_improved=False)
            test_ds_imp = build_test_dataset(data_root, fold_def, is_improved=True)
            model_bl    = load_model(bl_ckpt,  False, 200, pos_weight_bl).to(device)
            model_imp   = load_model(imp_ckpt, True,  205, pos_weight_imp).to(device)
        except Exception as e:
            print(f"  Error fold {fold_id}: {e}")
            continue

        # use same shared indices (fire pixels in GT)
        idxs = pick_samples(test_ds_bl, n_samples, seed=100 + fold_id)

        bl_samples  = run_inference(model_bl,  test_ds_bl,  idxs, device, threshold)
        imp_samples = run_inference(model_imp, test_ds_imp, idxs, device, threshold)

        n_rows = len(idxs)
        # columns: GT | Baseline prob | Improved prob | Baseline overlay | Improved overlay
        # Width ratios: equal image cols + thin colorbar
        gs = gridspec.GridSpec(n_rows, 6,
                               width_ratios=[1, 1, 1, 1, 1, 0.06],
                               wspace=0.04, hspace=0.20)
        fig = plt.figure(figsize=(20, 3.5 * n_rows))
        fig.suptitle(
            f"Baseline vs Improved -- Fold {fold_id} (test=2021)\n"
            f"Prob maps scaled to signal range  |  Overlay: Yellow=TP  Green=FP  Red=FN  Grey=TN",
            fontsize=11, fontweight="bold", y=1.02,
        )

        col_titles = ["True next-day fire",
                      "Baseline prob map", "Improved prob map",
                      "Baseline TP/FP/FN", "Improved TP/FP/FN"]

        last_prob_im = None
        for row, (bl, imp) in enumerate(zip(bl_samples, imp_samples)):
            prev_fire, gt, bl_prob,  bl_binary,  bl_vmax  = bl
            _,         _,  imp_prob, imp_binary, imp_vmax = imp

            bl_overlay  = make_overlay(gt, bl_binary)
            imp_overlay = make_overlay(gt, imp_binary)

            # Use the same vmax across both prob maps per row for fair comparison
            shared_vmax = max(bl_vmax, imp_vmax, 1e-6)

            panels = [gt,      bl_prob, imp_prob, bl_overlay,  imp_overlay]
            cmaps  = [CMAP_FIRE, CMAP_PROB, CMAP_PROB, CMAP_OVERLAY, CMAP_OVERLAY]
            vmins  = [0, 0, 0, 0, 0]
            vmaxs  = [1, shared_vmax, shared_vmax, 4, 4]

            for col, (panel, cmap, vmin, vmax) in enumerate(
                    zip(panels, cmaps, vmins, vmaxs)):
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9,
                                 fontweight="bold" if col >= 3 else "normal")
                if col == 0:
                    ax.set_ylabel(f"Sample {idxs[row]}", fontsize=8)
                if col == 1:  # save for colorbar
                    last_prob_im = im

        # Probability colorbar in dedicated column
        if last_prob_im is not None:
            cb_ax = fig.add_subplot(gs[:, 5])
            cbar = plt.colorbar(last_prob_im, cax=cb_ax)
            cbar.set_label("Fire probability\n(scaled)", fontsize=8)

        # Overlay legend
        patches = [
            mpatches.Patch(color=OVERLAY_COLORS[0], label="TN"),
            mpatches.Patch(color=OVERLAY_COLORS[1], label="TP (correct)"),
            mpatches.Patch(color=OVERLAY_COLORS[2], label="FP (false alarm)"),
            mpatches.Patch(color=OVERLAY_COLORS[3], label="FN (missed fire)"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.42, -0.03), fontsize=9)

        out_path = out_dir / f"fig4_baseline_vs_improved_fold{fold_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="experiments/results")
    p.add_argument("--data_dir",    default=DATA_DIR)
    p.add_argument("--n_samples",   type=int,   default=4)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--device",      default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = (_ROOT / args.results_dir
                   if not Path(args.results_dir).is_absolute()
                   else Path(args.results_dir))
    logs_dir  = results_dir / "logs"
    out_dir   = results_dir / "figures"
    data_root = (_ROOT / args.data_dir
                 if not Path(args.data_dir).is_absolute()
                 else Path(args.data_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nGenerating figures -> {out_dir}  (device={device})")
    print("-" * 55)

    print("\n[Fig 1] Learning curves...")
    plot_learning_curves(logs_dir, out_dir)

    print("\n[Fig 2] Test AP comparison (with param counts)...")
    plot_ap_comparison(results_dir, out_dir)

    print("\n[Fig 3] Per-config probability prediction maps...")
    for filename, display, prefix, n_ch, is_imp, ds_cls in CONFIGS:
        rows = load_results(results_dir, filename)
        if not rows:
            print(f"  No results for {display} -- skipping.")
            continue
        for fold_id, fold_def in enumerate(FOLDS):
            print(f"  Plotting: {display} fold {fold_id}")
            plot_predictions_for_config(
                display, prefix, n_ch, is_imp,
                data_root, fold_id, fold_def, out_dir,
                n_samples=args.n_samples, threshold=args.threshold, device=device,
                dataset_class_name=ds_cls,
            )

    print("\n[Fig 4] Baseline vs Improved overlay comparison (TP/FP/FN)...")
    plot_baseline_vs_improved(
        data_root, results_dir, out_dir,
        n_samples=args.n_samples, threshold=args.threshold, device=device,
    )

    print("\nDone. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
