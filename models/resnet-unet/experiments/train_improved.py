"""
train_improved.py
=================
Trains the improved Res18-UNet on WildfireSpreadTS_10pct for
3 folds x 20 epochs.  All three improvements active simultaneously:

  1. Wind u/v re-encoding     (speed+dir -> u+v)
  2. Temporal fire diff       (fire[t]-fire[t-1] appended per timestep)
  3. Boundary-weighted focal loss  (5x on fire-frontier pixels)

Results -> experiments/results/improved_results.json

Usage
-----
  python experiments/train_improved.py
  python experiments/train_improved.py --epochs 20 --batch_size 16 --device cuda
"""

import os, sys, json, argparse, time, tempfile
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_ROOT / "src"), str(_ROOT / "experiments")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchvision.ops import sigmoid_focal_loss

from models.SMPModel import SMPModel
from dataset_improved import ImprovedFireSpreadDataset
from data_utils import FOLDS, DATA_DIR, make_dataset, make_loader, compute_pos_class_weight

torch.set_float32_matmul_precision("high")

N_OBS      = 5
C_PER_STEP = 41   # 40 standard + 1 fire-diff
N_CHANNELS = N_OBS * C_PER_STEP  # 205


class ImprovedSMPModel(SMPModel):
    """
    Res18-UNet with boundary-weighted focal loss (improvement 3).
    Improvements 1 and 2 (wind u/v + fire diff) are in ImprovedFireSpreadDataset.
    """

    def __init__(self, boundary_weight=5.0, dilation_radius=5,
                 n_timesteps=N_OBS, c_per_step=C_PER_STEP, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.boundary_weight = boundary_weight
        self.dilation_radius = dilation_radius
        self.n_timesteps     = n_timesteps
        self.c_per_step      = c_per_step
        self.save_hyperparameters(ignore=["args", "kwargs"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def _boundary_weighted_focal_loss(self, y_hat, y, x):
        # Fire mask: last timestep, channel 39 (binary fire in 41-ch space)
        prev_fire = x[:, -1, 39, :, :]
        kernel    = 2 * self.dilation_radius + 1
        dilated   = F.max_pool2d(
            prev_fire.unsqueeze(1).float(),
            kernel_size=kernel, stride=1, padding=self.dilation_radius,
        ).squeeze(1)
        boundary   = (dilated > 0.5) & ~(prev_fire > 0.5)
        weight_map = torch.ones_like(y, dtype=torch.float32)
        weight_map[boundary] = self.boundary_weight
        alpha        = 1.0 - float(self.hparams.pos_class_weight)
        element_loss = sigmoid_focal_loss(y_hat, y.float(), alpha=alpha, gamma=2, reduction="none")
        return (element_loss * weight_map).mean()

    def training_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x).squeeze(1)
        loss  = self._boundary_weighted_focal_loss(y_hat, y, x)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss


def train_one_fold(fold_id, fold, data_root, epochs, batch_size, num_workers, device,
                   results_dir, boundary_weight=5.0, dilation_radius=5):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_id} | T={N_OBS} | train={fold['train_years']} val={fold['val_years']} test={fold['test_years']}")
    print(f"  Improvements: wind-uv [ON]  fire-diff [ON]  boundary-loss [ON]")
    print(f"{'='*60}")

    pl.seed_everything(42, workers=True)

    t0 = time.time()
    train_ds = make_dataset(fold["train"], data_root, N_OBS, fold["stats_years"],
                            is_train=True,  DatasetClass=ImprovedFireSpreadDataset)
    val_ds   = make_dataset(fold["val"],   data_root, N_OBS, fold["stats_years"],
                            is_train=False, DatasetClass=ImprovedFireSpreadDataset)
    test_ds  = make_dataset(fold["test"],  data_root, N_OBS, fold["stats_years"],
                            is_train=False, DatasetClass=ImprovedFireSpreadDataset)
    print(f"  Dataset sizes  train:{len(train_ds):,}  val:{len(val_ds):,}  test:{len(test_ds):,}  ({time.time()-t0:.1f}s)")

    train_dl = make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = make_loader(test_ds,  1,           shuffle=False, num_workers=num_workers)

    pos_weight = compute_pos_class_weight(fold["stats_years"])
    model = ImprovedSMPModel(
        boundary_weight=boundary_weight, dilation_radius=dilation_radius,
        n_timesteps=N_OBS, c_per_step=C_PER_STEP,
        encoder_name="resnet18", encoder_weights="imagenet",
        n_channels=N_CHANNELS, flatten_temporal_dimension=True,
        pos_class_weight=pos_weight, loss_function="Focal",
    )
    print(f"  Model: Improved Res18-UNet | n_channels={N_CHANNELS} | pos_weight={pos_weight:.4f} | bnd_wt={boundary_weight}")

    run_name = f"improved_T{N_OBS}_fold{fold_id}"
    ckpt_dir = Path(tempfile.gettempdir()) / "wfire_ckpts" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val_avg_precision",
        mode="max", save_top_k=1, filename="best",
    )
    logger = CSVLogger(save_dir=str(results_dir / "logs"), name=run_name)

    accelerator = "gpu" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1,
        logger=logger, callbacks=[ckpt_cb],
        enable_progress_bar=True, log_every_n_steps=5, deterministic=False,
    )

    trainer.fit(model, train_dl, val_dl)

    best_ckpt    = ckpt_cb.best_model_path
    test_results = trainer.test(model, test_dl, ckpt_path=best_ckpt, verbose=True)

    test_ap = float(test_results[0].get("test_AP", 0.0))
    test_f1 = float(test_results[0].get("test_f1", 0.0))
    val_ap  = float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else 0.0

    result = {
        "fold": fold_id, "n_obs": N_OBS,
        "train_years": fold["train_years"], "val_years": fold["val_years"],
        "test_years":  fold["test_years"],
        "improvements": ["wind_uv", "fire_diff", "boundary_loss"],
        "boundary_weight": boundary_weight, "dilation_radius": dilation_radius,
        "best_val_AP": round(val_ap, 4), "test_AP": round(test_ap, 4), "test_F1": round(test_f1, 4),
    }
    print(f"\n  [OK] Fold {fold_id} | T={N_OBS}  ->  val AP={val_ap:.4f}  test AP={test_ap:.4f}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        default=DATA_DIR)
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--device",          default="auto")
    p.add_argument("--boundary_weight", type=float, default=5.0)
    p.add_argument("--dilation_radius", type=int,   default=5)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = args.data_dir if os.path.isabs(args.data_dir) else str(_ROOT / args.data_dir)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset not found: {data_root}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}   data: {data_root}")

    results_dir = _ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"#  IMPROVED  T={N_OBS}  ({len(FOLDS)} folds x {args.epochs} epochs)")
    print(f"#  boundary_weight={args.boundary_weight}  dilation_radius={args.dilation_radius}")
    print(f"{'#'*60}")

    all_results = []
    for fold_id, fold in enumerate(FOLDS):
        r = train_one_fold(fold_id, fold, data_root, args.epochs, args.batch_size,
                           args.num_workers, device, results_dir,
                           args.boundary_weight, args.dilation_radius)
        all_results.append(r)

    aps = [r["test_AP"] for r in all_results]
    print(f"\n  -- IMPROVED T={N_OBS} Summary --")
    for r in all_results:
        print(f"    Fold {r['fold']}  val AP={r['best_val_AP']:.4f}  test AP={r['test_AP']:.4f}")
    print(f"    Mean test AP = {sum(aps)/len(aps):.4f}")

    out = results_dir / "improved_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out}")

    # Comparison with baseline
    bl_path = results_dir / "baseline_results.json"
    if bl_path.exists():
        bl = json.loads(bl_path.read_text())
        bl5 = [r for r in bl if r.get("n_obs") == N_OBS]
        print("\n" + "="*65)
        print("  COMPARISON TABLE  (test AP)")
        print("="*65)
        print(f"  {'Config':<28} {'Fold 0':>8} {'Fold 1':>8} {'Fold 2':>8} {'Mean':>8}")
        print("-"*65)
        if bl5:
            baps = [r["test_AP"] for r in sorted(bl5, key=lambda r: r["fold"])]
            print(f"  {'Baseline T=5':<28}" + "".join(f"{a:>8.4f}" for a in baps) + f"  {sum(baps)/len(baps):>6.4f}")
        print(f"  {'Improved T=5 (all 3)':<28}" + "".join(f"{a:>8.4f}" for a in aps) + f"  {sum(aps)/len(aps):>6.4f}")
        print("="*65)


if __name__ == "__main__":
    main()
