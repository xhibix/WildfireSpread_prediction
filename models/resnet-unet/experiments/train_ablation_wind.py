"""
train_ablation_wind.py
======================
Ablation: WIND U/V RE-ENCODING ONLY.

  [ON]  Wind vector re-encoding  (speed+dir -> u+v)
  [OFF] Temporal fire diff channels
  [OFF] Boundary-weighted focal loss

Results -> experiments/results/ablation_wind_results.json

Usage
-----
  python experiments/train_ablation_wind.py --epochs 20 --batch_size 16 --device cuda
"""

import os, sys, json, argparse, time, tempfile
from pathlib import Path

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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from models.SMPModel import SMPModel
from dataset_improved import ImprovedFireSpreadDataset
from data_utils import FOLDS, DATA_DIR, make_dataset, make_loader, compute_pos_class_weight

torch.set_float32_matmul_precision("high")

N_OBS      = 5
C_PER_STEP = 40   # no fire-diff, so still 40 channels/step
N_CHANNELS = N_OBS * C_PER_STEP  # 200


# ---------------------------------------------------------------------------
# Dataset: wind u/v only — inherits ImprovedFireSpreadDataset but skips
# the _add_temporal_fire_diff step -> outputs [T, 40, H, W]
# ---------------------------------------------------------------------------

class WindOnlyDataset(ImprovedFireSpreadDataset):
    def preprocess_and_augment(self, x, y):
        x, y = torch.Tensor(x), torch.Tensor(y)
        y = (y > 0).long()

        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        if self.is_pad:
            x, y = self.zero_pad_to_size(x, y)

        x = self._convert_wind_to_uv(x)
        x[:, [13], ...] = torch.sin(torch.deg2rad(x[:, [13], ...]))
        binary_af_mask  = (x[:, -1:, ...] > 0).float()
        x = self.standardize_features(x)
        x = torch.cat([x, binary_af_mask], dim=1)
        x = torch.nan_to_num(x, nan=0.0)

        T, _, H, W = x.shape
        lc_flat   = x[:, 16, ...].long().flatten() - 1
        lc_onehot = (
            self.one_hot_matrix[lc_flat]
            .reshape(T, H, W, self.one_hot_matrix.shape[0])
            .permute(0, 3, 1, 2)
        )
        x = torch.concatenate([x[:, :16, ...], lc_onehot, x[:, 17:, ...]], dim=1)
        # fire-diff step intentionally omitted -> [T, 40, H, W]
        return x, y


class AblationWindModel(SMPModel):
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)


def train_one_fold(fold_id, fold, data_root, epochs, batch_size, num_workers, device, results_dir):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_id} | T={N_OBS} | train={fold['train_years']} val={fold['val_years']} test={fold['test_years']}")
    print(f"  Ablation: wind-uv [ON]  fire-diff [OFF]  boundary-loss [OFF]")
    print(f"{'='*60}")

    pl.seed_everything(42, workers=True)

    t0 = time.time()
    train_ds = make_dataset(fold["train"], data_root, N_OBS, fold["stats_years"],
                            is_train=True,  DatasetClass=WindOnlyDataset)
    val_ds   = make_dataset(fold["val"],   data_root, N_OBS, fold["stats_years"],
                            is_train=False, DatasetClass=WindOnlyDataset)
    test_ds  = make_dataset(fold["test"],  data_root, N_OBS, fold["stats_years"],
                            is_train=False, DatasetClass=WindOnlyDataset)
    print(f"  Dataset sizes  train:{len(train_ds):,}  val:{len(val_ds):,}  test:{len(test_ds):,}  ({time.time()-t0:.1f}s)")

    train_dl = make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = make_loader(test_ds,  1,           shuffle=False, num_workers=num_workers)

    pos_weight = compute_pos_class_weight(fold["stats_years"])
    model = AblationWindModel(
        encoder_name="resnet18", encoder_weights="imagenet",
        n_channels=N_CHANNELS, flatten_temporal_dimension=True,
        pos_class_weight=pos_weight, loss_function="Focal",
    )
    print(f"  Model: Res18-UNet (wind-uv only) | n_channels={N_CHANNELS}")

    run_name = f"ablation_wind_T{N_OBS}_fold{fold_id}"
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
    test_results = trainer.test(model, test_dl, ckpt_path=ckpt_cb.best_model_path, verbose=True)

    test_ap = float(test_results[0].get("test_AP", 0.0))
    test_f1 = float(test_results[0].get("test_f1", 0.0))
    val_ap  = float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else 0.0

    result = {
        "fold": fold_id, "n_obs": N_OBS,
        "train_years": fold["train_years"], "val_years": fold["val_years"],
        "test_years":  fold["test_years"], "ablation": "wind_uv_only",
        "best_val_AP": round(val_ap, 4), "test_AP": round(test_ap, 4), "test_F1": round(test_f1, 4),
    }
    print(f"\n  [OK] Fold {fold_id} | T={N_OBS}  ->  val AP={val_ap:.4f}  test AP={test_ap:.4f}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default=DATA_DIR)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device",      default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = args.data_dir if os.path.isabs(args.data_dir) else str(_ROOT / args.data_dir)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset not found: {data_root}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    results_dir = _ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}\n#  ABLATION: wind u/v only  T={N_OBS}\n{'#'*60}")

    all_results = []
    for fold_id, fold in enumerate(FOLDS):
        all_results.append(train_one_fold(fold_id, fold, data_root, args.epochs,
                                          args.batch_size, args.num_workers, device, results_dir))

    aps = [r["test_AP"] for r in all_results]
    print(f"\n  -- ABLATION wind-uv Summary --  Mean AP = {sum(aps)/len(aps):.4f}")
    out = results_dir / "ablation_wind_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
