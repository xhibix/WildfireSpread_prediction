"""
data_utils.py
=============
Shared dataset factory and fold definitions used by ALL training and
plotting scripts.  Handles the WildfireSpreadTS_10pct pre-split layout:

    WildfireSpreadTS_10pct/
        train/
            2018/  fire_*.hdf5   (17 fires)
            2019/  fire_*.hdf5   ( 7 fires)
        val/
            2020/  fire_*.hdf5   (20 fires)
        test/
            2021/  fire_*.hdf5   (14 fires)

3-fold cross-validation
-----------------------
We rotate 2018 / 2019 / 2020 between train and val while the test set
(2021) is always held out.  Each fold entry is a dict with:

    "train"       list of (split_subdir, year) pairs for training
    "val"         list of (split_subdir, year) pairs for validation
    "test"        list of (split_subdir, year) pairs for testing (fixed)
    "stats_years" integer years used to look up pre-computed mean/std stats

Fold 0 uses the canonical (intended) split of the pre-processed dataset.
Folds 1 and 2 rotate years for variance estimation.

Usage
-----
    from data_utils import FOLDS, make_dataset, compute_pos_class_weight, DATA_DIR
"""

import os
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "experiments"))

from torch.utils.data import DataLoader, ConcatDataset

from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

# ---------------------------------------------------------------------------
# Default dataset root
# ---------------------------------------------------------------------------
DATA_DIR = str(_ROOT / "WildfireSpreadTS_10pct")

# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------
# Each entry: (split_subfolder, integer_year)
# test is ALWAYS test/2021 — never changes across folds.
FOLDS = [
    {   # Fold 0 — canonical split (as intended by the dataset creators)
        "train":       [("train", 2018), ("train", 2019)],
        "val":         [("val",   2020)],
        "test":        [("test",  2021)],
        "stats_years": [2018, 2019],
        "train_years": [2018, 2019],   # kept for labelling only
        "val_years":   [2020],
        "test_years":  [2021],
    },
    {   # Fold 1 — 2020 rotated into training; 2019 becomes validation
        "train":       [("train", 2018), ("val",   2020)],
        "val":         [("train", 2019)],
        "test":        [("test",  2021)],
        "stats_years": [2018, 2020],
        "train_years": [2018, 2020],
        "val_years":   [2019],
        "test_years":  [2021],
    },
    {   # Fold 2 — 2019 + 2020 train; 2018 becomes validation
        "train":       [("train", 2019), ("val",   2020)],
        "val":         [("train", 2018)],
        "test":        [("test",  2021)],
        "stats_years": [2019, 2020],
        "train_years": [2019, 2020],
        "val_years":   [2018],
        "test_years":  [2021],
    },
]

# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def make_dataset(
    split_year_pairs,
    data_root,
    n_obs,
    stats_years,
    is_train,
    DatasetClass=None,
    crop=128,
    **extra_kwargs,
):
    """
    Build a (possibly concatenated) dataset from a list of (split, year) pairs.

    Parameters
    ----------
    split_year_pairs : list of (str, int)
        e.g. [("train", 2018), ("val", 2020)]
    data_root : str or Path
        Root of WildfireSpreadTS_10pct (contains train/, val/, test/).
    n_obs : int
        Number of leading observation days (1 or 5).
    stats_years : list of int
        Years used for mean/std normalisation lookup.
    is_train : bool
        True → apply random augmentation; False → deterministic centre-crop.
    DatasetClass : class, optional
        Defaults to FireSpreadDataset.  Pass ImprovedFireSpreadDataset etc.
    crop : int
        Spatial crop size (default 128).
    **extra_kwargs
        Forwarded verbatim to the DatasetClass constructor.

    Returns
    -------
    Dataset (single FireSpreadDataset or ConcatDataset of several).
    """
    if DatasetClass is None:
        DatasetClass = FireSpreadDataset

    data_root = Path(data_root)
    datasets = []
    for split_subdir, year in split_year_pairs:
        ds = DatasetClass(
            data_dir=str(data_root / split_subdir),
            included_fire_years=[year],
            n_leading_observations=n_obs,
            crop_side_length=crop,
            load_from_hdf5=True,
            is_train=is_train,
            remove_duplicate_features=False,
            stats_years=stats_years,
            n_leading_observations_test_adjustment=None,
            features_to_keep=None,
            return_doy=False,
            is_pad=False,
            **extra_kwargs,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        drop_last=False,
    )


def compute_pos_class_weight(stats_years) -> float:
    """
    Returns the pre-transformed pos_class_weight for focal loss.
    raw = 1 / fire_rate  (≈ 964 for this dataset)
    returned value = raw / (1 + raw)  ≈ 0.999  (in [0,1] so BaseModel skips its own transform)
    """
    _, _, missing = get_means_stds_missing_values(stats_years)
    raw = float(1.0 / (1.0 - missing[-1]))
    return raw / (1.0 + raw)
