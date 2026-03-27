"""
ImprovedFireSpreadDataset
=========================
Extends FireSpreadDataset with two data-level improvements:

  1. Wind vector re-encoding
     Raw channels 6 (obs speed) + 7 (obs direction) → u_obs + v_obs
     Raw channels 18 (frc speed) + 19 (frc direction) → u_frc + v_frc
     Conversion: u = -speed * sin(dir_rad), v = -speed * cos(dir_rad)
     (meteorological convention: direction = where wind blows FROM)

  2. Temporal fire difference channels
     For each of the T input timesteps, append one channel:
       diff[t] = binary_fire[t] - binary_fire[t-1]   (values in {-1, 0, +1})
       diff[0] = zeros (no previous day available)
     Result: each timestep goes from 40 → 41 channels.
     Flattened for UNet: T*41 total (e.g. 5*41=205).

The third improvement (boundary-weighted focal loss) lives in the model,
not the dataset — see train_improved.py / ImprovedSMPModel.

Channel layout after preprocessing (same numbering as parent, 40 channels):
  0-15   : dynamic obs features  (includes ch6=u_obs, ch7=v_obs after conversion)
  16-32  : landcover one-hot (17 classes)
  33-38  : forecast features    (includes ch34=u_frc, ch35=v_frc after conversion)
  39     : binary active-fire mask
  40     : temporal fire difference  ← new channel added by this class

Usage:
  dataset = ImprovedFireSpreadDataset(
      data_dir="wildfire_10pct",
      included_fire_years=[2018, 2019],
      n_leading_observations=5,
      crop_side_length=128,
      load_from_hdf5=True,
      is_train=True,
      remove_duplicate_features=False,
      stats_years=[2018, 2019],
  )
"""

import os
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Make sure src/ is importable when running from experiments/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch
from dataloader.FireSpreadDataset import FireSpreadDataset


class ImprovedFireSpreadDataset(FireSpreadDataset):
    """
    FireSpreadDataset with wind u/v encoding and temporal fire diff channels.

    Parameters are identical to FireSpreadDataset; no new init args needed.
    The improvements are applied transparently inside preprocess_and_augment().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store obs/forecast wind speed stds BEFORE we modify self.stds.
        # These are used to normalise the new u/v channels, which have ~0 mean
        # and roughly the same scale as speed (since |u|,|v| ≤ speed).
        # self.stds has shape [1, 23, 1, 1] at this point.
        self._obs_wind_std = float(self.stds[0, 6, 0, 0])   # GRIDMET wind speed std
        self._frc_wind_std = float(self.stds[0, 18, 0, 0])  # GFS forecast wind speed std

        # ------------------------------------------------------------------ #
        # Adjust normalisation stats for the four channels affected by u/v    #
        # conversion.  The degree channels (7, 19) originally have mean=0,    #
        # std=1 (no normalisation applied, because sin() was used instead).   #
        # After conversion they become v-components and should use the speed   #
        # std.  The speed channels (6, 18) keep their std but their means     #
        # change to 0 (u/v components are centred around 0, unlike speed).    #
        # ------------------------------------------------------------------ #
        # ch6  = u_obs : mean → 0, std stays obs_wind_std
        self.means[0, 6, 0, 0] = 0.0
        # ch7  = v_obs : mean = 0 (already), std → obs_wind_std (was 1)
        self.stds[0, 7, 0, 0] = self._obs_wind_std
        # ch18 = u_frc : mean → 0, std stays frc_wind_std
        self.means[0, 18, 0, 0] = 0.0
        # ch19 = v_frc : mean = 0 (already), std → frc_wind_std (was 1)
        self.stds[0, 19, 0, 0] = self._frc_wind_std

        # NOTE: We intentionally do NOT override self.indices_of_degree_features
        # here.  That list ([7, 13, 19]) is used inside augment() to correctly
        # rotate/flip the wind *direction* angles — which are still in degree
        # space during augmentation.  We convert to u/v AFTER augmentation.

    # ---------------------------------------------------------------------- #
    #  Public override                                                         #
    # ---------------------------------------------------------------------- #

    def preprocess_and_augment(self, x, y):
        """
        Drop-in replacement for FireSpreadDataset.preprocess_and_augment().

        Processing order
        ----------------
        1. Convert to tensors, binarise y.
        2. Geometric augmentation  (uses self.indices_of_degree_features=[7,13,19]
           to correctly adjust wind-direction and aspect angles).
        3. Wind u/v conversion  (channels 6,7 and 18,19 in raw 23-ch space).
        4. Apply sin() to aspect only  (ch 13) — NOT to 7 or 19, which are now
           v-components, not degrees.
        5. Compute binary fire mask before normalisation.
        6. Normalise (standardise) with the modified means/stds set in __init__.
        7. Append binary fire mask as an extra channel  → 24 ch per timestep.
        8. Replace NaNs with 0 (= mean after standardisation).
        9. One-hot encode land cover (ch 16 → 17 binary channels) → 40 ch/step.
       10. Append temporal fire-difference channel  → 41 ch/step.
        """
        x, y = torch.Tensor(x), torch.Tensor(y)

        # Preprocessing already done in HDF5 (skip for TIF path, handled by parent).
        # (Active fire NaN→0 and hhmm→hh are in HDF5 already.)

        y = (y > 0).long()

        # Step 2 — geometric augmentation / center-crop
        # self.indices_of_degree_features = [7, 13, 19] from parent; channels
        # 7 and 19 are still raw degrees here, so the angle adjustments are correct.
        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        if self.is_pad:
            x, y = self.zero_pad_to_size(x, y)

        # Step 3 — wind u/v conversion
        x = self._convert_wind_to_uv(x)

        # Step 4 — sin() to remaining degree feature: aspect (ch 13 only)
        x[:, [13], ...] = torch.sin(torch.deg2rad(x[:, [13], ...]))

        # Step 5 — binary active-fire mask (ch 22 = last raw channel)
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        # Step 6 — standardise (modified means/stds already set in __init__)
        x = self.standardize_features(x)

        # Step 7 — append binary fire mask  →  [T, 24, H, W]
        x = torch.cat([x, binary_af_mask], dim=1)

        # Step 8 — NaN → 0
        x = torch.nan_to_num(x, nan=0.0)

        # Step 9 — one-hot land cover  →  [T, 40, H, W]
        T, _, H, W = x.shape
        new_shape = (T, H, W, self.one_hot_matrix.shape[0])
        lc_flat = x[:, 16, ...].long().flatten() - 1  # classes are 1-based
        lc_onehot = (
            self.one_hot_matrix[lc_flat]
            .reshape(new_shape)
            .permute(0, 3, 1, 2)
        )
        x = torch.concatenate(
            [x[:, :16, ...], lc_onehot, x[:, 17:, ...]], dim=1
        )
        # x: [T, 40, H, W]

        # Step 10 — temporal fire difference  →  [T, 41, H, W]
        x = self._add_temporal_fire_diff(x)

        return x, y

    # ---------------------------------------------------------------------- #
    #  Static helper: compute n_features (for model instantiation)            #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def get_n_features(n_observations, features_to_keep, deduplicate_static_features):
        """
        Same as FireSpreadDataset.get_n_features() but adds n_observations
        extra channels (one temporal fire-diff channel per timestep).
        """
        base = FireSpreadDataset.get_n_features(
            n_observations, features_to_keep, deduplicate_static_features
        )
        # +1 fire-diff channel per timestep.
        # When deduplicate=False the formula returns C_per_step, not T*C_per_step,
        # so we add 1 (per-step), not n_observations.
        # When deduplicate=True the formula returns flattened total, but we
        # don't support dedup in this improved dataset (always use dedup=False).
        return base + 1

    # ---------------------------------------------------------------------- #
    #  Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _convert_wind_to_uv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (speed, direction) → (u, v) for observed and forecast wind.

        Operates on the raw 23-channel tensor AFTER augmentation but BEFORE
        standardisation.  Channel indices (raw 23-ch space):
          6  = GRIDMET obs wind speed   →  becomes u_obs
          7  = GRIDMET obs wind dir     →  becomes v_obs
          18 = GFS forecast wind speed  →  becomes u_frc
          19 = GFS forecast wind dir    →  becomes v_frc

        Meteorological convention (dir = where wind blows FROM):
          u = -speed * sin(dir_rad)   [positive = eastward]
          v = -speed * cos(dir_rad)   [positive = northward]
        """
        # Observed wind
        speed_obs = x[:, 6, ...].clone()
        dir_rad_obs = torch.deg2rad(x[:, 7, ...])
        x[:, 6, ...] = -speed_obs * torch.sin(dir_rad_obs)   # u_obs
        x[:, 7, ...] = -speed_obs * torch.cos(dir_rad_obs)   # v_obs

        # Forecast wind
        speed_frc = x[:, 18, ...].clone()
        dir_rad_frc = torch.deg2rad(x[:, 19, ...])
        x[:, 18, ...] = -speed_frc * torch.sin(dir_rad_frc)  # u_frc
        x[:, 19, ...] = -speed_frc * torch.cos(dir_rad_frc)  # v_frc

        return x

    def _add_temporal_fire_diff(self, x: torch.Tensor) -> torch.Tensor:
        """
        Append a temporal fire-difference channel to each timestep.

        x shape in:  [T, 40, H, W]
        x shape out: [T, 41, H, W]

        Channel 39 of the preprocessed tensor is the binary active-fire mask.
        diff[t] = fire[t] - fire[t-1]  ∈ {-1, 0, +1}
        diff[0] = 0  (no previous day)
        """
        T, C, H, W = x.shape
        FIRE_IDX = 39  # binary fire mask is always the last of the 40 standard channels

        diffs = []
        for t in range(T):
            if t == 0:
                diff = torch.zeros(1, H, W, dtype=x.dtype, device=x.device)
            else:
                diff = (x[t, FIRE_IDX] - x[t - 1, FIRE_IDX]).unsqueeze(0)
            diffs.append(diff)

        diffs = torch.stack(diffs, dim=0)   # [T, 1, H, W]
        return torch.cat([x, diffs], dim=1)  # [T, 41, H, W]
