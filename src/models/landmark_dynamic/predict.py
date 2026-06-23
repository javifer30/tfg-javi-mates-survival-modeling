"""Prediction conversion helpers for dynamic_landmark models."""

from __future__ import annotations

import numpy as np
import pandas as pd


DAILY_HORIZONS = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="float32")


def survival_df_from_array(survival: np.ndarray) -> pd.DataFrame:
    survival = np.asarray(survival, dtype="float32")
    survival = np.clip(survival, 0.0, 1.0)
    return pd.DataFrame(survival.T, index=DAILY_HORIZONS[: survival.shape[1]])

