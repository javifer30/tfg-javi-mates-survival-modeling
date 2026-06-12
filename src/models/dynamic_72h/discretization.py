"""Daily target discretization for the dynamic_72h experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd


DAILY_CUTS = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="float32")


def discretize_duration_event(durations, events, cuts=DAILY_CUTS):
    """Map durations in days to interval indices 0..9.

    Intervals are (0,1], (1,2], ..., (9,10]. Durations at 10 days censored by
    administrative horizon map to idx 9 with event 0.
    """
    durations = np.asarray(durations, dtype="float32")
    events = np.asarray(events, dtype="int64")
    if np.any(~np.isfinite(durations)):
        raise ValueError("Non-finite duration in dynamic_72h targets")
    if np.any(durations < 0) or np.any(durations > float(cuts[-1])):
        raise ValueError("duration_eval_days must be inside [0, 10]")
    if not set(np.unique(events)).issubset({0, 1}):
        raise ValueError("event_eval must be binary")
    idx = np.searchsorted(cuts[1:], durations, side="left").astype("int64")
    idx = np.clip(idx, 0, len(cuts) - 2)
    return idx, events.astype("int64")


def target_summary(split_name: str, durations, events, t_idx) -> pd.DataFrame:
    rows = []
    for idx in range(len(DAILY_CUTS) - 1):
        mask = np.asarray(t_idx) == idx
        rows.append(
            {
                "split": split_name,
                "t_idx": idx,
                "interval_left": float(DAILY_CUTS[idx]),
                "interval_right": float(DAILY_CUTS[idx + 1]),
                "n": int(mask.sum()),
                "n_events": int(np.asarray(events)[mask].sum()),
                "n_censored": int(mask.sum() - np.asarray(events)[mask].sum()),
            }
        )
    return pd.DataFrame(rows)

