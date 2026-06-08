"""Time-dependent metrics for models that output survival curves."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.static_dataset import EVENT_COL, SPLIT_COL, TIME_COL
from src.models.static_common import cap_survival_targets


@dataclass
class TimeDependentPredictions:
    frame: pd.DataFrame
    cumulative_incidence: np.ndarray
    time_grid: np.ndarray


def survival_to_cumulative_incidence(survival):
    """Convert a survival curve DataFrame into F(t) = 1 - S(t)."""
    return 1.0 - survival.to_numpy(dtype=float).T


def cap_prediction_targets(frame, max_horizon_days):
    time, event = cap_survival_targets(frame[TIME_COL], frame[EVENT_COL], max_horizon_days)
    return time.to_numpy(dtype=float), event.to_numpy(dtype=int)


def time_to_index(time, time_grid):
    time_grid = np.asarray(time_grid, dtype=float)
    return int(np.clip(np.searchsorted(time_grid, float(time), side="left"), 0, len(time_grid) - 1))


def _concordance_from_risk(time, event, risk):
    numerator = 0.0
    denominator = 0.0
    comparable_events = np.where(event == 1)[0]
    for i in comparable_events:
        comparable = time[i] < time
        if not np.any(comparable):
            continue
        denominator += float(comparable.sum())
        numerator += float(np.sum(risk[i] > risk[comparable]))
    return numerator, denominator


def antolini_ctd(time, event, cumulative_incidence, time_grid):
    """
    Approximate Antolini's Ctd using event-time-specific cumulative incidence.

    For each observed event i, risks for i and j are both evaluated at i's
    event time. Ties receive no credit, matching the indicator-style metric.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    cumulative_incidence = np.asarray(cumulative_incidence, dtype=float)
    event_bins = np.asarray([time_to_index(value, time_grid) for value in time], dtype=int)

    numerator = 0.0
    denominator = 0.0
    comparable_events = np.where(event == 1)[0]
    for i in comparable_events:
        comparable = time[i] < time
        if not np.any(comparable):
            continue
        risk_at_i_time = cumulative_incidence[:, event_bins[i]]
        denominator += float(comparable.sum())
        numerator += float(np.sum(risk_at_i_time[i] > risk_at_i_time[comparable]))
    return {
        "ctd": float(numerator / denominator) if denominator else np.nan,
        "concordant_pairs": float(numerator),
        "comparable_pairs": float(denominator),
    }


def censoring_km(time, event):
    """Kaplan-Meier estimate of the censoring survival curve G(t)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time)
    sorted_time = time[order]
    sorted_event = event[order]
    unique_times = np.unique(sorted_time)

    survival = 1.0
    timeline = []
    survival_values = []
    n = len(sorted_time)
    start = 0
    for current_time in unique_times:
        while start < n and sorted_time[start] < current_time:
            start += 1
        at_risk = n - start
        at_time = sorted_time == current_time
        censoring_events = int(np.sum((sorted_event == 0) & at_time))
        if at_risk > 0:
            survival *= 1.0 - censoring_events / at_risk
        timeline.append(float(current_time))
        survival_values.append(float(survival))

    values = np.asarray(survival_values, dtype=float)
    non_zero = values[values > 0]
    if len(non_zero):
        values[values == 0] = non_zero[-1]
    return np.asarray(timeline, dtype=float), values


def _censoring_probability_at(time_value, timeline, survival):
    idx = int(np.searchsorted(timeline, float(time_value), side="left"))
    if idx >= len(survival):
        idx = len(survival) - 1
    return float(max(survival[idx], 1e-8))


def weighted_c_index_at_horizon(train_time, train_event, test_time, test_event, risk, horizon):
    """
    Reference-code style weighted C-index at a fixed horizon.

    The censoring distribution is estimated on the train split. Ties receive no
    credit, matching the original DeepHit implementation.
    """
    timeline, censor_survival = censoring_km(train_time, train_event)
    numerator = 0.0
    denominator = 0.0
    eligible = np.where((test_time <= float(horizon)) & (test_event == 1))[0]
    for i in eligible:
        comparable = test_time[i] < test_time
        if not np.any(comparable):
            continue
        censor_prob = _censoring_probability_at(test_time[i], timeline, censor_survival)
        weight = (1.0 / censor_prob) ** 2
        denominator += float(comparable.sum()) * weight
        numerator += float(np.sum(risk[i] > risk[comparable])) * weight
    return {
        "horizon_c_index": float(numerator / denominator) if denominator else np.nan,
        "weighted_c_index": float(numerator / denominator) if denominator else np.nan,
        "weighted_concordant_pairs": float(numerator),
        "weighted_comparable_pairs": float(denominator),
    }


def mean_horizon_c_index(weighted_rows):
    values = [row.get("horizon_c_index", row.get("weighted_c_index")) for row in weighted_rows]
    values = [float(value) for value in values if value is not None and not np.isnan(float(value))]
    return float(np.mean(values)) if values else np.nan


def horizon_c_index_dict(weighted_rows):
    return {
        str(int(row["horizon_days"]) if float(row["horizon_days"]).is_integer() else row["horizon_days"]): row.get(
            "horizon_c_index", row.get("weighted_c_index")
        )
        for row in weighted_rows
    }


def survival_time_dependent_metrics(split_name, time, event, survival, train_time, train_event, horizon_times):
    time_grid = survival.index.to_numpy(dtype=float)
    cumulative_incidence = survival_to_cumulative_incidence(survival)
    antolini = {"split": split_name, **antolini_ctd(time, event, cumulative_incidence, time_grid)}
    weighted_rows = []
    for horizon in horizon_times:
        bin_idx = time_to_index(horizon, time_grid)
        risk = cumulative_incidence[:, bin_idx]
        result = weighted_c_index_at_horizon(train_time, train_event, time, event, risk, horizon)
        weighted_rows.append(
            {
                "split": split_name,
                "horizon_days": float(horizon),
                "horizon_bin": int(bin_idx + 1),
                **result,
            }
        )
    return antolini, weighted_rows


def calculate_antolini_by_split(predictions, max_horizon_days):
    rows = []
    for split_name, split_frame in predictions.frame.groupby(SPLIT_COL, sort=False):
        idx = split_frame.index.to_numpy()
        time, event = cap_prediction_targets(split_frame, max_horizon_days)
        result = antolini_ctd(time, event, predictions.cumulative_incidence[idx], predictions.time_grid)
        rows.append({"split": split_name, **result})
    return pd.DataFrame(rows)


def calculate_weighted_c_index_by_horizon(predictions, eval_times_days, max_horizon_days):
    train_frame = predictions.frame[predictions.frame[SPLIT_COL] == "train"]
    train_time, train_event = cap_prediction_targets(train_frame, max_horizon_days)

    rows = []
    for split_name, split_frame in predictions.frame.groupby(SPLIT_COL, sort=False):
        idx = split_frame.index.to_numpy()
        time, event = cap_prediction_targets(split_frame, max_horizon_days)
        for horizon in eval_times_days:
            bin_idx = time_to_index(horizon, predictions.time_grid)
            risk = predictions.cumulative_incidence[idx, bin_idx]
            result = weighted_c_index_at_horizon(train_time, train_event, time, event, risk, horizon)
            rows.append(
                {
                    "split": split_name,
                    "horizon_days": float(horizon),
                    "horizon_bin": int(bin_idx + 1),
                    **result,
                }
            )
    return pd.DataFrame(rows)
