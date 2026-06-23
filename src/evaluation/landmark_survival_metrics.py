"""Evaluation utilities for the static_landmark_pycox experiment."""

import math

import numpy as np
import pandas as pd


def survival_at_times(surv_df, times):
    surv = surv_df.sort_index()
    times = np.asarray(times, dtype=float)
    eval_index = pd.Index(times)
    expanded = surv.reindex(surv.index.union(eval_index).sort_values()).ffill().bfill()
    return expanded.loc[eval_index].to_numpy(dtype=float).T


def metric_integration_grid(surv_df, durations, max_horizon_days, num_points=100):
    durations = np.asarray(durations, dtype=float)
    surv_index = np.asarray(surv_df.index, dtype=float)
    finite_durations = durations[np.isfinite(durations)]
    finite_index = surv_index[np.isfinite(surv_index)]
    if finite_durations.size == 0 or finite_index.size == 0:
        return np.asarray([], dtype=float)

    lower = max(float(finite_durations.min()), float(finite_index.min()))
    upper = min(float(finite_durations.max()), float(finite_index.max()), float(max_horizon_days))
    if not upper > lower:
        upper = min(float(finite_index.max()), float(max_horizon_days))
        lower = min(lower, upper)
    if upper <= lower:
        return np.asarray([upper], dtype=float)
    return np.linspace(lower, upper, int(num_points), dtype=float)


def eval_surv_metrics(surv_df, durations, events, time_grid):
    from pycox.evaluation import EvalSurv

    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    time_grid = np.asarray(time_grid, dtype=float)
    ev = EvalSurv(surv_df, durations, events, censor_surv="km")
    metrics = {
        "ctd_antolini": float(ev.concordance_td("antolini")),
        "ibs": math.nan,
        "ibll": math.nan,
        "nbll": math.nan,
    }
    try:
        metrics["ibs"] = float(ev.integrated_brier_score(time_grid))
    except Exception:
        metrics["ibs"] = math.nan
    try:
        nbll = float(ev.integrated_nbll(time_grid))
        metrics["ibll"] = nbll
        metrics["nbll"] = nbll
    except Exception:
        metrics["ibll"] = math.nan
        metrics["nbll"] = math.nan
    return metrics


def horizon_c_index(durations, events, risks, horizon_day):
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    horizon = float(horizon_day)
    concordant = 0.0
    comparable = 0
    event_indices = np.where((events == 1) & (durations <= horizon))[0]
    for i in event_indices:
        mask = durations[i] < durations
        n_pairs = int(mask.sum())
        if n_pairs == 0:
            continue
        comparable += n_pairs
        concordant += float((risks[i] > risks[mask]).sum())
        concordant += 0.5 * float((risks[i] == risks[mask]).sum())
    return {
        "c_index": float(concordant / comparable) if comparable else math.nan,
        "n_comparable_pairs": int(comparable),
    }


def horizon_c_index_rows(model_name, split_name, surv_df, durations, events, horizon_times):
    survival_values = survival_at_times(surv_df, horizon_times)
    rows = []
    for col_idx, horizon in enumerate(horizon_times):
        risks = 1.0 - survival_values[:, col_idx]
        result = horizon_c_index(durations, events, risks, horizon)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "horizon_day": float(horizon),
                **result,
            }
        )
    return rows


def mean_horizon_c_index(rows):
    values = [row["c_index"] for row in rows if not math.isnan(row["c_index"])]
    return float(np.mean(values)) if values else math.nan
