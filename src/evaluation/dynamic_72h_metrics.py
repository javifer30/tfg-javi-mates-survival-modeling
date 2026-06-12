"""Evaluation helpers for dynamic_72h models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.static_72h_metrics import (
    eval_surv_metrics,
    horizon_c_index_rows,
    mean_horizon_c_index,
    metric_integration_grid,
    survival_at_times,
)


def prediction_sanity_rows(model_name: str, split_name: str, surv_df: pd.DataFrame) -> dict:
    values = surv_df.to_numpy(dtype=float)
    diffs = np.diff(values, axis=0)
    s10 = survival_at_times(surv_df, [10.0])[:, 0]
    return {
        "model": model_name,
        "split": split_name,
        "n_patients": int(values.shape[1]),
        "n_times": int(values.shape[0]),
        "has_nan": bool(np.isnan(values).any()),
        "min_survival": float(np.nanmin(values)),
        "max_survival": float(np.nanmax(values)),
        "monotone_non_increasing": bool(np.nanmax(diffs) <= 1e-6),
        "share_s10_below_1e-6": float(np.mean(s10 < 1e-6)),
        "mean_s10": float(np.mean(s10)),
    }


def evaluate_survival_predictions(model_name, split_surv, split_targets, config, metrics_dir, audit_dir, predictions_dir):
    horizon_times = config["evaluation"]["horizon_times"]
    max_horizon = config["experiment"]["max_horizon_days"]
    metrics = {
        "model": model_name,
        "horizon_times": horizon_times,
        "splits": {},
    }
    horizon_rows = []
    sanity_rows = []
    for split_name, surv_df in split_surv.items():
        durations, events = split_targets[split_name]
        grid = metric_integration_grid(
            surv_df,
            durations,
            max_horizon,
            config["evaluation"].get("metric_integration_num_points", 100),
        )
        split_metrics = eval_surv_metrics(surv_df, durations, events, grid)
        rows = horizon_c_index_rows(model_name, split_name, surv_df, durations, events, horizon_times)
        split_metrics["horizon_c_index"] = {str(int(row["horizon_day"])): row["c_index"] for row in rows}
        split_metrics["mean_horizon_c_index"] = mean_horizon_c_index(rows)
        split_metrics["integration_grid_min"] = float(grid.min()) if grid.size else float("nan")
        split_metrics["integration_grid_max"] = float(grid.max()) if grid.size else float("nan")
        metrics["splits"][split_name] = split_metrics
        horizon_rows.extend(rows)
        sanity_rows.append(prediction_sanity_rows(model_name, split_name, surv_df))

    metrics_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(horizon_rows).to_csv(metrics_dir / "horizon_c_index.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(audit_dir / f"{model_name}_prediction_sanity.csv", index=False)

    if config.get("evaluation", {}).get("save_example_curves", True) and "test" in split_surv:
        _save_example_curves(model_name, split_surv["test"], split_targets["test"], config, predictions_dir)
    return metrics


def _save_example_curves(model_name, surv_df, target, config, predictions_dir):
    n = min(int(config["evaluation"].get("n_example_patients", 9)), surv_df.shape[1])
    if n <= 0:
        return
    durations, events = target
    risk = 1.0 - survival_at_times(surv_df, [float(config["experiment"]["max_horizon_days"])])[:, 0]
    order = np.argsort(risk)
    selected = sorted(set(order[: max(1, n // 3)].tolist() + order[-max(1, n // 3) :].tolist()))
    selected = selected[:n]
    curves = surv_df.iloc[:, selected].copy()
    curves.insert(0, "time_days", surv_df.index.to_numpy(dtype=float))
    curves.to_csv(predictions_dir / "survival_curve_examples.csv", index=False)
    pd.DataFrame(
        {
            "column_index": selected,
            "duration_eval_days": durations[selected],
            "event_eval": events[selected],
            "risk_at_10d": risk[selected],
        }
    ).to_csv(predictions_dir / f"example_survival_selection_{model_name}.csv", index=False)

