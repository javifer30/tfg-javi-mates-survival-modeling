"""Centralized survival metrics for the static TFG pipeline."""

import logging

import numpy as np
import pandas as pd

from src.evaluation.time_dependent_survival import censoring_km

logger = logging.getLogger(__name__)

DEFAULT_EVALUATION_TIME_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def calculate_concordance_index(y_time, y_event, risk_scores):
    """
    Calculate Harrell's C-index.

    risk_scores must use the survival-model convention: higher means higher risk.
    """
    try:
        from lifelines.utils import concordance_index
    except ImportError:
        return _manual_concordance_index(y_time, y_event, risk_scores)

    return float(concordance_index(y_time, -np.asarray(risk_scores), y_event))


def _manual_concordance_index(y_time, y_event, risk_scores):
    time = np.asarray(y_time, dtype=float)
    event = np.asarray(y_event, dtype=int)
    risk = np.asarray(risk_scores, dtype=float)
    concordant = 0.0
    comparable = 0.0
    for i in range(len(time)):
        if event[i] != 1:
            continue
        mask = time[i] < time
        comparable += float(mask.sum())
        concordant += float((risk[i] > risk[mask]).sum())
        concordant += 0.5 * float((risk[i] == risk[mask]).sum())
    return float(concordant / comparable) if comparable else np.nan


def _as_survival_dataframe(survival, duration_index=None):
    if survival is None:
        return None
    if isinstance(survival, pd.DataFrame):
        return survival
    array = np.asarray(survival)
    if array.ndim != 2:
        raise ValueError("Survival predictions must be a 2D array or DataFrame")
    if duration_index is None:
        duration_index = np.arange(array.shape[0])
    return pd.DataFrame(array, index=duration_index)


def _censoring_probability_before_or_at(time_value, timeline, survival):
    if len(timeline) == 0:
        return 1.0
    idx = int(np.searchsorted(timeline, float(time_value), side="left"))
    if idx >= len(survival):
        idx = len(survival) - 1
    return float(max(survival[idx], 1e-8))


def _survival_at_eval_times(survival, eval_times):
    surv_df = survival.sort_index()
    eval_index = pd.Index(np.asarray(eval_times, dtype=float))
    expanded = surv_df.reindex(surv_df.index.union(eval_index).sort_values()).ffill().bfill()
    return expanded.loc[eval_index].to_numpy(dtype=float).T


def calculate_survival_curve_metrics(
    y_time,
    y_event,
    survival,
    time_grid=None,
    evaluation_time_grid=None,
    censoring_time=None,
    censoring_event=None,
    compute_curve_metrics=True,
):
    """
    Calculate curve-based metrics when pycox is available.

    Returns NaN for unavailable metrics instead of failing the whole evaluation,
    because not every static model produces a full survival curve robustly.
    """
    surv_df = _as_survival_dataframe(survival, time_grid)
    if surv_df is None or not compute_curve_metrics:
        return {
            "ibs": np.nan,
            "ibll": np.nan,
            "nbll": np.nan,
        }

    durations = np.asarray(y_time, dtype=float)
    events = np.asarray(y_event, dtype=int)
    eval_times = np.asarray(evaluation_time_grid or DEFAULT_EVALUATION_TIME_GRID, dtype=float)
    eval_times = eval_times[(eval_times > durations.min()) & (eval_times < durations.max())]
    if len(eval_times) < 2:
        return {
            "ibs": np.nan,
            "ibll": np.nan,
            "nbll": np.nan,
        }

    censor_time = durations if censoring_time is None else np.asarray(censoring_time, dtype=float)
    censor_event = events if censoring_event is None else np.asarray(censoring_event, dtype=int)
    censor_timeline, censor_survival = censoring_km(censor_time, censor_event)
    survival_values = _survival_at_eval_times(surv_df, eval_times)
    survival_values = np.clip(survival_values, 1e-8, 1.0 - 1e-8)

    brier_scores = []
    nbll_scores = []
    for col_idx, eval_time in enumerate(eval_times):
        predicted_survival = survival_values[:, col_idx]
        survived = durations > eval_time
        event_before_or_at = (durations <= eval_time) & (events == 1)

        weights = np.zeros_like(durations, dtype=float)
        weights[survived] = 1.0 / _censoring_probability_before_or_at(eval_time, censor_timeline, censor_survival)
        event_indices = np.where(event_before_or_at)[0]
        for idx in event_indices:
            weights[idx] = 1.0 / _censoring_probability_before_or_at(durations[idx], censor_timeline, censor_survival)

        outcome = survived.astype(float)
        brier_scores.append(float(np.mean(weights * (outcome - predicted_survival) ** 2)))
        nbll_scores.append(
            float(
                -np.mean(
                    weights
                    * (
                        outcome * np.log(predicted_survival)
                        + (1.0 - outcome) * np.log(1.0 - predicted_survival)
                    )
                )
            )
        )

    return {
        "ibs": float(np.mean(brier_scores)),
        "ibll": float(np.mean(nbll_scores)),
        "nbll": float(np.mean(nbll_scores)),
    }


def evaluate_risk_predictions(y_time, y_event, risk_scores, metric_name="harrell_c_index"):
    return {metric_name: calculate_concordance_index(y_time, y_event, risk_scores)}


def evaluate_predictions(
    y_time,
    y_event,
    risk_scores=None,
    survival=None,
    time_grid=None,
    risk_metric_name="harrell_c_index",
    evaluation_time_grid=None,
    censoring_time=None,
    censoring_event=None,
    compute_curve_metrics=True,
):
    metrics = {}
    if risk_scores is not None:
        metrics.update(evaluate_risk_predictions(y_time, y_event, risk_scores, metric_name=risk_metric_name))
    metrics.update(
        calculate_survival_curve_metrics(
            y_time,
            y_event,
            survival,
            time_grid=time_grid,
            evaluation_time_grid=evaluation_time_grid,
            censoring_time=censoring_time,
            censoring_event=censoring_event,
            compute_curve_metrics=compute_curve_metrics,
        )
    )
    return metrics
