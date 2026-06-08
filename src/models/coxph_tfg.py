"""
TFG CoxPH wrapper.

Uses lifelines.CoxPHFitter instead of reimplementing the Cox partial likelihood.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.static_dataset import EVENT_COL, ID_COL, SPLIT_COL, TIME_COL
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.time_dependent_survival import horizon_c_index_dict, mean_horizon_c_index, survival_time_dependent_metrics
from src.models.static_common import (
    cap_survival_targets,
    configured_split_names,
    configured_split_frames,
    load_static_splits,
    make_time_grid,
    model_metrics_dir,
    save_json,
    save_model,
    should_save_predictions,
    should_save_test_survival_curves,
    split_xy,
)


def _default_grid(max_horizon_days):
    upper = int(np.floor(float(max_horizon_days)))
    if upper > 1:
        return list(range(1, upper))
    return [float(max_horizon_days)]


def _predict_split(
    model,
    df,
    split_name,
    time_grid,
    max_horizon_days,
    evaluation_time_grid,
    censoring_time,
    censoring_event,
):
    x, time, event, ids = split_xy(df)
    time, event = cap_survival_targets(time, event, max_horizon_days)
    risk = model.predict_partial_hazard(x).values.reshape(-1)
    survival = model.predict_survival_function(x, times=time_grid)
    predictions = ids.copy()
    predictions["risk_score"] = risk
    predictions["model"] = "coxph"
    predictions["split"] = split_name
    return (
        predictions,
        survival,
        evaluate_predictions(
            time,
            event,
            risk_scores=risk,
            survival=survival,
            risk_metric_name="harrell_c_index",
            evaluation_time_grid=evaluation_time_grid,
            censoring_time=censoring_time,
            censoring_event=censoring_event,
            compute_curve_metrics=split_name != "train",
        ),
        time.to_numpy(dtype=float),
        event.to_numpy(dtype=int),
    )


def train_coxph(config, logger):
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError("lifelines is required for CoxPH") from exc

    paths = config["paths"]
    model_cfg = config["model"]
    train, val, test = load_static_splits(paths, include_test="test" in configured_split_names(config))
    x_train, time_train, event_train, _ = split_xy(train)
    max_horizon_days = model_cfg.get("max_horizon_days", 10)
    eval_time_train, eval_event_train = cap_survival_targets(time_train, event_train, max_horizon_days)

    train_df = pd.concat(
        [x_train, pd.DataFrame({TIME_COL: time_train, EVENT_COL: event_train}, index=x_train.index)],
        axis=1,
    )
    model = CoxPHFitter(
        penalizer=model_cfg.get("penalizer", 0.01),
        l1_ratio=model_cfg.get("l1_ratio", 0.0),
    )
    model.fit(train_df, duration_col=TIME_COL, event_col=EVENT_COL)
    time_grid = make_time_grid(
        max_horizon_days,
        model_cfg.get("num_durations", 10),
    )

    eval_cfg = config.get("evaluation", {})
    evaluation_time_grid = eval_cfg.get("evaluation_time_grid", model_cfg.get("evaluation_time_grid", _default_grid(max_horizon_days)))
    horizon_times = eval_cfg.get("horizon_times", model_cfg.get("horizon_times", _default_grid(max_horizon_days)))
    metrics = {
        "model": "coxph",
        "best_val_loss": None,
        "evaluation_time_grid": evaluation_time_grid,
        "horizon_times": horizon_times,
        "splits": {},
    }
    save_predictions_flag = should_save_predictions(config)
    save_test_survival_flag = should_save_test_survival_curves(config)
    all_predictions = []
    antolini_rows = []
    weighted_rows = []
    for split_name, split_df in configured_split_frames(config, train, val, test).items():
        preds, survival, split_metrics, time_values, event_values = _predict_split(
            model,
            split_df,
            split_name,
            time_grid,
            max_horizon_days,
            evaluation_time_grid,
            eval_time_train.to_numpy(dtype=float),
            eval_event_train.to_numpy(dtype=int),
        )
        metrics["splits"][split_name] = split_metrics
        if save_predictions_flag:
            all_predictions.append(preds)
        antolini, weighted = survival_time_dependent_metrics(
            split_name,
            time_values,
            event_values,
            survival,
            eval_time_train.to_numpy(dtype=float),
            eval_event_train.to_numpy(dtype=int),
            horizon_times,
        )
        antolini_rows.append(antolini)
        weighted_rows.extend(weighted)
        metrics["splits"][split_name]["ctd_antolini"] = antolini["ctd"]
        metrics["splits"][split_name]["horizon_c_index"] = horizon_c_index_dict(weighted)
        metrics["splits"][split_name]["mean_horizon_c_index"] = mean_horizon_c_index(weighted)
        logger.info("CoxPH %s Harrell C-index: %.4f", split_name, split_metrics["harrell_c_index"])
        if split_name == "test" and save_test_survival_flag:
            survival.to_csv(Path(paths["predictions_dir"]) / "coxph_test_survival_curves.csv")

    if all_predictions:
        pred_path = Path(paths["predictions_dir"]) / "coxph_predictions.parquet"
        pd.concat(all_predictions, ignore_index=True).to_parquet(pred_path, index=False)

    metrics_dir = model_metrics_dir(paths, "coxph")
    coefficients = model.params_.rename("coef").to_frame()
    coefficients["hazard_ratio"] = model.hazard_ratios_
    coefficients.to_csv(metrics_dir / "coxph_coefficients.csv")
    pd.DataFrame(weighted_rows).to_csv(metrics_dir / "coxph_weighted_c_index_by_horizon.csv", index=False)
    pd.DataFrame(antolini_rows).to_csv(metrics_dir / "coxph_antolini_ctd.csv", index=False)

    if model_cfg.get("save_model", True):
        save_model(model, Path(paths["models_dir"]) / "coxph_model.pkl")
    save_json(metrics, metrics_dir / "coxph_metrics.json")
    return metrics
