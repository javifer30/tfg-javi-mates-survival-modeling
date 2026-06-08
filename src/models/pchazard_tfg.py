"""
TFG PCHazard implementation.

This replaces the older PWE Poisson baseline in the static pipeline and uses
pycox.models.PCHazard with torchtuples for the MLP.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.time_dependent_survival import (
    antolini_ctd,
    horizon_c_index_dict,
    mean_horizon_c_index,
    survival_to_cumulative_incidence,
    time_to_index,
    weighted_c_index_at_horizon,
)
from src.models.static_common import cap_survival_targets, get_device, load_static_splits, model_metrics_dir, save_json, split_xy
from src.utils.config import resolve_path


def _validate_pchazard_targets(durations, events, split_name):
    durations = np.asarray(durations, dtype=np.float32)
    events = np.asarray(events, dtype=np.int64)

    if durations.ndim != 1 or events.ndim != 1:
        raise ValueError(f"PCHazard {split_name} targets must be 1D arrays")
    if len(durations) != len(events):
        raise ValueError(f"PCHazard {split_name} durations/events length mismatch")
    if np.isnan(durations).any():
        raise ValueError(f"PCHazard {split_name} durations contain NaN values")
    if np.isnan(events.astype(np.float32)).any():
        raise ValueError(f"PCHazard {split_name} events contain NaN values")
    if (durations <= 0).any():
        raise ValueError(f"PCHazard {split_name} durations must be positive")
    if not np.isin(events, [0, 1]).all():
        raise ValueError(f"PCHazard {split_name} observed_event must be binary")
    return durations, events


def _validate_dropout(dropout):
    if dropout is None:
        return
    if isinstance(dropout, bool):
        raise ValueError("PCHazard dropout must be a float, a list of floats, or None")
    if isinstance(dropout, (float, int)):
        return
    if isinstance(dropout, list) and all(isinstance(value, (float, int)) and not isinstance(value, bool) for value in dropout):
        return
    raise ValueError("PCHazard dropout must be a float, a list of floats, or None")


def _default_grid(max_horizon_days):
    upper = int(np.floor(float(max_horizon_days)))
    if upper > 1:
        return list(range(1, upper))
    return [float(max_horizon_days)]


def _output_path(configured_path, default_path):
    return resolve_path(configured_path) if configured_path else Path(default_path)


def _pchazard_time_dependent_metrics(
    split_name,
    time_values,
    event_values,
    cumulative_incidence,
    time_grid,
    train_time,
    train_event,
    eval_times_days,
):
    antolini = {"split": split_name, **antolini_ctd(time_values, event_values, cumulative_incidence, time_grid)}
    weighted_rows = []
    for horizon in eval_times_days:
        bin_idx = time_to_index(horizon, time_grid)
        risk = cumulative_incidence[:, bin_idx]
        result = weighted_c_index_at_horizon(train_time, train_event, time_values, event_values, risk, horizon)
        weighted_rows.append(
            {
                "split": split_name,
                "horizon_days": float(horizon),
                "horizon_bin": int(bin_idx + 1),
                **result,
            }
        )
    return antolini, weighted_rows


def train_pchazard(config, logger):
    try:
        import torch
        import torchtuples as tt
        from pycox.models import PCHazard
    except ImportError as exc:
        raise ImportError("torch, pycox and torchtuples are required for PCHazard") from exc

    paths = config["paths"]
    model_cfg = config["model"]
    train, val, test = load_static_splits(paths)
    x_train, time_train, event_train, _ = split_xy(train)
    x_val, time_val, event_val, _ = split_xy(val)
    _validate_pchazard_targets(time_train.values, event_train.values, "train")
    _validate_pchazard_targets(time_val.values, event_val.values, "validation")
    max_horizon_days = model_cfg.get("max_horizon_days", 10)
    time_train, event_train = cap_survival_targets(time_train, event_train, max_horizon_days)
    time_val, event_val = cap_survival_targets(time_val, event_val, max_horizon_days)
    durations_train, events_train = _validate_pchazard_targets(time_train.values, event_train.values, "train capped")
    durations_val, events_val = _validate_pchazard_targets(time_val.values, event_val.values, "validation capped")

    num_durations = int(model_cfg.get("num_durations", 10))
    labtrans = PCHazard.label_transform(
        num_durations,
        scheme=model_cfg.get("scheme", "equidistant"),
        min_=0.0,
    )
    y_train = labtrans.fit_transform(durations_train, events_train)
    y_val = labtrans.transform(durations_val, events_val)

    device = get_device(model_cfg.get("device", "auto"))
    logger.info("PCHazard device: %s", device)

    in_features = int(x_train.shape[1])
    num_nodes = model_cfg.get("hidden_layers", [128, 64])
    out_features = int(labtrans.out_features)
    batch_norm = model_cfg.get("batch_norm", True)
    dropout = model_cfg.get("dropout", 0.1)
    output_bias = model_cfg.get("output_bias", True)

    if not isinstance(num_nodes, list):
        raise ValueError("PCHazard hidden_layers must be a list of integers")
    if not all(isinstance(node, int) and not isinstance(node, bool) for node in num_nodes):
        raise ValueError("PCHazard hidden_layers must only contain integers")
    if not isinstance(batch_norm, bool):
        raise ValueError("PCHazard batch_norm must be a boolean")
    if not isinstance(output_bias, bool):
        raise ValueError("PCHazard output_bias must be a boolean")
    _validate_dropout(dropout)

    logger.info("PCHazard X_train shape: %s", x_train.shape)
    logger.info("PCHazard in_features: %s (%s)", in_features, type(in_features).__name__)
    logger.info("PCHazard out_features: %s (%s)", out_features, type(out_features).__name__)
    logger.info("PCHazard num_nodes: %s", num_nodes)
    logger.info("PCHazard batch_norm: %s", batch_norm)
    logger.info("PCHazard dropout: %s", dropout)
    logger.info("PCHazard output_bias: %s", output_bias)

    assert isinstance(in_features, int)
    assert isinstance(out_features, int)
    assert isinstance(num_nodes, list)
    assert all(isinstance(n, int) for n in num_nodes)
    assert isinstance(batch_norm, bool)
    assert isinstance(output_bias, bool)

    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=num_nodes,
        out_features=out_features,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=output_bias,
    ).to(device)
    model = PCHazard(
        net,
        tt.optim.Adam(model_cfg.get("learning_rate", 1e-3)),
        device=device,
        duration_index=labtrans.cuts,
    )

    callbacks = []
    if model_cfg.get("early_stopping_patience", 10):
        callbacks.append(tt.callbacks.EarlyStopping(patience=model_cfg["early_stopping_patience"]))
    log = model.fit(
        x_train.values,
        y_train,
        batch_size=model_cfg.get("batch_size", 256),
        epochs=model_cfg.get("epochs", 50),
        callbacks=callbacks,
        val_data=(x_val.values, y_val),
        verbose=model_cfg.get("verbose", False),
    )
    model_path = Path(paths["models_dir"]) / "pchazard_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_net(str(model_path.with_suffix(".pt")))
    metrics_dir = model_metrics_dir(paths, "pchazard")
    train_log = pd.DataFrame(log.to_pandas())
    train_log.to_csv(metrics_dir / "pchazard_train_log.csv", index=False)
    best_val_loss = None
    if "val_loss" in train_log:
        best_val_loss = float(train_log["val_loss"].min())

    eval_cfg = config.get("evaluation", {})
    evaluation_time_grid = eval_cfg.get("evaluation_time_grid", model_cfg.get("evaluation_time_grid", _default_grid(max_horizon_days)))
    horizon_times = eval_cfg.get("horizon_times", model_cfg.get("horizon_times", _default_grid(max_horizon_days)))
    metrics = {
        "model": "pchazard",
        "best_val_loss": best_val_loss,
        "evaluation_time_grid": evaluation_time_grid,
        "horizon_times": horizon_times,
        "splits": {},
    }
    predictions = []
    antolini_rows = []
    weighted_rows = []
    for split_name, split_df in {"train": train, "validation": val, "test": test}.items():
        x, time, event, ids = split_xy(split_df)
        _validate_pchazard_targets(time.values, event.values, split_name)
        time, event = cap_survival_targets(time, event, max_horizon_days)
        time_values, event_values = _validate_pchazard_targets(time.values, event.values, f"{split_name} capped")
        surv = model.predict_surv_df(x.values)
        time_grid = surv.index.to_numpy(dtype=float)
        cumulative_incidence = survival_to_cumulative_incidence(surv)
        risk = 1.0 - surv.iloc[-1].values
        preds = ids.copy()
        preds["risk_score"] = risk
        preds["model"] = "pchazard"
        preds["split"] = split_name
        predictions.append(preds)
        metrics["splits"][split_name] = evaluate_predictions(
            time_values,
            event_values,
            risk_scores=risk,
            survival=surv,
            risk_metric_name="harrell_c_index_final_risk",
            evaluation_time_grid=evaluation_time_grid,
            censoring_time=durations_train,
            censoring_event=events_train,
            compute_curve_metrics=split_name != "train",
        )
        antolini, weighted = _pchazard_time_dependent_metrics(
            split_name,
            time_values,
            event_values,
            cumulative_incidence,
            time_grid,
            durations_train,
            events_train,
            horizon_times,
        )
        antolini_rows.append(antolini)
        weighted_rows.extend(weighted)
        metrics["splits"][split_name]["ctd_antolini"] = antolini["ctd"]
        metrics["splits"][split_name]["horizon_c_index"] = horizon_c_index_dict(weighted)
        metrics["splits"][split_name]["mean_horizon_c_index"] = mean_horizon_c_index(weighted)
        logger.info(
            "PCHazard %s final-risk C-index: %.4f",
            split_name,
            metrics["splits"][split_name]["harrell_c_index_final_risk"],
        )
        if split_name == "test":
            surv.to_csv(Path(paths["predictions_dir"]) / "pchazard_test_survival_curves.csv")

    pd.concat(predictions, ignore_index=True).to_parquet(Path(paths["predictions_dir"]) / "pchazard_predictions.parquet", index=False)
    weighted_path = _output_path(
        eval_cfg.get("weighted_c_index_path"),
        metrics_dir / "pchazard_weighted_c_index_by_horizon.csv",
    )
    antolini_path = _output_path(
        eval_cfg.get("antolini_ctd_path"),
        metrics_dir / "pchazard_antolini_ctd.csv",
    )
    weighted_path.parent.mkdir(parents=True, exist_ok=True)
    antolini_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weighted_rows).to_csv(weighted_path, index=False)
    pd.DataFrame(antolini_rows).to_csv(antolini_path, index=False)
    save_json(metrics, metrics_dir / "pchazard_metrics.json")
    return metrics
