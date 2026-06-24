"""
Library-based static models for the parametrizable landmark experiment.

The implementation is inspired by the DySurv static MIMIC-IV notebook but uses
the TFG landmark cohort, train-only preprocessing and validation-only model
selection. The same wrapper trains Kaplan-Meier, CoxPH, DeepSurv-style Cox,
LogisticHazard, PCHazard and DeepHitSingle so their outputs share one metric
format.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.landmark_static_dataset import DURATION_COL, EVENT_COL, ID_COL, SPLIT_COL, feature_columns
from src.evaluation.landmark_survival_metrics import (
    eval_surv_metrics,
    horizon_c_index_rows,
    mean_horizon_c_index,
    metric_integration_grid,
    survival_at_times,
)


MODEL_ALIASES = {
    "kaplan_meier": "kaplan_meier",
    "coxph": "coxph",
    "deepsurv": "deepsurv",
    "logistic_hazard": "logistic_hazard",
    "pchazard": "pchazard",
    "deephit_single": "deephit_single",
}


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_split(paths, split_name):
    suffix = paths.get("static_file_suffix", paths.get("file_suffix", "static_landmark"))
    path = Path(paths["processed_dir"]) / f"{split_name}_{suffix}.parquet"
    return pd.read_parquet(path)


def load_landmark_static_splits(config):
    requested = config.get("evaluation", {}).get("splits", ["train", "validation", "test"])
    mapping = {"train": "train", "validation": "val", "test": "test"}
    unknown = [split for split in requested if split not in mapping]
    if unknown:
        raise ValueError(f"Unknown static_landmark split(s): {unknown}")
    if "test" in requested and not config.get("evaluation", {}).get("allow_test_metrics", True):
        raise ValueError("Test metrics requested while allow_test_metrics is false")
    return {split: _load_split(config["paths"], mapping[split]) for split in requested}


def split_xy(df):
    cols = feature_columns(df)
    x = df[cols].astype("float32")
    durations = df[DURATION_COL].astype("float32").to_numpy()
    events = df[EVENT_COL].astype("int64").to_numpy()
    ids = df[[ID_COL, DURATION_COL, EVENT_COL, SPLIT_COL]].copy()
    return x, durations, events, ids


def _time_grid(config):
    return list(config.get("evaluation", {}).get("evaluation_time_grid", list(range(1, 11))))


def _horizon_times(config):
    return list(config.get("evaluation", {}).get("horizon_times", list(range(1, 11))))


def _audit_dir(config):
    configured = config.get("paths", {}).get("audit_dir")
    if configured:
        path = Path(configured)
    else:
        output_dir = Path(config["paths"]["outputs_dir"])
        parts = list(output_dir.parts)
        path = output_dir
        for marker in ["tuning", "final", "audit_runs"]:
            if marker in parts:
                path = Path(*parts[: parts.index(marker)])
                break
        path = path / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_dirs(config, model_name):
    root = Path(config["paths"]["outputs_dir"])
    metrics_dir = root / "metrics" / model_name
    predictions_dir = root / "predictions" / model_name
    models_dir = root / "models" / model_name
    for path in [metrics_dir, predictions_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return metrics_dir, predictions_dir, models_dir


def _float_list(values):
    return [float(x) for x in np.asarray(values, dtype=float)]


def _tail_stats(surv, horizon):
    values = survival_at_times(surv, [float(horizon)])[:, 0]
    return {
        "min_survival_at_10d": float(np.nanmin(values)),
        "max_survival_at_10d": float(np.nanmax(values)),
        "mean_survival_at_10d": float(np.nanmean(values)),
        "share_patients_survival_at_10d_below_1e-6": float(np.mean(values < 1e-6)),
        "share_patients_survival_at_10d_equal_zero_or_near_zero": float(np.mean(values <= 1e-12)),
    }


def _survival_sanity_row(model_name, split_name, surv, target, config):
    durations, _ = target
    values = surv.to_numpy(dtype=float)
    diffs = np.diff(values, axis=0) if values.shape[0] > 1 else np.zeros((0, values.shape[1]))
    monotonicity_violations = int(np.sum(diffs > 1e-8))
    has_nan = bool(np.isnan(values).any())
    out_of_bounds = bool(np.nanmin(values) < -1e-8 or np.nanmax(values) > 1.0 + 1e-8)
    tail = _tail_stats(surv, config["experiment"]["max_horizon_days"])
    status = "ok"
    notes = []
    if has_nan:
        status = "issue"
        notes.append("contains_nan")
    if out_of_bounds:
        status = "issue"
        notes.append("outside_[0,1]")
    if monotonicity_violations:
        status = "issue"
        notes.append("non_monotone")
    if model_name == "deephit_single" and tail["share_patients_survival_at_10d_below_1e-6"] > 0.95:
        status = "issue"
        notes.append("deephit_tail_nearly_zero_for_most_patients")
    if model_name == "pchazard" and tail["mean_survival_at_10d"] > 0.99:
        status = "issue"
        notes.append("pchazard_tail_suspiciously_high")
    return {
        "model": model_name,
        "split": split_name,
        "n_patients": int(values.shape[1]),
        "n_time_points": int(values.shape[0]),
        "has_nan": has_nan,
        "min_survival": float(np.nanmin(values)),
        "max_survival": float(np.nanmax(values)),
        "monotonicity_violations": monotonicity_violations,
        **tail,
        "status": status,
        "notes": ";".join(notes),
    }


def _write_audit_table(path, rows, key_cols):
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def _write_discrete_cuts_audit(model_name, model_cfg, labtrans, config):
    if labtrans is None:
        return
    cuts = np.asarray(getattr(labtrans, "cuts", []), dtype=float)
    diffs = np.diff(cuts) if cuts.size > 1 else np.asarray([], dtype=float)
    row = {
        "model": model_name,
        "labtrans.cuts": _float_list(cuts),
        "duration_index": _float_list(cuts),
        "num_durations": int(model_cfg.get("num_durations", len(cuts))),
        "min_cut": float(np.min(cuts)) if cuts.size else math.nan,
        "max_cut": float(np.max(cuts)) if cuts.size else math.nan,
        "are_cuts_sorted": bool(np.all(diffs >= -1e-8)) if diffs.size else True,
        "approx_equal_spacing": bool(np.allclose(diffs, diffs[0], rtol=1e-4, atol=1e-4)) if diffs.size else True,
    }
    path = _audit_dir(config) / "discrete_time_cuts_summary.json"
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data[model_name] = row
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_grid_audit(config, grids_by_split):
    payload = {
        "metric_integration_grid": {
            "description": "Per-split grid used only for integrated Brier score and integrated NBLL.",
            "num_points_requested": int(config.get("evaluation", {}).get("metric_integration_num_points", 100)),
            "by_split": {split: _float_list(grid) for split, grid in grids_by_split.items()},
        },
        "horizon_times": {
            "description": "Daily horizons used only for horizon C-index.",
            "values": _horizon_times(config),
        },
    }
    (_audit_dir(config) / "evaluation_grids.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_deephit_tail_audit(model_name, split_surv, labtrans, config):
    if model_name != "deephit_single" or "validation" not in split_surv:
        return
    surv = split_surv["validation"]
    horizon = config["experiment"]["max_horizon_days"]
    values = survival_at_times(surv, [horizon])[:, 0]
    audit = {
        "labtrans.cuts": _float_list(getattr(labtrans, "cuts", [])),
        "duration_index": _float_list(surv.index),
        **_tail_stats(surv, horizon),
    }
    audit_dir = _audit_dir(config)
    (audit_dir / "deephit_single_time_grid_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame({"patient_column": np.arange(len(values)), "survival_at_10d": values}).to_csv(
        audit_dir / "deephit_single_survival_tail_check.csv",
        index=False,
    )


def _old_pchazard_metrics(config):
    path = _audit_dir(config).parent / "tuning" / "pchazard" / "tuning_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return {}
    row = completed.sort_values("validation_ctd_antolini", ascending=False).iloc[0]
    return {
        "old_validation_ctd_antolini": float(row["validation_ctd_antolini"]),
        "old_mean_horizon_c_index": float(row["validation_mean_horizon_c_index"]),
        "old_config_id": str(row["config_id"]),
    }


def _write_pchazard_audit(model_name, model, split_surv, metrics, config):
    if model_name != "pchazard" or "validation" not in split_surv:
        return
    surv = split_surv["validation"]
    values = surv.to_numpy(dtype=float)
    diffs = np.diff(values, axis=0) if values.shape[0] > 1 else np.zeros((0, values.shape[1]))
    payload = {
        **_old_pchazard_metrics(config),
        "new_validation_ctd_antolini": float(metrics["splits"]["validation"]["ctd_antolini"]),
        "new_mean_horizon_c_index": float(metrics["splits"]["validation"]["mean_horizon_c_index"]),
        "model_sub_used": int(getattr(model, "sub", config["model"].get("sub", 1))),
        "duration_index": _float_list(surv.index),
        "survival_monotonicity_check": {
            "monotonicity_violations": int(np.sum(diffs > 1e-8)),
            "is_monotone_non_increasing": bool(np.all(diffs <= 1e-8)),
        },
        **_tail_stats(surv, config["experiment"]["max_horizon_days"]),
    }
    (_audit_dir(config) / "pchazard_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _network(num_features, model_cfg, out_features):
    import torchtuples as tt

    return tt.practical.MLPVanilla(
        in_features=int(num_features),
        num_nodes=list(model_cfg.get("hidden_layers", [64, 32])),
        out_features=int(out_features),
        batch_norm=bool(model_cfg.get("batch_norm", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_bias=bool(model_cfg.get("output_bias", False)),
    )


def _optimizer(model_cfg):
    import torchtuples as tt

    return tt.optim.Adam(
        lr=float(model_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0)),
    )


def _callbacks(model_cfg):
    import torchtuples as tt

    if not model_cfg.get("early_stopping", True):
        return []
    return [tt.callbacks.EarlyStopping(patience=int(model_cfg.get("patience", 20)))]


def _save_labtrans_metadata(model_name, model_cfg, labtrans, metrics_dir, config):
    metadata = {
        "model": model_name,
        "num_durations": model_cfg.get("num_durations"),
        "max_horizon_days": config["experiment"]["max_horizon_days"],
        "time_unit": config["experiment"]["time_unit"],
        "metric_integration_num_points": int(config.get("evaluation", {}).get("metric_integration_num_points", 100)),
        "horizon_times": _horizon_times(config),
        "cuts": None,
        "duration_index": None,
    }
    if labtrans is not None:
        metadata["cuts"] = [float(x) for x in np.asarray(getattr(labtrans, "cuts", []), dtype=float)]
        metadata["duration_index"] = metadata["cuts"]
    save_json(metadata, metrics_dir / "time_discretization.json")
    _write_discrete_cuts_audit(model_name, model_cfg, labtrans, config)


def _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir, model=None, labtrans=None):
    horizon_times = _horizon_times(config)
    integration_points = int(config.get("evaluation", {}).get("metric_integration_num_points", 100))
    metrics = {
        "model": model_name,
        "metric_integration_num_points": integration_points,
        "horizon_times": horizon_times,
        "metric_integration_grid_by_split": {},
        "splits": {},
    }
    horizon_rows = []
    sanity_rows = []
    grids_by_split = {}
    for split_name, surv in split_surv.items():
        durations, events = split_targets[split_name]
        integration_grid = metric_integration_grid(
            surv,
            durations,
            config["experiment"]["max_horizon_days"],
            num_points=integration_points,
        )
        grids_by_split[split_name] = integration_grid
        metrics["metric_integration_grid_by_split"][split_name] = _float_list(integration_grid)
        split_metrics = eval_surv_metrics(surv, durations, events, integration_grid)
        rows = horizon_c_index_rows(model_name, split_name, surv, durations, events, horizon_times)
        split_metrics["horizon_c_index"] = {str(int(row["horizon_day"])): row["c_index"] for row in rows}
        split_metrics["mean_horizon_c_index"] = mean_horizon_c_index(rows)
        metrics["splits"][split_name] = split_metrics
        horizon_rows.extend(rows)
        sanity_rows.append(_survival_sanity_row(model_name, split_name, surv, (durations, events), config))

    pd.DataFrame(horizon_rows).to_csv(metrics_dir / "horizon_c_index.csv", index=False)
    save_json(metrics, metrics_dir / f"{model_name}_metrics.json")
    audit_dir = _audit_dir(config)
    _write_grid_audit(config, grids_by_split)
    _write_audit_table(
        audit_dir / "survival_curve_sanity_checks.csv",
        sanity_rows,
        key_cols=["model", "split"],
    )
    _write_deephit_tail_audit(model_name, split_surv, labtrans, config)
    _write_pchazard_audit(model_name, model, split_surv, metrics, config)

    if config.get("evaluation", {}).get("save_example_curves", True) and "test" in split_surv:
        _save_example_curves(model_name, split_surv["test"], split_targets["test"], config, predictions_dir)
    return metrics


def _save_example_curves(model_name, surv, target, config, predictions_dir):
    durations, events = target
    horizon = float(config["experiment"]["max_horizon_days"])
    risk = 1.0 - survival_at_times(surv, [horizon])[:, 0]
    n = min(int(config.get("evaluation", {}).get("n_example_patients", 9)), len(risk))
    if n <= 0:
        return
    quantiles = np.linspace(0.1, 0.9, n)
    selected = []
    used = set()
    for quantile in quantiles:
        target_risk = float(np.quantile(risk, quantile))
        order = np.argsort(np.abs(risk - target_risk))
        idx = next(int(i) for i in order if int(i) not in used)
        used.add(idx)
        selected.append(
            {
                "column_index": idx,
                "risk_quantile": float(quantile),
                "risk_at_10d": float(risk[idx]),
                "duration_eval_days": float(durations[idx]),
                "event_eval": int(events[idx]),
            }
        )

    curves = surv.iloc[:, [row["column_index"] for row in selected]].copy()
    curves.insert(0, "time_days", curves.index.astype(float))
    curves.to_csv(predictions_dir / f"example_survival_curves_{model_name}.csv", index=False)
    pd.DataFrame(selected).to_csv(predictions_dir / f"example_survival_selection_{model_name}.csv", index=False)


def _train_log_to_csv(log, path):
    if log is None:
        return
    try:
        df = log.to_pandas()
    except Exception:
        return
    df.to_csv(path, index=False)


def _fit_discrete_pycox(model_name, model_class, labtrans_class, splits, config, logger):
    model_cfg = config["model"]
    x_train, durations_train, events_train, _ = split_xy(splits["train"])
    x_val, durations_val, events_val, _ = split_xy(splits["validation"])
    label_arg = int(model_cfg.get("num_durations", 10)) if model_name == "pchazard" else np.asarray(model_cfg["cuts"], dtype="float32") if "cuts" in model_cfg else int(model_cfg.get("num_durations", 10))
    labtrans = labtrans_class.label_transform(label_arg, **model_cfg.get("label_transform_kwargs", {}))
    y_train = labtrans.fit_transform(durations_train, events_train)
    y_val = labtrans.transform(durations_val, events_val)

    net = _network(x_train.shape[1], model_cfg, labtrans.out_features)
    kwargs = {"duration_index": labtrans.cuts}
    if model_name == "deephit_single":
        kwargs.update({"alpha": float(model_cfg.get("alpha", 0.2)), "sigma": float(model_cfg.get("sigma", 0.1))})
    model = model_class(net, _optimizer(model_cfg), **kwargs)
    log = model.fit(
        x_train.values,
        y_train,
        int(model_cfg.get("batch_size", 256)),
        int(model_cfg.get("epochs", 256)),
        callbacks=_callbacks(model_cfg),
        verbose=bool(model_cfg.get("verbose", False)),
        val_data=(x_val.values, y_val),
        val_batch_size=int(model_cfg.get("batch_size", 256)),
    )

    metrics_dir, predictions_dir, models_dir = _output_dirs(config, model_name)
    _train_log_to_csv(log, metrics_dir / "train_log.csv")
    _save_labtrans_metadata(model_name, model_cfg, labtrans, metrics_dir, config)
    if model_cfg.get("save_model", False):
        model.save_net(str(models_dir / f"{model_name}.pt"))
    if model_name == "pchazard":
        model.sub = int(model_cfg.get("sub", 10))

    split_surv = {}
    split_targets = {}
    interpolate = model_cfg.get("interpolate")
    for split_name, df in splits.items():
        x, durations, events, _ = split_xy(df)
        predictor = model.interpolate(int(interpolate)) if interpolate else model
        split_surv[split_name] = predictor.predict_surv_df(x.values)
        split_targets[split_name] = (durations, events)
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir, model=model, labtrans=labtrans)


def train_kaplan_meier(config, logger):
    from lifelines import KaplanMeierFitter

    model_name = "kaplan_meier"
    splits = load_landmark_static_splits(config)
    _, durations_train, events_train, _ = split_xy(splits["train"])
    km = KaplanMeierFitter()
    km.fit(durations_train, event_observed=events_train)
    time_index = np.asarray(_time_grid(config), dtype=float)
    values = km.survival_function_at_times(time_index).to_numpy(dtype=float)
    metrics_dir, predictions_dir, _ = _output_dirs(config, model_name)

    split_surv = {}
    split_targets = {}
    for split_name, df in splits.items():
        _, durations, events, _ = split_xy(df)
        surv = np.tile(values.reshape(-1, 1), (1, len(df)))
        split_surv[split_name] = pd.DataFrame(surv, index=time_index)
        split_targets[split_name] = (durations, events)
    _save_labtrans_metadata(model_name, config["model"], None, metrics_dir, config)
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir)


def train_lifelines_coxph(config, logger):
    from lifelines import CoxPHFitter

    model_name = "coxph"
    model_cfg = config["model"]
    splits = load_landmark_static_splits(config)
    x_train, durations_train, events_train, _ = split_xy(splits["train"])
    train_df = x_train.copy()
    train_df[DURATION_COL] = durations_train
    train_df[EVENT_COL] = events_train
    model = CoxPHFitter(
        penalizer=float(model_cfg.get("penalizer", 0.1)),
        l1_ratio=float(model_cfg.get("l1_ratio", 0.0)),
    )
    model.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    time_index = np.asarray(_time_grid(config), dtype=float)
    metrics_dir, predictions_dir, models_dir = _output_dirs(config, model_name)

    split_surv = {}
    split_targets = {}
    for split_name, df in splits.items():
        x, durations, events, _ = split_xy(df)
        split_surv[split_name] = model.predict_survival_function(x, times=time_index)
        split_targets[split_name] = (durations, events)
    if model_cfg.get("save_model", False):
        import joblib

        joblib.dump(model, models_dir / "coxph_model.pkl")
    _save_labtrans_metadata(model_name, model_cfg, None, metrics_dir, config)
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir, model=model)


def train_deepsurv(config, logger):
    from pycox.models import CoxPH

    model_name = "deepsurv"
    model_cfg = config["model"]
    splits = load_landmark_static_splits(config)
    x_train, durations_train, events_train, _ = split_xy(splits["train"])
    x_val, durations_val, events_val, _ = split_xy(splits["validation"])
    y_train = (durations_train, events_train)
    y_val = (durations_val, events_val)
    net = _network(x_train.shape[1], model_cfg, 1)
    model = CoxPH(net, _optimizer(model_cfg))
    log = model.fit(
        x_train.values,
        y_train,
        int(model_cfg.get("batch_size", 256)),
        int(model_cfg.get("epochs", 256)),
        callbacks=_callbacks(model_cfg),
        verbose=bool(model_cfg.get("verbose", False)),
        val_data=(x_val.values, y_val),
        val_batch_size=int(model_cfg.get("batch_size", 256)),
    )
    model.compute_baseline_hazards()

    metrics_dir, predictions_dir, models_dir = _output_dirs(config, model_name)
    _train_log_to_csv(log, metrics_dir / "train_log.csv")
    if model_cfg.get("save_model", False):
        model.save_net(str(models_dir / "deepsurv.pt"))
    _save_labtrans_metadata(model_name, model_cfg, None, metrics_dir, config)

    split_surv = {}
    split_targets = {}
    for split_name, df in splits.items():
        x, durations, events, _ = split_xy(df)
        split_surv[split_name] = model.predict_surv_df(x.values)
        split_targets[split_name] = (durations, events)
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir, model=model)


def train_logistic_hazard(config, logger):
    from pycox.models import LogisticHazard

    return _fit_discrete_pycox("logistic_hazard", LogisticHazard, LogisticHazard, load_landmark_static_splits(config), config, logger)


def train_pchazard(config, logger):
    from pycox.models import PCHazard

    return _fit_discrete_pycox("pchazard", PCHazard, PCHazard, load_landmark_static_splits(config), config, logger)


def train_deephit_single(config, logger):
    from pycox.models import DeepHitSingle

    return _fit_discrete_pycox("deephit_single", DeepHitSingle, DeepHitSingle, load_landmark_static_splits(config), config, logger)


TRAINERS = {
    "kaplan_meier": train_kaplan_meier,
    "coxph": train_lifelines_coxph,
    "deepsurv": train_deepsurv,
    "logistic_hazard": train_logistic_hazard,
    "pchazard": train_pchazard,
    "deephit_single": train_deephit_single,
}


def train_landmark_static_model(config, logger):
    model_name = MODEL_ALIASES.get(config["model"]["name"])
    if model_name not in TRAINERS:
        raise ValueError(f"Unsupported static_landmark model: {config['model']['name']}")
    config["model"]["name"] = model_name
    logger.info("Training static_landmark model: %s", model_name)
    return TRAINERS[model_name](config, logger)
