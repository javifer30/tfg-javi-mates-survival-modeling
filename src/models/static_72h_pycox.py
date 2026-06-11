"""
Library-based static models for the static_72h_pycox experiment.

The implementation is inspired by the DySurv static MIMIC-IV notebook but uses
the new TFG 72-hour cohort, train-only preprocessing and validation-only model
selection.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.static_72h_dataset import DURATION_COL, EVENT_COL, ID_COL, SPLIT_COL, feature_columns
from src.evaluation.static_72h_metrics import eval_surv_metrics, horizon_c_index_rows, mean_horizon_c_index, survival_at_times
from src.models.static_common import save_json


MODEL_ALIASES = {
    "kaplan_meier": "kaplan_meier",
    "coxph": "coxph",
    "deepsurv": "deepsurv",
    "logistic_hazard": "logistic_hazard",
    "pchazard": "pchazard",
    "deephit_single": "deephit_single",
}


def _load_split(paths, split_name):
    path = Path(paths["processed_dir"]) / f"{split_name}_static_72h.parquet"
    return pd.read_parquet(path)


def load_static_72h_splits(config):
    requested = config.get("evaluation", {}).get("splits", ["train", "validation", "test"])
    mapping = {"train": "train", "validation": "val", "test": "test"}
    unknown = [split for split in requested if split not in mapping]
    if unknown:
        raise ValueError(f"Unknown static_72h split(s): {unknown}")
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


def _output_dirs(config, model_name):
    root = Path(config["paths"]["outputs_dir"])
    metrics_dir = root / "metrics" / model_name
    predictions_dir = root / "predictions" / model_name
    models_dir = root / "models" / model_name
    for path in [metrics_dir, predictions_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return metrics_dir, predictions_dir, models_dir


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
        "evaluation_time_grid": _time_grid(config),
        "horizon_times": _horizon_times(config),
        "cuts": None,
        "duration_index": None,
    }
    if labtrans is not None:
        metadata["cuts"] = [float(x) for x in np.asarray(getattr(labtrans, "cuts", []), dtype=float)]
        metadata["duration_index"] = metadata["cuts"]
    save_json(metadata, metrics_dir / "time_discretization.json")


def _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir):
    evaluation_time_grid = _time_grid(config)
    horizon_times = _horizon_times(config)
    metrics = {
        "model": model_name,
        "evaluation_time_grid": evaluation_time_grid,
        "horizon_times": horizon_times,
        "splits": {},
    }
    horizon_rows = []
    for split_name, surv in split_surv.items():
        durations, events = split_targets[split_name]
        split_metrics = eval_surv_metrics(surv, durations, events, evaluation_time_grid)
        rows = horizon_c_index_rows(model_name, split_name, surv, durations, events, horizon_times)
        split_metrics["horizon_c_index"] = {str(int(row["horizon_day"])): row["c_index"] for row in rows}
        split_metrics["mean_horizon_c_index"] = mean_horizon_c_index(rows)
        metrics["splits"][split_name] = split_metrics
        horizon_rows.extend(rows)

    pd.DataFrame(horizon_rows).to_csv(metrics_dir / "horizon_c_index.csv", index=False)
    save_json(metrics, metrics_dir / f"{model_name}_metrics.json")

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
    labtrans = labtrans_class.label_transform(
        int(model_cfg.get("num_durations", 10)),
        **model_cfg.get("label_transform_kwargs", {}),
    )
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

    split_surv = {}
    split_targets = {}
    interpolate = model_cfg.get("interpolate")
    for split_name, df in splits.items():
        x, durations, events, _ = split_xy(df)
        predictor = model.interpolate(int(interpolate)) if interpolate else model
        split_surv[split_name] = predictor.predict_surv_df(x.values)
        split_targets[split_name] = (durations, events)
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir)


def train_kaplan_meier(config, logger):
    from lifelines import KaplanMeierFitter

    model_name = "kaplan_meier"
    splits = load_static_72h_splits(config)
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
    splits = load_static_72h_splits(config)
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
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir)


def train_deepsurv(config, logger):
    from pycox.models import CoxPH

    model_name = "deepsurv"
    model_cfg = config["model"]
    splits = load_static_72h_splits(config)
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
    return _evaluate_and_save(model_name, split_surv, split_targets, config, metrics_dir, predictions_dir)


def train_logistic_hazard(config, logger):
    from pycox.models import LogisticHazard

    return _fit_discrete_pycox("logistic_hazard", LogisticHazard, LogisticHazard, load_static_72h_splits(config), config, logger)


def train_pchazard(config, logger):
    from pycox.models import PCHazard

    return _fit_discrete_pycox("pchazard", PCHazard, PCHazard, load_static_72h_splits(config), config, logger)


def train_deephit_single(config, logger):
    from pycox.models import DeepHitSingle

    return _fit_discrete_pycox("deephit_single", DeepHitSingle, DeepHitSingle, load_static_72h_splits(config), config, logger)


TRAINERS = {
    "kaplan_meier": train_kaplan_meier,
    "coxph": train_lifelines_coxph,
    "deepsurv": train_deepsurv,
    "logistic_hazard": train_logistic_hazard,
    "pchazard": train_pchazard,
    "deephit_single": train_deephit_single,
}


def train_static_72h_model(config, logger):
    model_name = MODEL_ALIASES.get(config["model"]["name"])
    if model_name not in TRAINERS:
        raise ValueError(f"Unsupported static_72h model: {config['model']['name']}")
    config["model"]["name"] = model_name
    logger.info("Training static_72h model: %s", model_name)
    return TRAINERS[model_name](config, logger)
