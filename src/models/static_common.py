import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.static_dataset import EVENT_COL, ID_COL, SPLIT_COL, TIME_COL, feature_columns


def load_static_splits(paths):
    train = pd.read_parquet(paths["train_path"])
    val = pd.read_parquet(paths["val_path"])
    test = pd.read_parquet(paths["test_path"])
    return train, val, test


def split_xy(df):
    cols = feature_columns(df)
    return (
        df[cols].astype("float32"),
        df[TIME_COL].astype("float32"),
        df[EVENT_COL].astype("int64"),
        df[[ID_COL, TIME_COL, EVENT_COL, SPLIT_COL]].copy(),
    )


def get_device(device_name):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_output_dirs(paths):
    for key in ["models_dir", "checkpoints_dir", "metrics_dir", "predictions_dir", "figures_dir"]:
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def model_metrics_dir(paths, model_name):
    path = Path(paths["metrics_dir"]) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def make_time_grid(max_horizon_days, num_durations):
    return np.linspace(0.0, float(max_horizon_days), int(num_durations) + 1)[1:]


def cap_survival_targets(time, event, max_horizon_days):
    """Censor stays that do not have the event before the configured horizon."""
    time_values = np.asarray(time, dtype="float32")
    event_values = np.asarray(event, dtype="int64")
    capped_time = np.minimum(time_values, float(max_horizon_days))
    capped_event = ((event_values == 1) & (time_values <= float(max_horizon_days))).astype("int64")
    return pd.Series(capped_time), pd.Series(capped_event)
