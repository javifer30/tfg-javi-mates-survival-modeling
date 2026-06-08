"""DeepHit loaders for the shared time-dependent survival metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation import time_dependent_survival as td
from src.models.static_common import make_time_grid


@dataclass
class DeepHitPredictions:
    frame: pd.DataFrame
    event_probabilities: np.ndarray
    cumulative_incidence: np.ndarray
    time_grid: np.ndarray


def load_deephit_predictions(predictions_path, max_horizon_days, num_categories):
    frame = pd.read_parquet(predictions_path)
    event_columns = [f"event_probability_bin_{idx + 1}" for idx in range(int(num_categories))]
    missing = [col for col in event_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing DeepHit event probability columns: {missing}")

    event_probabilities = frame[event_columns].to_numpy(dtype=float)
    cumulative_incidence = np.cumsum(event_probabilities, axis=1)
    time_grid = make_time_grid(max_horizon_days, num_categories)
    return DeepHitPredictions(frame, event_probabilities, cumulative_incidence, time_grid)


def cap_prediction_targets(frame, max_horizon_days):
    return td.cap_prediction_targets(frame, max_horizon_days)


def horizon_to_bin(horizon, time_grid):
    return td.time_to_index(horizon, time_grid)


def antolini_ctd(time, event, cumulative_incidence, max_horizon_days, num_categories):
    time_grid = make_time_grid(max_horizon_days, num_categories)
    return td.antolini_ctd(time, event, cumulative_incidence, time_grid)


def censoring_km(time, event):
    return td.censoring_km(time, event)


def weighted_c_index_at_horizon(train_time, train_event, test_time, test_event, risk, horizon):
    return td.weighted_c_index_at_horizon(train_time, train_event, test_time, test_event, risk, horizon)


def calculate_antolini_by_split(predictions, max_horizon_days, num_categories):
    return td.calculate_antolini_by_split(predictions, max_horizon_days)


def calculate_weighted_c_index_by_horizon(predictions, eval_times_days, max_horizon_days):
    return td.calculate_weighted_c_index_by_horizon(predictions, eval_times_days, max_horizon_days)
