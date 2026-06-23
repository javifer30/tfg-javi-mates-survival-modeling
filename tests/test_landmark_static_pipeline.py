import math

import numpy as np
import pandas as pd

from scripts.landmark_static_tuning_impl import select_best_row
from src.data.landmark_static_dataset import (
    DURATION_COL,
    EVENT_COL,
    ID_COL,
    LandmarkStaticPreprocessor,
    RAW_EVENT_COL,
    REL_DURATION_DAYS_COL,
    load_landmark_static_base_table,
    validate_landmark_static_datasets,
)
from src.evaluation.landmark_survival_metrics import eval_surv_metrics, horizon_c_index


def _logger():
    class Logger:
        def info(self, *_args, **_kwargs):
            pass

    return Logger()


def test_static_landmark_target_filters_and_caps(tmp_path):
    flat = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3, 4],
            "age": [70, 60, 50, 40],
            "height": [170, 180, 175, 165],
            "weight": [80, 90, 85, 70],
        }
    )
    labels = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3, 4],
            "actualhospitalmortality": [1, 0, 1, 1],
            "actualiculos": [2.0, 4.0, 8.0, 15.0],
        }
    )
    flat_path = tmp_path / "flat.csv"
    labels_path = tmp_path / "labels.csv"
    flat.to_csv(flat_path, index=False)
    labels.to_csv(labels_path, index=False)
    config = {
        "paths": {"flat_features_path": str(flat_path), "labels_path": str(labels_path)},
        "columns": {
            "id_col": "patientunitstayid",
            "event_col": "actualhospitalmortality",
            "duration_col": "actualiculos",
            "duration_unit": "days",
        },
        "target": {"prediction_time_hours": 72, "max_horizon_days": 10},
    }

    df = load_landmark_static_base_table(config, _logger())

    assert set(df[ID_COL]) == {2, 3, 4}
    assert (df[REL_DURATION_DAYS_COL] > 0).all()
    assert df.loc[df[ID_COL] == 3, DURATION_COL].iloc[0] == 5.0
    assert df.loc[df[ID_COL] == 3, EVENT_COL].iloc[0] == 1
    assert df.loc[df[ID_COL] == 4, DURATION_COL].iloc[0] == 10.0
    assert df.loc[df[ID_COL] == 4, EVENT_COL].iloc[0] == 0


def test_static_landmark_preprocessor_train_fit_and_validation_checks():
    raw = pd.DataFrame(
        {
            ID_COL: [1, 2, 3],
            DURATION_COL: [1.0, 5.0, 10.0],
            EVENT_COL: [1, 1, 0],
            RAW_EVENT_COL: [1, 1, 1],
            REL_DURATION_DAYS_COL: [1.0, 5.0, 12.0],
            "gender": ["M", "F", "M"],
            "age": [60.0, 70.0, np.nan],
            "height": [170.0, 180.0, 175.0],
            "weight": [80.0, 90.0, 85.0],
            "hour": [72.0, 72.0, 72.0],
            "ethnicity": ["A", "B", "A"],
        }
    )
    prep = LandmarkStaticPreprocessor(
        standard_cols=["age", "height", "weight"],
        leave_numeric_cols=["hour"],
        categorical_cols=["ethnicity"],
        binary_maps={"gender": {"M": 1.0, "F": 0.0}},
        drop_feature_cols=[],
        rare_min_count=1,
    )
    train = prep.fit_transform(raw.iloc[[0]], "train")
    val = prep.transform(raw.iloc[[1]], "validation")
    test = prep.transform(raw.iloc[[2]], "test")

    validate_landmark_static_datasets(train, val, test, max_horizon_days=10)
    assert list(train.columns) == list(val.columns) == list(test.columns)
    assert not train.isna().any().any()


def test_evalsurv_metrics_and_horizon_c_index_are_finite():
    surv = pd.DataFrame(
        {
            0: [0.9, 0.6, 0.3],
            1: [0.95, 0.8, 0.7],
            2: [0.99, 0.95, 0.9],
        },
        index=[1.0, 2.0, 3.0],
    )
    durations = np.asarray([1.0, 2.0, 3.0], dtype=float)
    events = np.asarray([1, 1, 0], dtype=int)
    metrics = eval_surv_metrics(surv, durations, events, [1.0, 2.0, 3.0])
    assert math.isfinite(metrics["ctd_antolini"])
    result = horizon_c_index(durations, events, [0.9, 0.5, 0.1], 2.0)
    assert result["n_comparable_pairs"] > 0


def test_static_landmark_tuning_selection_ignores_failed_rows():
    rows = [
        {"status": "failed", "validation_ctd_antolini": 0.99, "validation_ibll": 0.1, "validation_ibs": 0.1, "config_id": "bad"},
        {"status": "completed", "validation_ctd_antolini": 0.70, "validation_ibll": 0.4, "validation_ibs": 0.2, "config_id": "ok_a"},
        {"status": "completed", "validation_ctd_antolini": 0.72, "validation_ibll": 0.5, "validation_ibs": 0.2, "config_id": "ok_b"},
    ]
    assert select_best_row(rows)["config_id"] == "ok_b"
