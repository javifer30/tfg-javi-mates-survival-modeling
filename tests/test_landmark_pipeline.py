from pathlib import Path

import numpy as np
import pandas as pd

from src.data.landmark_faithful_dataset import prepare_arrays, validate_prepared
from src.data.landmark_static_dataset import EVENT_COL, DURATION_COL, REL_DURATION_DAYS_COL, load_landmark_static_base_table
from src.utils.landmark import (
    apply_landmark_dynamic_data_config,
    apply_landmark_faithful_config,
    apply_landmark_static_data_config,
    apply_landmark_static_tuning_config,
    dynamic_suffix,
    split_files_for_suffix,
    static_suffix,
)


class _Logger:
    def info(self, *args, **kwargs):
        pass


def test_landmark_config_resolution_uses_cli_hours_and_current_flat_features():
    static_cfg = {
        "paths": {"flat_features_path": "data/processed/mimic_extraction/flat_features_with_time_since_admission.csv"},
        "target": {"prediction_time_hours": 72, "max_horizon_days": 10},
    }
    resolved_static = apply_landmark_static_data_config(static_cfg, 24)
    assert resolved_static["target"]["prediction_time_hours"] == 24
    assert resolved_static["paths"]["flat_features_path"].endswith("flat_features_with_time_since_admission.csv")
    assert resolved_static["paths"]["processed_dir"] == "data/processed/landmark_24h/static"
    assert resolved_static["output_file_suffix"] == "static_24h"

    dynamic_cfg = {"paths": {}, "static": {}, "temporal": {}}
    resolved_dynamic = apply_landmark_dynamic_data_config(dynamic_cfg, 48)
    assert resolved_dynamic["temporal"]["hours"] == 48
    assert resolved_dynamic["temporal"]["max_offset_minutes_exclusive"] == 48 * 60
    assert resolved_dynamic["static"]["train_path"].endswith("train_static_48h.parquet")
    assert resolved_dynamic["output_file_suffix"] == "dynamic_48h"

    tuning_cfg = {"experiment": {}, "paths": {}}
    resolved_tuning = apply_landmark_static_tuning_config(tuning_cfg, 72)
    assert resolved_tuning["paths"]["outputs_dir"] == "outputs/landmark_72h/static"
    assert resolved_tuning["static_file_suffix"] == static_suffix(72)


def test_landmark_static_target_formula_for_24h(tmp_path):
    flat = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3, 4],
            "gender": ["M", "F", "M", "F"],
            "age": [50, 60, 70, 80],
        }
    )
    labels = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3, 4],
            "actualhospitalmortality": [1, 1, 0, 1],
            "actualiculos": [0.5, 2.0, 12.0, 15.0],
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
        "target": {"prediction_time_hours": 24, "max_horizon_days": 10},
    }
    df = load_landmark_static_base_table(config, _Logger()).sort_values("patientunitstayid")
    assert df["patientunitstayid"].tolist() == [2, 3, 4]
    assert np.allclose(df[REL_DURATION_DAYS_COL].to_numpy(), [1.0, 11.0, 14.0])
    assert np.allclose(df[DURATION_COL].to_numpy(), [1.0, 10.0, 10.0])
    assert df[EVENT_COL].tolist() == [1, 0, 0]


def test_faithful_preparation_accepts_variable_landmark_length():
    n, hours, temporal_features, static_features = 3, 24, 2, 4
    source = {}
    for split in ["train", "validation", "test"]:
        x_seq = np.arange(n * hours * temporal_features, dtype="float32").reshape(n, hours, temporal_features)
        m_seq = np.ones_like(x_seq, dtype="float32")
        source[split] = {
            "patient_ids": np.arange(n) + (0 if split == "train" else 10 if split == "validation" else 20),
            "X_seq": x_seq,
            "M_seq": m_seq,
            "X_static": np.ones((n, static_features), dtype="float32"),
            "duration_eval_days": np.array([1.0, 5.0, 10.0], dtype="float32"),
            "duration_rel_days": np.array([1.0, 5.0, 12.0], dtype="float32"),
            "event_eval": np.array([1, 0, 0], dtype="int64"),
        }
    prepared, _ = prepare_arrays(source)
    checks = validate_prepared(prepared)
    assert checks["train_validation_no_overlap"] is True
    assert prepared["train"]["X_seq"].shape == (n, hours, temporal_features)


def test_faithful_split_files_are_landmark_specific():
    suffix = dynamic_suffix(48)
    assert split_files_for_suffix(suffix) == {
        "train": "train_dynamic_48h.npz",
        "validation": "val_dynamic_48h.npz",
        "test": "test_dynamic_48h.npz",
    }
    config = apply_landmark_faithful_config({"experiment": {}, "paths": {}, "data": {}}, 48, "dysurv_faithful")
    assert config["data"]["source_split_files"]["train"] == "train_dynamic_48h.npz"
    assert config["paths"]["source_dataset_dir"] == "data/processed/landmark_48h/dynamic_dysurv_features"
