import numpy as np
import pandas as pd

from src.data.landmark_dynamic_dataset import build_landmark_dynamic_dataset


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


def _static_df(ids, split):
    return pd.DataFrame(
        {
            "patientunitstayid": ids,
            "duration_eval_days": [1.0 + i for i in range(len(ids))],
            "event_eval": [1 if i % 2 == 0 else 0 for i in range(len(ids))],
            "duration_from_admission_days": [4.0 + i for i in range(len(ids))],
            "observed_event_from_admission": [1 if i % 2 == 0 else 0 for i in range(len(ids))],
            "duration_rel_days": [1.0 + i for i in range(len(ids))],
            "split": split,
            "static_a": np.linspace(0.0, 1.0, len(ids), dtype="float32"),
            "static_b": np.linspace(1.0, 2.0, len(ids), dtype="float32"),
        }
    )


def _config(tmp_path):
    return {
        "seed": 42,
        "paths": {
            "timeseries_path": str(tmp_path / "timeseries.csv"),
            "timeserieslab_path": str(tmp_path / "timeserieslab.csv"),
            "output_dir": str(tmp_path / "dynamic"),
            "audit_dir": str(tmp_path / "audit"),
            "preprocessor_path": str(tmp_path / "dynamic" / "preprocessor.joblib"),
        },
        "static": {
            "train_path": str(tmp_path / "train.parquet"),
            "val_path": str(tmp_path / "val.parquet"),
            "test_path": str(tmp_path / "test.parquet"),
        },
        "temporal": {
            "hours": 72,
            "max_offset_minutes_exclusive": 4320,
            "temporal_feature_min_patient_coverage": 0.5,
            "chunksize": 3,
            "aggregation": "last_measurement_within_hour",
            "imputation": "forward_fill_then_train_median",
            "scaling": "train_p05_p95_to_unit_interval",
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        "columns": {
            "chart": {"offset_col": "chartoffset", "feature_col": "chartvaluelabel", "value_col": "chartvalue"},
            "lab": {"offset_col": "labresultoffset", "feature_col": "labname", "value_col": "labresult"},
        },
    }


def test_dynamic_landmark_build_uses_static_ids_and_strict_72h_filter(tmp_path):
    train = _static_df([101, 102], "train")
    val = _static_df([201], "validation")
    test = _static_df([301], "test")
    train.to_parquet(tmp_path / "train.parquet", index=False)
    val.to_parquet(tmp_path / "val.parquet", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False)

    pd.DataFrame(
        {
            "patientunitstayid": [101, 101, 101, 102, 201, 301, 999],
            "chartoffset": [10, 50, 4320, 61, 70, -1, 10],
            "chartvaluelabel": ["HR", "HR", "HR", "HR", "HR", "HR", "HR"],
            "chartvalue": [1.0, 3.0, 99.0, 5.0, 7.0, 9.0, 11.0],
        }
    ).to_csv(tmp_path / "timeseries.csv", index=False)
    pd.DataFrame(
        {
            "patientunitstayid": [101, 102, 201, 301],
            "labresultoffset": [30, 30, 30, 30],
            "labname": ["Creatinine", "Creatinine", "Creatinine", "UnusedOnlyNonTrain"],
            "labresult": [2.0, 4.0, 6.0, 8.0],
        }
    ).to_csv(tmp_path / "timeserieslab.csv", index=False)

    arrays, summary = build_landmark_dynamic_dataset(_config(tmp_path), _Logger(), dry_run=True)

    assert arrays["train"]["X_seq"].shape == (2, 72, 2)
    assert arrays["train"]["M_seq"].shape == arrays["train"]["X_seq"].shape
    assert arrays["train"]["X_static"].shape == (2, 2)
    assert not np.isnan(arrays["train"]["X_seq"]).any()
    assert np.isin(arrays["train"]["M_seq"], [0.0, 1.0]).all()
    assert arrays["train"]["patient_ids"].tolist() == [101, 102]
    assert summary["checks"]["ids_match_static_landmark_exact_order"] is True
    assert summary["checks"]["no_offset_minutes_ge_4320_used"] is True
    assert set(summary["temporal_features"]) == {"chart::HR", "lab::Creatinine"}

    hr_idx = summary["temporal_features"].index("chart::HR")
    assert arrays["train"]["M_seq"][0, 0, hr_idx] == 1.0
    assert arrays["train"]["M_seq"][0, 71, hr_idx] == 0.0


def test_dynamic_landmark_write_outputs(tmp_path):
    train = _static_df([101, 102], "train")
    val = _static_df([201], "validation")
    test = _static_df([301], "test")
    train.to_parquet(tmp_path / "train.parquet", index=False)
    val.to_parquet(tmp_path / "val.parquet", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False)
    pd.DataFrame(
        {
            "patientunitstayid": [101, 102, 201, 301],
            "chartoffset": [10, 10, 10, 10],
            "chartvaluelabel": ["HR", "HR", "HR", "HR"],
            "chartvalue": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(tmp_path / "timeseries.csv", index=False)
    pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"]).to_csv(tmp_path / "timeserieslab.csv", index=False)

    build_landmark_dynamic_dataset(_config(tmp_path), _Logger(), force=True, dry_run=False)

    assert (tmp_path / "dynamic" / "train_dynamic_landmark.npz").exists()
    assert (tmp_path / "dynamic" / "val_dynamic_landmark.npz").exists()
    assert (tmp_path / "dynamic" / "test_dynamic_landmark.npz").exists()
    assert (tmp_path / "audit" / "dynamic_landmark_data_audit.json").exists()
    loaded = np.load(tmp_path / "dynamic" / "train_dynamic_landmark.npz")
    assert loaded["X_seq"].shape == (2, 72, 1)
