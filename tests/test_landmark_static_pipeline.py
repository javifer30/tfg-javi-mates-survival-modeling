import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.landmark_static_tuning_impl import expand_grid, select_best_row, tune_models
from scripts.landmark_static_final_impl import run_final_models
from src.data.landmark_static_dataset import (
    DURATION_COL,
    EVENT_COL,
    ID_COL,
    SPLIT_COL,
    LandmarkStaticPreprocessor,
    RAW_EVENT_COL,
    REL_DURATION_DAYS_COL,
    load_landmark_static_base_table,
    validate_landmark_static_datasets,
)
from src.evaluation.landmark_survival_metrics import eval_surv_metrics, horizon_c_index
from src.models.landmark_static_pycox import (
    rsf_survival_dataframe,
    rsf_target,
    train_random_survival_forest,
)
from src.utils.landmark import apply_landmark_static_tuning_config


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


def _synthetic_rsf_data(n_samples=72, seed=7):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "age_scaled": rng.normal(size=n_samples),
            "severity_scaled": rng.normal(size=n_samples),
            "careunit_indicator": rng.integers(0, 2, size=n_samples).astype(float),
        },
        index=np.arange(1000, 1000 + n_samples),
    )
    signal = 1.5 * x["severity_scaled"].to_numpy() - 0.5 * x["age_scaled"].to_numpy()
    durations = np.clip(5.5 - signal + rng.normal(scale=0.5, size=n_samples), 0.5, 10.0)
    durations[::11] = 10.0
    events = (rng.random(n_samples) < 0.75).astype(int)
    events[durations >= 10.0] = 0
    return x, durations, events


def test_rsf_target_uses_boolean_event_and_float_time():
    target = rsf_target([1.0, 3.0, 10.0], [1, 0, 1])

    assert target.dtype.names == ("event", "time")
    assert target["event"].dtype == np.dtype(bool)
    assert target["event"].tolist() == [True, False, True]
    assert np.allclose(target["time"], [1.0, 3.0, 10.0])


def test_rsf_survival_orientation_monotonicity_and_reproducibility():
    from sksurv.ensemble import RandomSurvivalForest

    x, durations, events = _synthetic_rsf_data()
    target = rsf_target(durations, events)
    params = {
        "n_estimators": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "low_memory": False,
        "n_jobs": 1,
        "random_state": 42,
    }
    first = RandomSurvivalForest(**params).fit(x, target)
    second = RandomSurvivalForest(**params).fit(x, target)
    times = np.arange(1.0, 11.0)

    first_survival = rsf_survival_dataframe(first, x, times, batch_size=7)
    second_survival = rsf_survival_dataframe(second, x, times, batch_size=len(x))

    assert first_survival.shape == (10, len(x))
    assert first_survival.index.tolist() == times.tolist()
    assert first_survival.columns.tolist() == x.index.tolist()
    assert np.isfinite(first_survival.to_numpy()).all()
    assert ((first_survival >= 0.0) & (first_survival <= 1.0)).all().all()
    assert (np.diff(first_survival.to_numpy(), axis=0) <= 1e-8).all()
    assert first_survival.iloc[-1].round(8).nunique() > 1
    np.testing.assert_allclose(first_survival, second_survival, rtol=0.0, atol=0.0)


def test_rsf_prediction_batches_never_exceed_configured_size():
    from sksurv.ensemble import RandomSurvivalForest

    x, durations, events = _synthetic_rsf_data(n_samples=29)
    fitted = RandomSurvivalForest(
        n_estimators=4,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        low_memory=False,
        n_jobs=1,
        random_state=42,
    ).fit(x, rsf_target(durations, events))

    class RecordingModel:
        def __init__(self, model):
            self.model = model
            self.batch_sizes = []

        def predict_survival_function(self, x_batch):
            self.batch_sizes.append(len(x_batch))
            return self.model.predict_survival_function(x_batch)

    recording_model = RecordingModel(fitted)
    survival = rsf_survival_dataframe(recording_model, x, np.arange(1.0, 11.0), batch_size=8)

    assert survival.shape == (10, len(x))
    assert recording_model.batch_sizes == [8, 8, 8, 5]


def test_rsf_training_uses_validation_only_when_test_is_disabled(tmp_path):
    x, durations, events = _synthetic_rsf_data(n_samples=90)
    frame = x.reset_index(names=ID_COL)
    frame[DURATION_COL] = durations
    frame[EVENT_COL] = events
    train = frame.iloc[:60].copy()
    validation = frame.iloc[60:].copy()
    train[SPLIT_COL] = "train"
    validation[SPLIT_COL] = "validation"

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    train.to_parquet(processed_dir / "train_static_test.parquet", index=False)
    validation.to_parquet(processed_dir / "val_static_test.parquet", index=False)
    output_dir = tmp_path / "outputs" / "tuning" / "random_survival_forest" / "candidate"
    config = {
        "seed": 42,
        "experiment": {
            "max_horizon_days": 10,
            "time_unit": "days_since_landmark",
        },
        "paths": {
            "processed_dir": str(processed_dir),
            "outputs_dir": str(output_dir),
            "static_file_suffix": "static_test",
        },
        "model": {
            "name": "random_survival_forest",
            "n_estimators": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": 1,
            "low_memory": False,
            "prediction_batch_size": 8,
            "save_model": False,
        },
        "evaluation": {
            "splits": ["train", "validation"],
            "allow_test_metrics": False,
            "metric_integration_num_points": 20,
            "horizon_times": list(range(1, 11)),
            "save_example_curves": False,
        },
    }

    metrics = train_random_survival_forest(config, _logger())

    assert set(metrics["splits"]) == {"validation"}
    assert math.isfinite(metrics["splits"]["validation"]["ctd_antolini"])
    metrics_path = output_dir / "metrics" / "random_survival_forest" / "random_survival_forest_metrics.json"
    assert metrics_path.exists()


def test_rsf_grid_and_landmark_dry_runs_are_stable(tmp_path):
    base_config = yaml.safe_load(Path("configs/landmark_static_tuning.yaml").read_text(encoding="utf-8"))
    grid = base_config["models"]["random_survival_forest"]["grid"]
    assert len(expand_grid(grid)) == 8

    for hours in (24, 48, 72):
        config = apply_landmark_static_tuning_config(base_config, hours)
        config["paths"]["outputs_dir"] = str(tmp_path / f"landmark_{hours}h" / "static")
        config_path = tmp_path / f"rsf_{hours}h.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        planned = tune_models(
            str(config_path),
            requested_models=["random_survival_forest"],
            dry_run=True,
            max_runs=1,
        )

        assert len(planned) == 1
        assert planned[0]["model"] == "random_survival_forest"
        assert planned[0]["config_id"] == "random_survival_forest_cfg_001"
        assert f"landmark_{hours}h" in planned[0]["output_dir"]


def test_rsf_resume_skips_completed_hyperparameters(tmp_path):
    config = yaml.safe_load(Path("configs/landmark_static_tuning.yaml").read_text(encoding="utf-8"))
    config["paths"]["outputs_dir"] = str(tmp_path / "static")
    config_path = tmp_path / "rsf_resume.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    first_params = expand_grid(config["models"]["random_survival_forest"]["grid"])[0]
    results_dir = tmp_path / "static" / "tuning" / "random_survival_forest"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "completed",
                "hyperparameters": json.dumps(first_params, sort_keys=True),
                "validation_ctd_antolini": 0.7,
                "validation_ibll": 0.4,
                "validation_ibs": 0.2,
                "config_id": "random_survival_forest_cfg_001",
            }
        ]
    ).to_csv(results_dir / "tuning_results.csv", index=False)

    planned = tune_models(
        str(config_path),
        requested_models=["random_survival_forest"],
        dry_run=True,
        max_runs=1,
        resume=True,
    )

    assert len(planned) == 1
    assert planned[0]["config_id"] == "random_survival_forest_cfg_002"


def test_rsf_final_dry_run_uses_exact_three_seeds(tmp_path):
    config = yaml.safe_load(Path("configs/landmark_static_tuning.yaml").read_text(encoding="utf-8"))
    config["paths"]["outputs_dir"] = str(tmp_path / "static")
    config_path = tmp_path / "rsf_final.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    selected_params = expand_grid(config["models"]["random_survival_forest"]["grid"])[0]
    tuning_dir = tmp_path / "static" / "tuning" / "random_survival_forest"
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "best_hyperparameters.json").write_text(
        json.dumps(
            {
                "config_id": "random_survival_forest_cfg_001",
                "hyperparameters": json.dumps(selected_params, sort_keys=True),
            }
        ),
        encoding="utf-8",
    )

    planned = run_final_models(
        str(config_path),
        requested_models=["random_survival_forest"],
        dry_run=True,
    )

    assert [row["seed"] for row in planned] == [42, 123, 2026]
    assert all(row["model"] == "random_survival_forest" for row in planned)
