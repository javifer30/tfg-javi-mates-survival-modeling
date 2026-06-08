import math

import pytest

from scripts.run_final_static_seeds import FINAL_SEEDS, _validate_final_seeds
from scripts.tune_static_models import expand_grid, prepare_run_config, select_best_row


def _base_config():
    return {
        "seed": 42,
        "paths": {
            "train_path": "data/processed/static/train_static.parquet",
            "val_path": "data/processed/static/val_static.parquet",
            "test_path": "data/processed/static/test_static.parquet",
        },
        "model": {
            "name": "deephit",
            "num_Category": 10,
            "max_horizon_days": 10,
            "save_best_checkpoint": True,
            "save_last_checkpoint": True,
            "save_every_n_epochs": 5,
        },
    }


def test_expand_grid_is_deterministic():
    grid = {"penalizer": [0.01, 0.1], "l1_ratio": [0.0, 0.5]}
    assert expand_grid(grid) == [
        {"penalizer": 0.01, "l1_ratio": 0.0},
        {"penalizer": 0.01, "l1_ratio": 0.5},
        {"penalizer": 0.1, "l1_ratio": 0.0},
        {"penalizer": 0.1, "l1_ratio": 0.5},
    ]


def test_tuning_run_config_disables_test_metrics_and_heavy_artifacts():
    run_config, run_dir = prepare_run_config(
        _base_config(),
        "deephit",
        "deephit_cfg_001",
        {"learning_rate": 0.001},
        42,
        "outputs/tuning",
        phase="tuning",
        include_test=False,
        save_predictions=False,
        save_models=False,
        save_checkpoints=False,
    )
    assert run_dir.as_posix().endswith("outputs/tuning/deephit/deephit_cfg_001/seed_42")
    assert run_config["evaluation"]["splits"] == ["train", "validation"]
    assert run_config["evaluation"]["allow_test_metrics"] is False
    assert run_config["evaluation"]["save_predictions"] is False
    assert run_config["model"]["save_best_checkpoint"] is False
    assert run_config["model"]["save_last_checkpoint"] is False
    assert run_config["model"]["save_every_n_epochs"] is None
    assert run_config["model"]["evaluation_time_grid"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_select_best_uses_validation_ctd_then_ibll():
    rows = [
        {"config_id": "cfg_a", "seed": 42, "validation_ctd_antolini": 0.75, "validation_ibll": 0.40},
        {"config_id": "cfg_b", "seed": 42, "validation_ctd_antolini": 0.76, "validation_ibll": 0.55},
        {"config_id": "cfg_c", "seed": 42, "validation_ctd_antolini": 0.76, "validation_ibll": 0.35},
    ]
    assert select_best_row(rows)["config_id"] == "cfg_c"


def test_select_best_treats_missing_ctd_as_worst():
    rows = [
        {"config_id": "cfg_a", "seed": 42, "validation_ctd_antolini": math.nan, "validation_ibll": 0.20},
        {"config_id": "cfg_b", "seed": 42, "validation_ctd_antolini": 0.70, "validation_ibll": 0.50},
    ]
    assert select_best_row(rows)["config_id"] == "cfg_b"


def test_final_seed_contract_is_exact():
    _validate_final_seeds(FINAL_SEEDS)
    with pytest.raises(ValueError):
        _validate_final_seeds([42, 123])
