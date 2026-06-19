"""Helpers for CLI-driven landmark pipeline configuration."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ALLOWED_LANDMARK_HOURS = (24, 48, 72)


def validate_landmark_hours(value: int) -> int:
    hours = int(value)
    if hours not in ALLOWED_LANDMARK_HOURS:
        raise ValueError(f"landmark_hours must be one of {ALLOWED_LANDMARK_HOURS}; got {hours}")
    return hours


def landmark_tag(landmark_hours: int) -> str:
    return f"landmark_{validate_landmark_hours(landmark_hours)}h"


def static_suffix(landmark_hours: int) -> str:
    return f"static_{validate_landmark_hours(landmark_hours)}h"


def dynamic_suffix(landmark_hours: int) -> str:
    return f"dynamic_{validate_landmark_hours(landmark_hours)}h"


def split_files_for_suffix(suffix: str) -> dict[str, str]:
    return {
        "train": f"train_{suffix}.npz",
        "validation": f"val_{suffix}.npz",
        "test": f"test_{suffix}.npz",
    }


def apply_landmark_static_data_config(config: dict, landmark_hours: int) -> dict:
    hours = validate_landmark_hours(landmark_hours)
    cfg = copy.deepcopy(config)
    tag = landmark_tag(hours)
    suffix = static_suffix(hours)
    cfg.setdefault("target", {})["prediction_time_hours"] = hours
    cfg.setdefault("target", {}).setdefault("max_horizon_days", 10)
    cfg.setdefault("paths", {})["processed_dir"] = f"data/processed/{tag}/static"
    cfg["paths"]["preprocessor_path"] = f"outputs/{tag}/static/preprocessors/static_preprocessor.pkl"
    cfg["paths"]["summary_path"] = f"outputs/{tag}/static/metrics/static_dataset_summary.json"
    cfg.setdefault("outputs", {})["config_used_path"] = f"outputs/{tag}/static/config_used.yaml"
    cfg["output_file_suffix"] = suffix
    return cfg


def apply_landmark_static_tuning_config(config: dict, landmark_hours: int) -> dict:
    hours = validate_landmark_hours(landmark_hours)
    cfg = copy.deepcopy(config)
    tag = landmark_tag(hours)
    cfg.setdefault("experiment", {})["name"] = f"static_{hours}h_pycox"
    cfg["experiment"]["prediction_time_hours"] = hours
    cfg["experiment"]["time_unit"] = f"days_since_{hours}h"
    cfg.setdefault("paths", {})["processed_dir"] = f"data/processed/{tag}/static"
    cfg["paths"]["outputs_dir"] = f"outputs/{tag}/static"
    cfg.setdefault("outputs", {})["config_used_path"] = f"outputs/{tag}/static/config_used.yaml"
    cfg["static_file_suffix"] = static_suffix(hours)
    return cfg


def apply_landmark_dynamic_data_config(config: dict, landmark_hours: int) -> dict:
    hours = validate_landmark_hours(landmark_hours)
    cfg = copy.deepcopy(config)
    tag = landmark_tag(hours)
    static_file = static_suffix(hours)
    dynamic_file = dynamic_suffix(hours)
    cfg.setdefault("paths", {})["static_72h_dir"] = f"data/processed/{tag}/static"
    cfg["paths"]["output_dir"] = f"data/processed/{tag}/dynamic"
    cfg["paths"]["audit_dir"] = f"outputs/{tag}/dynamic/audit"
    cfg["paths"]["preprocessor_path"] = f"data/processed/{tag}/dynamic/preprocessor.joblib"
    cfg.setdefault("static", {})["train_path"] = f"data/processed/{tag}/static/train_{static_file}.parquet"
    cfg["static"]["val_path"] = f"data/processed/{tag}/static/val_{static_file}.parquet"
    cfg["static"]["test_path"] = f"data/processed/{tag}/static/test_{static_file}.parquet"
    cfg.setdefault("temporal", {})["prediction_time_hours"] = hours
    cfg["temporal"]["hours"] = hours
    cfg["temporal"]["max_offset_minutes_exclusive"] = hours * 60
    cfg.setdefault("outputs", {})["config_used_path"] = f"outputs/{tag}/dynamic/config_used.yaml"
    cfg["output_file_suffix"] = dynamic_file
    return cfg


def apply_landmark_faithful_config(config: dict, landmark_hours: int, family: str) -> dict:
    hours = validate_landmark_hours(landmark_hours)
    cfg = copy.deepcopy(config)
    tag = landmark_tag(hours)
    experiment_name = family
    cfg.setdefault("experiment", {})["name"] = f"{experiment_name}_{hours}h"
    cfg["experiment"]["prediction_time_hours"] = hours
    cfg["experiment"]["time_unit"] = f"days_since_{hours}h"
    cfg.setdefault("paths", {})
    cfg["paths"]["prepared_dataset_dir"] = f"data/processed/{tag}/faithful"
    cfg["paths"]["outputs_dir"] = f"outputs/{tag}/{family}"
    if family == "dysurv_faithful":
        cfg["paths"]["source_dataset_dir"] = f"data/processed/{tag}/dynamic_dysurv_features"
    cfg.setdefault("data", {})["source_split_files"] = split_files_for_suffix(dynamic_suffix(hours))
    cfg["data"]["output_split_files"] = split_files_for_suffix(dynamic_suffix(hours))
    cfg.setdefault("outputs", {})["config_used_path"] = f"outputs/{tag}/{family}/config_used.yaml"
    return cfg


def save_config_used(config: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
