"""Prepare the isolated DySurv-faithful 72h dataset.

The source arrays already enforce the 72-hour landmark and train-fitted
scaling. Missing entries are restored with ``M_seq`` and then imputed using
the DySurv-like order: within-patient forward fill, backward fill, and a
train-only residual median. Static features are standardized with train-only
statistics. Masks are retained for reconstruction weighting and audits, never
as model input channels.
"""

from __future__ import annotations

import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_FILES = {
    "train": "train_dynamic_landmark.npz",
    "validation": "val_dynamic_landmark.npz",
    "test": "test_dynamic_landmark.npz",
}


def split_files(config: dict | None = None, key: str = "source_split_files") -> dict[str, str]:
    if config is None:
        return SPLIT_FILES
    configured = config.get("data", {}).get(key)
    return configured or SPLIT_FILES


def load_source_split(source_dir: str | Path, split: str, files: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    path = Path(source_dir) / (files or SPLIT_FILES)[split]
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def restore_missing(x_seq: np.ndarray, m_seq: np.ndarray) -> np.ndarray:
    if x_seq.shape != m_seq.shape:
        raise ValueError(f"X_seq and M_seq shapes differ: {x_seq.shape} vs {m_seq.shape}")
    return np.where(m_seq > 0.5, x_seq, np.nan).astype("float32")


def within_patient_fill(raw: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill each patient/feature over 72 hours."""
    filled = np.empty_like(raw, dtype="float32")
    for feature_idx in range(raw.shape[2]):
        values = pd.DataFrame(raw[:, :, feature_idx])
        filled[:, :, feature_idx] = values.ffill(axis=1).bfill(axis=1).to_numpy(dtype="float32")
    return filled


def fit_residual_medians(train_filled: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(train_filled, axis=(0, 1)).astype("float32")
    return np.where(np.isfinite(medians), medians, 0.0).astype("float32")


def fill_residual_missing(filled: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = filled
    missing = np.isnan(result)
    if missing.any():
        result[missing] = np.broadcast_to(medians, result.shape)[missing]
    return result.astype("float32")


def fit_static_standardizer(x_static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_static, axis=0, dtype=np.float64).astype("float32")
    scale = np.std(x_static, axis=0, dtype=np.float64).astype("float32")
    scale[~np.isfinite(scale) | (scale < 1e-7)] = 1.0
    return mean, scale


def transform_static(x_static: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = (x_static.astype("float32") - mean) / scale
    return np.clip(transformed, -5.0, 5.0).astype("float32")


def prepare_arrays(source_splits: dict[str, dict[str, np.ndarray]]) -> tuple[dict, dict]:
    restored = {
        split: restore_missing(arrays["X_seq"], arrays["M_seq"])
        for split, arrays in source_splits.items()
    }
    within_filled = {split: within_patient_fill(values) for split, values in restored.items()}
    temporal_medians = fit_residual_medians(within_filled["train"])
    static_mean, static_scale = fit_static_standardizer(source_splits["train"]["X_static"])

    prepared = {}
    split_summary = {}
    for split, arrays in source_splits.items():
        x_seq = fill_residual_missing(within_filled[split], temporal_medians)
        x_static = transform_static(arrays["X_static"], static_mean, static_scale)
        prepared[split] = {
            "patient_ids": arrays["patient_ids"],
            "X_seq": x_seq,
            "M_seq": arrays["M_seq"].astype("float32", copy=False),
            "X_static": x_static,
            "duration_eval_days": arrays["duration_eval_days"].astype("float32", copy=False),
            "duration_rel_days": arrays["duration_rel_days"].astype("float32", copy=False),
            "event_eval": arrays["event_eval"].astype("int64", copy=False),
        }
        split_summary[split] = {
            "n_patients": int(x_seq.shape[0]),
            "x_seq_shape": list(x_seq.shape),
            "x_static_shape": list(x_static.shape),
            "observed_fraction": float(arrays["M_seq"].mean()),
            "patients_with_residual_fill": int(np.isnan(within_filled[split]).any(axis=(1, 2)).sum()),
            "event_rate": float(arrays["event_eval"].mean()),
        }

    stats = {
        "temporal_residual_medians": temporal_medians.tolist(),
        "static_train_mean": static_mean.tolist(),
        "static_train_scale": static_scale.tolist(),
        "fit_split": "train",
        "split_summary": split_summary,
    }
    return prepared, stats


def validate_prepared(prepared: dict[str, dict[str, np.ndarray]]) -> dict:
    ids = {split: set(map(str, arrays["patient_ids"])) for split, arrays in prepared.items()}
    checks = {
        "train_validation_no_overlap": not bool(ids["train"] & ids["validation"]),
        "train_test_no_overlap": not bool(ids["train"] & ids["test"]),
        "validation_test_no_overlap": not bool(ids["validation"] & ids["test"]),
    }
    for split, arrays in prepared.items():
        if arrays["X_seq"].ndim != 3:
            raise ValueError(f"{split} must have X_seq [N, T, F]")
        if arrays["X_seq"].shape != arrays["M_seq"].shape:
            raise ValueError(f"{split} X_seq/M_seq shape mismatch")
        if not np.isfinite(arrays["X_seq"]).all() or not np.isfinite(arrays["X_static"]).all():
            raise ValueError(f"{split} contains non-finite model inputs")
        if not np.isin(arrays["M_seq"], [0.0, 1.0]).all():
            raise ValueError(f"{split} M_seq must be binary")
        if not np.isin(arrays["event_eval"], [0, 1]).all():
            raise ValueError(f"{split} event_eval must be binary")
        if not ((arrays["duration_eval_days"] >= 0) & (arrays["duration_eval_days"] <= 10)).all():
            raise ValueError(f"{split} duration_eval_days outside [0, 10]")
    if not all(checks.values()):
        raise ValueError(f"Patient overlap detected: {checks}")
    return checks


def write_prepared_dataset(config: dict, prepared: dict, stats: dict, checks: dict) -> None:
    source_dir = Path(config["paths"]["source_dataset_dir"])
    output_dir = Path(config["paths"]["prepared_dataset_dir"])
    output_files = split_files(config, "output_split_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in prepared.items():
        filename = output_files[split]
        np.savez_compressed(output_dir / filename, **arrays)

    for filename in ["temporal_feature_columns.json", "static_feature_columns.json"]:
        source = source_dir / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    landmark_hours = int(config.get("experiment", {}).get("prediction_time_hours", prepared["train"]["X_seq"].shape[1]))
    metadata = {
        "dataset": f"dysurv_faithful_{landmark_hours}h",
        "source_dataset": str(source_dir),
        "landmark_hours": landmark_hours,
        "input_after_landmark": False,
        "temporal_imputation": ["within_patient_forward_fill", "within_patient_backward_fill", "train_residual_median"],
        "static_preprocessing": "train_mean_standardization",
        "mask_as_input": False,
        "mask_retained_for_reconstruction_weighting_only": True,
        "imputation_fit_split": "train",
        "checks": checks,
        **stats,
    }
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_dataset(config: dict, force: bool = False) -> dict:
    output_dir = Path(config["paths"]["prepared_dataset_dir"])
    source_files = split_files(config, "source_split_files")
    output_files = split_files(config, "output_split_files")
    existing = [output_dir / filename for filename in output_files.values()]
    if any(path.exists() for path in existing) and not force:
        raise FileExistsError(f"Faithful dataset already exists in {output_dir}; use --force to replace it")
    source_dir = Path(config["paths"]["source_dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_source = load_source_split(source_dir, "train", source_files)
    train_filled = within_patient_fill(restore_missing(train_source["X_seq"], train_source["M_seq"]))
    temporal_medians = fit_residual_medians(train_filled)
    static_mean, static_scale = fit_static_standardizer(train_source["X_static"])

    split_summary = {}
    id_sets = {}
    for split in output_files:
        source = train_source if split == "train" else load_source_split(source_dir, split, source_files)
        within = train_filled if split == "train" else within_patient_fill(restore_missing(source["X_seq"], source["M_seq"]))
        residual_patients = int(np.isnan(within).any(axis=(1, 2)).sum())
        x_seq = fill_residual_missing(within, temporal_medians)
        arrays = {
            "patient_ids": source["patient_ids"],
            "X_seq": x_seq,
            "M_seq": source["M_seq"].astype("float32", copy=False),
            "X_static": transform_static(source["X_static"], static_mean, static_scale),
            "duration_eval_days": source["duration_eval_days"].astype("float32", copy=False),
            "duration_rel_days": source["duration_rel_days"].astype("float32", copy=False),
            "event_eval": source["event_eval"].astype("int64", copy=False),
        }
        if not np.isfinite(arrays["X_seq"]).all() or not np.isfinite(arrays["X_static"]).all():
            raise ValueError(f"{split} contains non-finite model inputs")
        if arrays["X_seq"].shape != arrays["M_seq"].shape:
            raise ValueError(f"{split} has invalid temporal shapes")
        if not np.isin(arrays["M_seq"], [0.0, 1.0]).all():
            raise ValueError(f"{split} M_seq must be binary")
        if not np.isin(arrays["event_eval"], [0, 1]).all():
            raise ValueError(f"{split} event_eval must be binary")
        if not ((arrays["duration_eval_days"] >= 0) & (arrays["duration_eval_days"] <= 10)).all():
            raise ValueError(f"{split} duration_eval_days outside [0, 10]")
        np.savez_compressed(output_dir / output_files[split], **arrays)
        id_sets[split] = set(map(str, arrays["patient_ids"]))
        split_summary[split] = {
            "n_patients": int(x_seq.shape[0]),
            "x_seq_shape": list(x_seq.shape),
            "x_static_shape": list(arrays["X_static"].shape),
            "observed_fraction": float(arrays["M_seq"].mean()),
            "patients_with_residual_fill": residual_patients,
            "event_rate": float(arrays["event_eval"].mean()),
        }
        if split != "train":
            del source, within, x_seq, arrays
            gc.collect()

    checks = {
        "train_validation_no_overlap": not bool(id_sets["train"] & id_sets["validation"]),
        "train_test_no_overlap": not bool(id_sets["train"] & id_sets["test"]),
        "validation_test_no_overlap": not bool(id_sets["validation"] & id_sets["test"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Patient overlap detected: {checks}")
    stats = {
        "temporal_residual_medians": temporal_medians.tolist(),
        "static_train_mean": static_mean.tolist(),
        "static_train_scale": static_scale.tolist(),
        "fit_split": "train",
        "split_summary": split_summary,
    }
    for filename in ["temporal_feature_columns.json", "static_feature_columns.json"]:
        source = source_dir / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    landmark_hours = int(config.get("experiment", {}).get("prediction_time_hours", train_source["X_seq"].shape[1]))
    metadata = {
        "dataset": f"dysurv_faithful_{landmark_hours}h",
        "source_dataset": str(source_dir),
        "landmark_hours": landmark_hours,
        "input_after_landmark": False,
        "temporal_imputation": ["within_patient_forward_fill", "within_patient_backward_fill", "train_residual_median"],
        "static_preprocessing": "train_mean_standardization",
        "mask_as_input": False,
        "mask_retained_for_reconstruction_weighting_only": True,
        "imputation_fit_split": "train",
        "checks": checks,
        **stats,
    }
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "checks": checks, **split_summary}
