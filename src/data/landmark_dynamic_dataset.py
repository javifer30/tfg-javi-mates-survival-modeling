"""
Build landmark dynamic tensors from the static landmark cohort and MIMIC tables.

The split, cohort and targets are inherited from the corresponding static
landmark dataset. Temporal features use only measurements with
0 <= offset < landmark_hours and train-only feature selection, imputation and
scaling.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "patientunitstayid"
SPLIT_COL = "split"
DURATION_EVAL_COL = "duration_eval_days"
DURATION_REL_COL = "duration_rel_days"
EVENT_COL = "event_eval"
STATIC_META_COLS = {
    ID_COL,
    SPLIT_COL,
    DURATION_EVAL_COL,
    DURATION_REL_COL,
    EVENT_COL,
    "duration_from_admission_days",
    "observed_event_from_admission",
}


@dataclass
class LandmarkDynamicPreprocessor:
    temporal_features: list
    train_medians: dict
    train_p05: dict
    train_p95: dict
    scaling: str
    imputation: str
    aggregation: str
    min_patient_coverage: float


def load_landmark_static_splits(config):
    static_cfg = config["static"]
    splits = {
        "train": pd.read_parquet(static_cfg["train_path"]),
        "validation": pd.read_parquet(static_cfg["val_path"]),
        "test": pd.read_parquet(static_cfg["test_path"]),
    }
    return splits


def static_feature_columns(df):
    return [col for col in df.columns if col not in STATIC_META_COLS]


def validate_static_reference(splits):
    ids = {name: set(df[ID_COL].astype(str)) for name, df in splits.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("Overlap detected between static_landmark split IDs")
    columns = static_feature_columns(splits["train"])
    for split_name, df in splits.items():
        if static_feature_columns(df) != columns:
            raise ValueError(f"{split_name} static feature columns differ from train")
        if not (df[DURATION_REL_COL] > 0).all():
            raise ValueError(f"{split_name} contains duration_rel_days <= 0")
        if not ((df[DURATION_EVAL_COL] >= 0) & (df[DURATION_EVAL_COL] <= 10)).all():
            raise ValueError(f"{split_name} duration_eval_days outside [0, 10]")
        if not set(df[EVENT_COL].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{split_name} event_eval must be binary")
        if df[columns].isna().any().any():
            raise ValueError(f"{split_name} static features contain NaNs")


def _read_temporal_source(path, source, column_cfg, id_set, max_offset, chunksize, sample_ids=None):
    usecols = [ID_COL, column_cfg["offset_col"], column_cfg["feature_col"], column_cfg["value_col"]]
    frames = []
    min_offset = None
    max_used_offset = None
    max_excluded_offset = None
    rows_seen = 0
    rows_kept = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        rows_seen += int(len(chunk))
        chunk = chunk.rename(
            columns={
                column_cfg["offset_col"]: "offset_minutes",
                column_cfg["feature_col"]: "feature",
                column_cfg["value_col"]: "value",
            }
        )
        chunk[ID_COL] = chunk[ID_COL].astype(str)
        chunk = chunk[chunk[ID_COL].isin(id_set)]
        if sample_ids is not None:
            chunk = chunk[chunk[ID_COL].isin(sample_ids)]
        if chunk.empty:
            continue
        chunk["offset_minutes"] = pd.to_numeric(chunk["offset_minutes"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["offset_minutes", "feature", "value"])
        if chunk.empty:
            continue
        excluded = chunk.loc[chunk["offset_minutes"] >= max_offset, "offset_minutes"]
        if not excluded.empty:
            value = float(excluded.max())
            max_excluded_offset = value if max_excluded_offset is None else max(max_excluded_offset, value)
        chunk = chunk[(chunk["offset_minutes"] >= 0) & (chunk["offset_minutes"] < max_offset)].copy()
        if chunk.empty:
            continue
        min_value = float(chunk["offset_minutes"].min())
        max_value = float(chunk["offset_minutes"].max())
        min_offset = min_value if min_offset is None else min(min_offset, min_value)
        max_used_offset = max_value if max_used_offset is None else max(max_used_offset, max_value)
        chunk["hour"] = np.floor(chunk["offset_minutes"] / 60).astype("int16")
        chunk["source"] = source
        frames.append(chunk[[ID_COL, "offset_minutes", "hour", "feature", "value", "source"]])
        rows_kept += int(len(chunk))
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[ID_COL, "offset_minutes", "hour", "feature", "value", "source"])
    stats = {
        "source": source,
        "rows_seen": rows_seen,
        "rows_kept_after_id_and_landmark_filter": rows_kept,
        "min_used_offset_minutes": min_offset,
        "max_used_offset_minutes": max_used_offset,
        "max_excluded_offset_minutes": max_excluded_offset,
    }
    return data, stats


def load_temporal_measurements(config, id_order, sample_ids=None):
    temporal_cfg = config["temporal"]
    columns = config["columns"]
    max_offset = int(temporal_cfg["max_offset_minutes_exclusive"])
    chunksize = int(temporal_cfg.get("chunksize", 500000))
    id_set = set(id_order)
    chart, chart_stats = _read_temporal_source(
        config["paths"]["timeseries_path"],
        "chart",
        columns["chart"],
        id_set,
        max_offset,
        chunksize,
        sample_ids=sample_ids,
    )
    lab, lab_stats = _read_temporal_source(
        config["paths"]["timeserieslab_path"],
        "lab",
        columns["lab"],
        id_set,
        max_offset,
        chunksize,
        sample_ids=sample_ids,
    )
    frames = [frame for frame in [chart, lab] if not frame.empty]
    temporal = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[ID_COL, "offset_minutes", "hour", "feature", "value", "source"])
    if not temporal.empty:
        temporal["feature"] = temporal["source"].astype(str) + "::" + temporal["feature"].astype(str)
        temporal = temporal.sort_values([ID_COL, "feature", "hour", "offset_minutes"])
        temporal = temporal.drop_duplicates([ID_COL, "feature", "hour"], keep="last")
    return temporal, {"sources": [chart_stats, lab_stats]}


def temporal_feature_coverage(temporal, split_ids, features=None):
    rows = []
    features = sorted(features if features is not None else temporal["feature"].dropna().unique().tolist())
    for split_name, ids in split_ids.items():
        split_temporal = temporal[temporal[ID_COL].isin(ids)]
        denominator = max(len(ids), 1)
        counts = split_temporal.groupby("feature")[ID_COL].nunique()
        for feature in features:
            count = int(counts.get(feature, 0))
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "n_patients_observed": count,
                    "patient_coverage": float(count / denominator),
                }
            )
    return pd.DataFrame(rows)


def select_temporal_features(temporal, train_ids, min_coverage):
    train_temporal = temporal[temporal[ID_COL].isin(train_ids)]
    denominator = max(len(train_ids), 1)
    coverage = train_temporal.groupby("feature")[ID_COL].nunique().sort_index() / denominator
    selected = coverage[coverage >= float(min_coverage)].index.tolist()
    return selected, coverage


def _fit_temporal_stats(raw_train_tensor, features):
    medians = {}
    p05 = {}
    p95 = {}
    for idx, feature in enumerate(features):
        values = raw_train_tensor[:, :, idx].reshape(-1)
        values = values[~np.isnan(values)]
        if values.size == 0:
            median = 0.0
            q05 = 0.0
            q95 = 1.0
        else:
            median = float(np.median(values))
            q05 = float(np.percentile(values, 5))
            q95 = float(np.percentile(values, 95))
            if q95 == q05:
                q95 = q05 + 1.0
        medians[feature] = median
        p05[feature] = q05
        p95[feature] = q95
    return medians, p05, p95


def _impute_and_scale(raw_tensor, features, medians, p05, p95, clip_min, clip_max):
    tensor = raw_tensor.copy().astype("float32")
    for feature_idx, feature in enumerate(features):
        values = pd.DataFrame(tensor[:, :, feature_idx]).ffill(axis=1).to_numpy(dtype="float32")
        values = np.where(np.isnan(values), medians[feature], values)
        scaled = 2.0 * (values - p05[feature]) / (p95[feature] - p05[feature]) - 1.0
        tensor[:, :, feature_idx] = np.clip(scaled, clip_min, clip_max)
    return tensor.astype("float32")


def _build_raw_tensor(temporal, ids, features, hours):
    raw = np.full((len(ids), hours, len(features)), np.nan, dtype="float32")
    mask = np.zeros((len(ids), hours, len(features)), dtype="float32")
    if temporal.empty or not features or not ids:
        return raw, mask
    id_to_idx = {patient_id: idx for idx, patient_id in enumerate(ids)}
    feature_to_idx = {feature: idx for idx, feature in enumerate(features)}
    rows = temporal[temporal[ID_COL].isin(id_to_idx) & temporal["feature"].isin(feature_to_idx)]
    if rows.empty:
        return raw, mask
    patient_idx = rows[ID_COL].map(id_to_idx).to_numpy(dtype=int)
    hour_idx = rows["hour"].to_numpy(dtype=int)
    feature_idx = rows["feature"].map(feature_to_idx).to_numpy(dtype=int)
    raw[patient_idx, hour_idx, feature_idx] = rows["value"].to_numpy(dtype="float32")
    mask[patient_idx, hour_idx, feature_idx] = 1.0
    return raw, mask


def build_dynamic_split(split_name, static_df, temporal, features, preprocessor, config):
    hours = int(config["temporal"]["hours"])
    ids = static_df[ID_COL].astype(str).tolist()
    raw, mask = _build_raw_tensor(temporal, ids, features, hours)
    tensor = _impute_and_scale(
        raw,
        features,
        preprocessor.train_medians,
        preprocessor.train_p05,
        preprocessor.train_p95,
        float(config["temporal"].get("clip_min", -5.0)),
        float(config["temporal"].get("clip_max", 5.0)),
    )
    static_cols = static_feature_columns(static_df)
    return {
        "patient_ids": static_df[ID_COL].to_numpy(),
        "X_seq": tensor,
        "M_seq": mask.astype("float32"),
        "X_static": static_df[static_cols].to_numpy(dtype="float32"),
        "duration_eval_days": static_df[DURATION_EVAL_COL].to_numpy(dtype="float32"),
        "duration_rel_days": static_df[DURATION_REL_COL].to_numpy(dtype="float32"),
        "event_eval": static_df[EVENT_COL].to_numpy(dtype="int64"),
    }, raw, mask


def validate_dynamic_arrays(split_name, arrays, static_df, features, hours):
    n = len(static_df)
    expected_shape = (n, hours, len(features))
    if arrays["X_seq"].shape != expected_shape:
        raise ValueError(f"{split_name} X_seq shape {arrays['X_seq'].shape} != {expected_shape}")
    if arrays["M_seq"].shape != expected_shape:
        raise ValueError(f"{split_name} M_seq shape {arrays['M_seq'].shape} != {expected_shape}")
    if np.isnan(arrays["X_seq"]).any():
        raise ValueError(f"{split_name} X_seq contains NaNs")
    if not np.isin(arrays["M_seq"], [0.0, 1.0]).all():
        raise ValueError(f"{split_name} M_seq must be binary")
    if arrays["X_static"].shape[0] != n:
        raise ValueError(f"{split_name} X_static row count mismatch")
    if not ((arrays["duration_eval_days"] >= 0) & (arrays["duration_eval_days"] <= 10)).all():
        raise ValueError(f"{split_name} duration_eval_days outside [0, 10]")
    if not (arrays["duration_rel_days"] > 0).all():
        raise ValueError(f"{split_name} duration_rel_days must be positive")
    if not np.isin(arrays["event_eval"], [0, 1]).all():
        raise ValueError(f"{split_name} event_eval must be binary")
    static_ids = static_df[ID_COL].to_numpy()
    if not np.array_equal(arrays["patient_ids"], static_ids):
        raise ValueError(f"{split_name} dynamic IDs do not match static_landmark order")


def _missingness_by_feature(raw_by_split, features):
    rows = []
    for split_name, raw in raw_by_split.items():
        total = raw.shape[0] * raw.shape[1]
        for idx, feature in enumerate(features):
            missing = int(np.isnan(raw[:, :, idx]).sum())
            rows.append({"split": split_name, "feature": feature, "missing_fraction_before_imputation": float(missing / total)})
    return pd.DataFrame(rows)


def _hourly_missingness(raw_by_split):
    rows = []
    for split_name, raw in raw_by_split.items():
        denom = raw.shape[0] * raw.shape[2]
        for hour in range(raw.shape[1]):
            missing = int(np.isnan(raw[:, hour, :]).sum())
            rows.append({"split": split_name, "hour": hour, "missing_fraction_before_imputation": float(missing / denom)})
    return pd.DataFrame(rows)


def _patient_any_measurement(mask_by_split):
    rows = []
    for split_name, mask in mask_by_split.items():
        any_measurement = mask.sum(axis=(1, 2)) > 0
        rows.append(
            {
                "split": split_name,
                "n_patients": int(mask.shape[0]),
                "n_patients_with_any_temporal_measurement": int(any_measurement.sum()),
                "share_with_any_temporal_measurement": float(any_measurement.mean()) if len(any_measurement) else 0.0,
            }
        )
    return rows


def _summary(splits, arrays_by_split, features, static_features, temporal_stats, mask_by_split, raw_by_split):
    hours = int(next(iter(arrays_by_split.values()))["X_seq"].shape[1]) if arrays_by_split else 0
    result = {
        "dataset": f"dynamic_{hours}h" if hours else "dynamic_landmark",
        "landmark_hours": hours,
        "n_temporal_features": len(features),
        "n_static_features": len(static_features),
        "temporal_features": features,
        "static_features": static_features,
        "temporal_source_stats": temporal_stats,
        "splits": {},
        "patient_temporal_measurement_coverage": _patient_any_measurement(mask_by_split),
    }
    for split_name, static_df in splits.items():
        arrays = arrays_by_split[split_name]
        result["splits"][split_name] = {
            "n_patients": int(len(static_df)),
            "X_seq_shape": list(arrays["X_seq"].shape),
            "M_seq_shape": list(arrays["M_seq"].shape),
            "X_static_shape": list(arrays["X_static"].shape),
            "event_rate": float(arrays["event_eval"].mean()) if len(arrays["event_eval"]) else 0.0,
            "min_duration_eval_days": float(arrays["duration_eval_days"].min()),
            "max_duration_eval_days": float(arrays["duration_eval_days"].max()),
            "min_duration_rel_days": float(arrays["duration_rel_days"].min()),
            "max_duration_rel_days": float(arrays["duration_rel_days"].max()),
            "observed_temporal_fraction": float(mask_by_split[split_name].mean()) if mask_by_split[split_name].size else 0.0,
            "raw_nan_fraction_before_imputation": float(np.isnan(raw_by_split[split_name]).mean()) if raw_by_split[split_name].size else 0.0,
        }
    return result


def write_outputs(config, arrays_by_split, raw_by_split, mask_by_split, features, static_features, coverage_df, preprocessor, summary):
    output_dir = Path(config["paths"]["output_dir"])
    audit_dir = Path(config["paths"]["audit_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    suffix = config.get("output_file_suffix", "dynamic_landmark")
    for split_name, arrays in arrays_by_split.items():
        file_split = "val" if split_name == "validation" else split_name
        file_name = f"{file_split}_{suffix}.npz"
        np.savez_compressed(output_dir / file_name, **arrays)

    (output_dir / "temporal_feature_columns.json").write_text(
        json.dumps({"temporal_features": features, "coverage": coverage_df.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "static_feature_columns.json").write_text(json.dumps({"static_features": static_features}, indent=2), encoding="utf-8")
    metadata = {
        "aggregation": config["temporal"]["aggregation"],
        "imputation": config["temporal"]["imputation"],
        "scaling": config["temporal"]["scaling"],
        "feature_selection": {
            "split": "train",
            "temporal_feature_min_patient_coverage": float(config["temporal"]["temporal_feature_min_patient_coverage"]),
        },
        "hours": int(config["temporal"]["hours"]),
        "max_offset_minutes_exclusive": int(config["temporal"]["max_offset_minutes_exclusive"]),
        "delta_seq_implemented": False,
        "delta_seq_todo": "Add time-since-last-observation channels if required by DySurv/Dynamic-DeepHit training.",
    }
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(config["paths"]["preprocessor_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config["paths"]["preprocessor_path"])
    summary_name = f"{suffix}_dataset_summary.json"
    (output_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _missingness_by_feature(raw_by_split, features).to_csv(audit_dir / "missingness_summary.csv", index=False)
    _hourly_missingness(raw_by_split).to_csv(audit_dir / "hourly_missingness_summary.csv", index=False)
    coverage_df.to_csv(audit_dir / "feature_coverage_by_split.csv", index=False)
    coverage_df.to_csv(audit_dir / "temporal_coverage_summary.csv", index=False)
    (audit_dir / f"{suffix}_data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _sample_splits(splits, sample_size):
    if sample_size is None:
        return splits
    sampled = {}
    for split_name, df in splits.items():
        sampled[split_name] = df.head(int(sample_size)).copy()
    return sampled


def build_landmark_dynamic_dataset(config, logger, force=False, dry_run=False, sample_size=None):
    output_dir = Path(config["paths"]["output_dir"])
    suffix = config.get("output_file_suffix", "dynamic_landmark")
    if output_dir.exists() and any(output_dir.glob(f"*_{suffix}.npz")) and not force and not dry_run:
        raise FileExistsError(f"Dynamic landmark outputs already exist in {output_dir}. Use --force to overwrite.")

    splits = load_landmark_static_splits(config)
    validate_static_reference(splits)
    splits = _sample_splits(splits, sample_size)
    split_ids = {name: set(df[ID_COL].astype(str)) for name, df in splits.items()}
    id_order = pd.concat([df[ID_COL].astype(str) for df in splits.values()], ignore_index=True).tolist()

    temporal, temporal_stats = load_temporal_measurements(config, id_order, sample_ids=set(id_order) if sample_size else None)
    if not temporal.empty and not temporal["hour"].between(0, int(config["temporal"]["hours"]) - 1).all():
        raise ValueError("Temporal rows outside hour grid after filtering")
    if not temporal.empty and (temporal["offset_minutes"] >= int(config["temporal"]["max_offset_minutes_exclusive"])).any():
        raise ValueError("Found temporal rows with offset_minutes >= landmark max offset")

    min_cov = float(config["temporal"]["temporal_feature_min_patient_coverage"])
    features, train_coverage = select_temporal_features(temporal, split_ids["train"], min_cov)
    if not features:
        raise ValueError("No temporal features selected from train coverage")
    temporal = temporal[temporal["feature"].isin(features)].copy()
    coverage_df = temporal_feature_coverage(temporal, split_ids, features=features)
    train_selected_coverage = train_coverage.loc[features]
    train_coverage_df = pd.DataFrame(
        {
            "split": "train",
            "feature": train_selected_coverage.index,
            "n_patients_observed": (train_selected_coverage * max(len(split_ids["train"]), 1)).astype(int).values,
            "patient_coverage": train_selected_coverage.values,
            "selected_train_only": True,
        }
    )
    coverage_df = pd.concat([coverage_df, train_coverage_df], ignore_index=True).drop_duplicates(["split", "feature"], keep="last")
    hours = int(config["temporal"]["hours"])

    train_raw, train_mask = _build_raw_tensor(temporal, splits["train"][ID_COL].astype(str).tolist(), features, hours)
    medians, p05, p95 = _fit_temporal_stats(train_raw, features)
    preprocessor = LandmarkDynamicPreprocessor(
        temporal_features=features,
        train_medians=medians,
        train_p05=p05,
        train_p95=p95,
        scaling=config["temporal"]["scaling"],
        imputation=config["temporal"]["imputation"],
        aggregation=config["temporal"]["aggregation"],
        min_patient_coverage=min_cov,
    )

    arrays_by_split = {}
    raw_by_split = {}
    mask_by_split = {}
    for split_name, static_df in splits.items():
        arrays, raw, mask = build_dynamic_split(split_name, static_df, temporal, features, preprocessor, config)
        validate_dynamic_arrays(split_name, arrays, static_df, features, hours)
        arrays_by_split[split_name] = arrays
        raw_by_split[split_name] = raw
        mask_by_split[split_name] = mask

    static_features = static_feature_columns(splits["train"])
    summary = _summary(splits, arrays_by_split, features, static_features, temporal_stats, mask_by_split, raw_by_split)
    summary["checks"] = {
        "ids_match_static_landmark_exact_order": True,
        "ids_match_static_landmark_exact_order": True,
        "no_overlap_between_splits": True,
        "all_duration_rel_days_positive": True,
        "no_offset_minutes_ge_landmark_used": True,
        "no_offset_minutes_ge_4320_used": int(config["temporal"]["max_offset_minutes_exclusive"]) == 4320,
        "feature_selection_split": "train",
        "imputation_fit_split": "train",
        "scaling_fit_split": "train",
        "delta_seq_implemented": False,
    }

    logger.info("dynamic landmark selected %d temporal features and %d static features", len(features), len(static_features))
    for split_name, arrays in arrays_by_split.items():
        logger.info(
            "%s shapes: X_seq=%s M_seq=%s X_static=%s event_rate=%.4f",
            split_name,
            arrays["X_seq"].shape,
            arrays["M_seq"].shape,
            arrays["X_static"].shape,
            float(arrays["event_eval"].mean()),
        )
    if dry_run:
        return arrays_by_split, summary

    write_outputs(config, arrays_by_split, raw_by_split, mask_by_split, features, static_features, coverage_df, preprocessor, summary)
    return arrays_by_split, summary
