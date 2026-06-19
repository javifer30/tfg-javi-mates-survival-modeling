"""
Build the static_72h_pycox dataset.

This module creates a new static benchmark cohort conditioned on patients still
being observable after 72 hours in ICU. Targets are measured relative to that
prediction time and administratively censored at 10 days.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COL = "patientunitstayid"
SPLIT_COL = "split"
DURATION_COL = "duration_eval_days"
EVENT_COL = "event_eval"
RAW_DURATION_DAYS_COL = "duration_from_admission_days"
RAW_EVENT_COL = "observed_event_from_admission"
REL_DURATION_DAYS_COL = "duration_rel_days"
TARGET_COLS = [
    ID_COL,
    SPLIT_COL,
    DURATION_COL,
    EVENT_COL,
    RAW_DURATION_DAYS_COL,
    RAW_EVENT_COL,
    REL_DURATION_DAYS_COL,
]


@dataclass
class Static72hPreprocessor:
    standard_cols: list
    leave_numeric_cols: list
    categorical_cols: list
    binary_maps: dict
    drop_feature_cols: list
    rare_min_count: int = 1000
    clip_min: float = -5.0
    clip_max: float = 5.0

    def fit(self, df):
        self.available_standard_cols_ = [c for c in self.standard_cols if c in df.columns]
        self.available_leave_numeric_cols_ = [c for c in self.leave_numeric_cols if c in df.columns]
        self.available_categorical_cols_ = [c for c in self.categorical_cols if c in df.columns]
        self.available_binary_cols_ = [c for c in self.binary_maps if c in df.columns]
        self.drop_feature_cols_ = [c for c in self.drop_feature_cols if c in df.columns]

        self.numeric_medians_ = {}
        self.standard_params_ = {}
        for col in self.available_standard_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            if pd.isna(median):
                median = 0.0
            filled = values.fillna(median)
            std = filled.std()
            self.numeric_medians_[col] = float(median)
            self.standard_params_[col] = {
                "mean": float(filled.mean()) if pd.notna(filled.mean()) else 0.0,
                "std": float(std) if pd.notna(std) and std != 0 else 1.0,
            }

        for col in self.available_leave_numeric_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            self.numeric_medians_[col] = float(median) if pd.notna(median) else 0.0

        self.rare_values_ = {}
        self.categories_ = {}
        for col in self.available_categorical_cols_:
            values = df[col].fillna("missing").astype(str)
            counts = values.value_counts(dropna=False)
            self.rare_values_[col] = set(counts[counts < self.rare_min_count].index)
            grouped = values.mask(values.isin(self.rare_values_[col]), "misc")
            self.categories_[col] = sorted(grouped.unique().tolist())

        self.feature_cols_ = self._feature_names()
        return self

    def transform(self, df, split_name):
        out = pd.DataFrame(index=df.index)
        for col in TARGET_COLS:
            if col in df.columns:
                out[col] = df[col].values
        out[SPLIT_COL] = split_name

        features = pd.DataFrame(index=df.index)
        for col in self.available_binary_cols_:
            features[col] = df[col].map(self.binary_maps[col]).fillna(0.5).astype(float)

        for col in self.available_standard_cols_:
            values = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_medians_[col])
            params = self.standard_params_[col]
            scaled = (values - params["mean"]) / params["std"]
            features[col] = scaled.clip(self.clip_min, self.clip_max).astype(float)

        for col in self.available_leave_numeric_cols_:
            values = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_medians_[col])
            features[col] = values.astype(float)

        for col in self.available_categorical_cols_:
            values = df[col].fillna("missing").astype(str)
            grouped = values.mask(values.isin(self.rare_values_[col]), "misc")
            grouped = grouped.where(grouped.isin(self.categories_[col]), "misc")
            for category in self.categories_[col]:
                features[f"{col}_{category}"] = (grouped == category).astype(int)

        features = features.drop(columns=[c for c in self.drop_feature_cols_ if c in features.columns], errors="ignore")
        features = features.reindex(columns=self.feature_cols_, fill_value=0).astype("float32")
        return pd.concat([out.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    def fit_transform(self, df, split_name):
        return self.fit(df).transform(df, split_name)

    def _feature_names(self):
        names = []
        names.extend(self.available_binary_cols_)
        names.extend(self.available_standard_cols_)
        names.extend(self.available_leave_numeric_cols_)
        for col in self.available_categorical_cols_:
            names.extend(f"{col}_{category}" for category in self.categories_[col])
        return [name for name in names if name not in set(self.drop_feature_cols_)]


def _duration_to_days(values, unit):
    values = pd.to_numeric(values, errors="coerce").astype(float)
    if unit == "days":
        return values
    if unit == "hours":
        return values / 24.0
    raise ValueError(f"Unsupported duration unit: {unit}")


def load_static_72h_base_table(config, logger):
    paths = config["paths"]
    columns = config["columns"]
    flat = pd.read_csv(paths["flat_features_path"])
    labels = pd.read_csv(paths["labels_path"])

    id_col = columns.get("id_col", ID_COL)
    event_col = columns["event_col"]
    duration_col = columns["duration_col"]
    duration_unit = columns.get("duration_unit", "days")

    label_cols = [id_col, event_col, duration_col]
    optional_cols = [col for col in columns.get("optional_label_cols", []) if col in labels.columns and col not in label_cols]
    df = flat.merge(labels[label_cols + optional_cols], on=id_col, how="inner")
    df = df.drop_duplicates(subset=[id_col]).copy()
    df = df.rename(columns={id_col: ID_COL, event_col: RAW_EVENT_COL})
    df[RAW_DURATION_DAYS_COL] = _duration_to_days(df[duration_col], duration_unit)
    if duration_col != RAW_DURATION_DAYS_COL:
        df = df.drop(columns=[duration_col])

    df[RAW_EVENT_COL] = pd.to_numeric(df[RAW_EVENT_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, RAW_EVENT_COL, RAW_DURATION_DAYS_COL])
    df[RAW_EVENT_COL] = df[RAW_EVENT_COL].astype(int)
    df = df[df[RAW_DURATION_DAYS_COL] > 0].copy()

    pred_hours = float(config["target"]["prediction_time_hours"])
    pred_days = pred_hours / 24.0
    horizon_days = float(config["target"]["max_horizon_days"])
    df = df[df[RAW_DURATION_DAYS_COL] > pred_days].copy()
    df[REL_DURATION_DAYS_COL] = df[RAW_DURATION_DAYS_COL] - pred_days
    df[DURATION_COL] = np.minimum(df[REL_DURATION_DAYS_COL], horizon_days).astype("float32")
    df[EVENT_COL] = ((df[RAW_EVENT_COL] == 1) & (df[REL_DURATION_DAYS_COL] <= horizon_days)).astype("int64")

    logger.info("static_72h base table: %d stays after Y_i > %.1fh filter", len(df), pred_hours)
    return df


def make_static_72h_split(df, config, logger):
    split_cfg = config["split"]
    seed = int(config.get("seed", 42))
    train_size = float(split_cfg.get("train_size", 0.6))
    val_size = float(split_cfg.get("val_size", 0.2))
    test_size = float(split_cfg.get("test_size", 0.2))
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must be 1.0")

    stratify = df[EVENT_COL] if split_cfg.get("stratify", True) else None
    train_df, rest_df = train_test_split(df, train_size=train_size, random_state=seed, stratify=stratify)
    rest_stratify = rest_df[EVENT_COL] if split_cfg.get("stratify", True) else None
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(rest_df, test_size=relative_test_size, random_state=seed, stratify=rest_stratify)
    logger.info("static_72h split sizes: train=%d validation=%d test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.copy(), val_df.copy(), test_df.copy()


def build_static_72h_preprocessor(config):
    prep_cfg = config["preprocessing"]
    return Static72hPreprocessor(
        standard_cols=prep_cfg.get("standard_cols", []),
        leave_numeric_cols=prep_cfg.get("leave_numeric_cols", []),
        categorical_cols=prep_cfg.get("categorical_cols", []),
        binary_maps=prep_cfg.get("binary_maps", {}),
        drop_feature_cols=prep_cfg.get("drop_feature_cols", []),
        rare_min_count=prep_cfg.get("rare_min_count", 1000),
        clip_min=prep_cfg.get("clip_min", -5.0),
        clip_max=prep_cfg.get("clip_max", 5.0),
    )


def feature_columns(df):
    return [col for col in df.columns if col not in TARGET_COLS]


def validate_static_72h_datasets(train_df, val_df, test_df, max_horizon_days=10.0):
    datasets = {"train": train_df, "validation": val_df, "test": test_df}
    ids = {name: set(df[ID_COL]) for name, df in datasets.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("patientunitstayid overlap detected between static_72h splits")
    columns = list(train_df.columns)
    for name, df in datasets.items():
        if list(df.columns) != columns:
            raise ValueError(f"{name} columns do not match train columns")
        if not (df[REL_DURATION_DAYS_COL] > 0).all():
            raise ValueError(f"{name} contains patients not observable after the landmark")
        if not (df[DURATION_COL] > 0).all():
            raise ValueError(f"{name} duration_eval_days must be positive")
        if not (df[DURATION_COL] <= float(max_horizon_days)).all():
            raise ValueError(f"{name} duration_eval_days exceeds horizon")
        if not set(df[EVENT_COL].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{name} event_eval must be binary")
        if df[feature_columns(df)].isna().any().any():
            raise ValueError(f"{name} has unexpected NaNs in feature columns")
        late_events = (df[RAW_EVENT_COL] == 1) & (df[REL_DURATION_DAYS_COL] > float(max_horizon_days))
        if not (df.loc[late_events, EVENT_COL] == 0).all():
            raise ValueError(f"{name} late post-horizon events must be censored at horizon")


def build_summary(train_df, val_df, test_df, preprocessor, config):
    pred_hours = float(config["target"]["prediction_time_hours"])
    summary = {
        "pipeline": config.get("experiment", {}).get("name", "static_72h_pycox"),
        "methodology": {
            "prediction_time_hours": pred_hours,
            "max_horizon_days": float(config["target"]["max_horizon_days"]),
            "time_unit": f"days since hour {int(pred_hours)}",
            "inclusion_rule": f"raw_duration_hours > {int(pred_hours)}",
        },
        "feature_columns": preprocessor.feature_cols_,
        "n_features": len(preprocessor.feature_cols_),
        "preprocessing": {
            "standard_cols": preprocessor.available_standard_cols_,
            "leave_numeric_cols": preprocessor.available_leave_numeric_cols_,
            "categorical_cols": preprocessor.available_categorical_cols_,
            "binary_cols": preprocessor.available_binary_cols_,
            "drop_feature_cols": preprocessor.drop_feature_cols_,
            "rare_min_count": preprocessor.rare_min_count,
            "fit_split": "train",
        },
        "splits": {},
    }
    for name, df in {"train": train_df, "validation": val_df, "test": test_df}.items():
        n = int(len(df))
        events = int(df[EVENT_COL].sum())
        summary["splits"][name] = {
            "n_patients": n,
            "events_within_10d_after_landmark": events,
            "administratively_censored_or_censored": int(n - events),
            "event_rate": float(events / n) if n else 0.0,
            "min_duration_eval_days": float(df[DURATION_COL].min()),
            "max_duration_eval_days": float(df[DURATION_COL].max()),
            "min_duration_rel_days": float(df[REL_DURATION_DAYS_COL].min()),
            "max_duration_rel_days": float(df[REL_DURATION_DAYS_COL].max()),
        }
    return summary


def build_static_72h_dataset(config, logger):
    df = load_static_72h_base_table(config, logger)
    train_raw, val_raw, test_raw = make_static_72h_split(df, config, logger)
    preprocessor = build_static_72h_preprocessor(config)
    train = preprocessor.fit_transform(train_raw, "train")
    val = preprocessor.transform(val_raw, "validation")
    test = preprocessor.transform(test_raw, "test")
    validate_static_72h_datasets(train, val, test, config["target"]["max_horizon_days"])

    paths = config["paths"]
    output_dir = Path(paths["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = config.get("output_file_suffix", "static_72h")
    train.to_parquet(output_dir / f"train_{suffix}.parquet", index=False)
    val.to_parquet(output_dir / f"val_{suffix}.parquet", index=False)
    test.to_parquet(output_dir / f"test_{suffix}.parquet", index=False)

    split_assignments = pd.concat(
        [
            train[TARGET_COLS],
            val[TARGET_COLS],
            test[TARGET_COLS],
        ],
        ignore_index=True,
    )
    split_assignments.to_parquet(output_dir / "split_assignments.parquet", index=False)

    preprocessor_path = Path(paths["preprocessor_path"])
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)

    summary = build_summary(train, val, test, preprocessor, config)
    summary_path = Path(paths["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("static_72h datasets saved in %s", output_dir)
    return train, val, test, summary
