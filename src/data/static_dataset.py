"""
Static dataset builder for the TFG survival experiments.

The preprocessing rules follow the MIMIC-IV flat preprocessing script in:
src/preprocessing paper/MIMIC_IV-preprocessing/flat_and_labels.py

TFG adaptation:
- split is created before preprocessing;
- all preprocessing parameters are fitted only on train;
- validation and test receive transform only.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COL = "patientunitstayid"
TIME_COL = "time_to_event"
EVENT_COL = "observed_event"
SPLIT_COL = "split"
TARGET_COLS = [ID_COL, TIME_COL, EVENT_COL, SPLIT_COL]


@dataclass
class StaticPreprocessor:
    categorical_cols: list
    standard_cols: list
    minmax_cols: list
    binary_maps: dict
    rare_min_count: int = 1000
    clip_min: float = -4.0
    clip_max: float = 4.0

    def fit(self, df):
        self.feature_cols_ = []
        self.available_categorical_cols_ = [c for c in self.categorical_cols if c in df.columns]
        self.available_standard_cols_ = [c for c in self.standard_cols if c in df.columns]
        self.available_minmax_cols_ = [c for c in self.minmax_cols if c in df.columns]
        self.available_binary_cols_ = [c for c in self.binary_maps if c in df.columns]

        self.rare_values_ = {}
        self.categories_ = {}
        for col in self.available_categorical_cols_:
            values = df[col].fillna("missing").astype(str)
            counts = values.value_counts(dropna=False)
            self.rare_values_[col] = set(counts[counts < self.rare_min_count].index)
            grouped = values.mask(values.isin(self.rare_values_[col]), "misc")
            self.categories_[col] = sorted(grouped.unique().tolist())

        self.standard_params_ = {}
        for col in self.available_standard_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            std = values.std()
            self.standard_params_[col] = {
                "mean": float(values.mean()) if not np.isnan(values.mean()) else 0.0,
                "std": float(std) if pd.notna(std) and std != 0 else 1.0,
            }

        self.minmax_params_ = {}
        for col in self.available_minmax_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            q05 = values.quantile(0.05)
            q95 = values.quantile(0.95)
            if pd.isna(q05):
                q05 = 0.0
            if pd.isna(q95) or q95 == q05:
                q95 = float(q05) + 1.0
            self.minmax_params_[col] = {"q05": float(q05), "q95": float(q95)}

        self.feature_cols_ = self._build_feature_names()
        return self

    def transform(self, df, split_name):
        out = pd.DataFrame(index=df.index)
        out[ID_COL] = df[ID_COL].values
        out[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").values
        out[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").astype(int).values
        out[SPLIT_COL] = split_name

        features = pd.DataFrame(index=df.index)

        for col in self.available_binary_cols_:
            mapping = self.binary_maps[col]
            features[col] = df[col].map(mapping).fillna(0.5).astype(float)

        for col in self.available_standard_cols_:
            features[f"null{col}"] = df[col].isna().astype(int)
            values = pd.to_numeric(df[col], errors="coerce")
            params = self.standard_params_[col]
            scaled = (values - params["mean"]) / params["std"]
            features[col] = scaled.clip(self.clip_min, self.clip_max).fillna(0.0)

        for col in self.available_minmax_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            params = self.minmax_params_[col]
            scaled = 2.0 * (values - params["q05"]) / (params["q95"] - params["q05"]) - 1.0
            features[col] = scaled.clip(self.clip_min, self.clip_max).fillna(0.0)

        for col in self.available_categorical_cols_:
            values = df[col].fillna("missing").astype(str)
            grouped = values.mask(values.isin(self.rare_values_[col]), "misc")
            grouped = grouped.where(grouped.isin(self.categories_[col]), "misc")
            for category in self.categories_[col]:
                features[f"{col}_{category}"] = (grouped == category).astype(int)

        features = features.reindex(columns=self.feature_cols_, fill_value=0)
        features = features.astype(float)
        return pd.concat([out, features], axis=1)

    def fit_transform(self, df, split_name):
        return self.fit(df).transform(df, split_name)

    def _build_feature_names(self):
        names = []
        names.extend(self.available_binary_cols_)
        for col in self.available_standard_cols_:
            names.append(f"null{col}")
            names.append(col)
        names.extend(self.available_minmax_cols_)
        for col in self.available_categorical_cols_:
            names.extend(f"{col}_{category}" for category in self.categories_[col])
        return names


def load_base_static_table(config, logger):
    paths = config["paths"]
    flat = pd.read_csv(paths["flat_features_path"])
    labels = pd.read_csv(paths["labels_path"])

    id_col = config["columns"].get("id_col", ID_COL)
    event_source = config["columns"]["event_col"]
    time_source = config["columns"]["duration_col"]

    df = flat.merge(labels[[id_col, event_source, time_source]], on=id_col, how="inner")
    df = df.drop_duplicates(subset=[id_col]).copy()
    df = df.rename(columns={event_source: EVENT_COL, time_source: TIME_COL})

    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, EVENT_COL, TIME_COL])
    df[EVENT_COL] = df[EVENT_COL].astype(int)
    df = df[df[TIME_COL] > 0].copy()
    logger.info("Base static table: %d stays after merge and target cleaning", len(df))
    return df


def make_static_split(df, config, logger):
    split_cfg = config["split"]
    seed = config.get("seed", 42)
    train_size = split_cfg.get("train_size", 0.6)
    val_size = split_cfg.get("val_size", 0.2)
    test_size = split_cfg.get("test_size", 0.2)
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must be 1.0")

    stratify = df[EVENT_COL] if split_cfg.get("stratify", True) else None
    train_df, rest_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    rest_stratify = rest_df[EVENT_COL] if split_cfg.get("stratify", True) else None
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=rest_stratify,
    )
    logger.info("Split sizes: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.copy(), val_df.copy(), test_df.copy()


def build_static_preprocessor(config):
    prep_cfg = config["preprocessing"]
    return StaticPreprocessor(
        categorical_cols=prep_cfg["categorical_cols"],
        standard_cols=prep_cfg["standard_cols"],
        minmax_cols=prep_cfg["minmax_cols"],
        binary_maps=prep_cfg.get("binary_maps", {}),
        rare_min_count=prep_cfg.get("rare_min_count", 1000),
        clip_min=prep_cfg.get("clip_min", -4.0),
        clip_max=prep_cfg.get("clip_max", 4.0),
    )


def validate_static_datasets(train_df, val_df, test_df, allow_null_cols=None):
    allow_null_cols = set(allow_null_cols or [])
    datasets = {"train": train_df, "validation": val_df, "test": test_df}
    ids = {name: set(df[ID_COL]) for name, df in datasets.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("patientunitstayid overlap detected between splits")

    columns = list(train_df.columns)
    for name, df in datasets.items():
        if list(df.columns) != columns:
            raise ValueError(f"{name} columns do not match train columns")
        if not set(df[EVENT_COL].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{name} observed_event must be binary")
        if not (df[TIME_COL] > 0).all():
            raise ValueError(f"{name} time_to_event must be positive")
        null_cols = set(df.columns[df.isna().any()]) - allow_null_cols
        if null_cols:
            raise ValueError(f"{name} has unexpected null columns: {sorted(null_cols)}")


def build_summary(train_df, val_df, test_df, preprocessor):
    summary = {
        "feature_columns": preprocessor.feature_cols_,
        "n_features": len(preprocessor.feature_cols_),
        "splits": {},
        "preprocessing": {
            "source": "src/preprocessing paper/MIMIC_IV-preprocessing/flat_and_labels.py",
            "rare_min_count": preprocessor.rare_min_count,
            "standard_cols": preprocessor.available_standard_cols_,
            "minmax_cols": preprocessor.available_minmax_cols_,
            "categorical_cols": preprocessor.available_categorical_cols_,
            "binary_cols": preprocessor.available_binary_cols_,
        },
    }
    for name, df in {"train": train_df, "validation": val_df, "test": test_df}.items():
        events = int(df[EVENT_COL].sum())
        n = int(len(df))
        summary["splits"][name] = {
            "n_patients": n,
            "events": events,
            "censored": int(n - events),
            "event_rate": float(events / n) if n else 0.0,
            "min_time_to_event": float(df[TIME_COL].min()),
            "max_time_to_event": float(df[TIME_COL].max()),
        }
    return summary


def build_static_dataset(config, logger):
    df = load_base_static_table(config, logger)
    train_raw, val_raw, test_raw = make_static_split(df, config, logger)

    preprocessor = build_static_preprocessor(config)
    train = preprocessor.fit_transform(train_raw, "train")
    val = preprocessor.transform(val_raw, "validation")
    test = preprocessor.transform(test_raw, "test")

    validate_static_datasets(train, val, test, config["validation"].get("allow_null_cols", []))

    paths = config["paths"]
    output_dir = Path(paths["processed_static_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train_static.parquet", index=False)
    val.to_parquet(output_dir / "val_static.parquet", index=False)
    test.to_parquet(output_dir / "test_static.parquet", index=False)

    split_assignments = pd.concat(
        [train[[ID_COL, EVENT_COL, TIME_COL, SPLIT_COL]], val[[ID_COL, EVENT_COL, TIME_COL, SPLIT_COL]], test[[ID_COL, EVENT_COL, TIME_COL, SPLIT_COL]]],
        ignore_index=True,
    )
    split_assignments.to_parquet(output_dir / "split_assignments.parquet", index=False)

    preprocessor_path = Path(paths["preprocessor_path"])
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)

    summary = build_summary(train, val, test, preprocessor)
    summary_path = Path(paths["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Static datasets saved in %s", output_dir)
    logger.info("Static preprocessor saved in %s", preprocessor_path)
    return train, val, test, summary


def feature_columns(df):
    return [col for col in df.columns if col not in TARGET_COLS]
