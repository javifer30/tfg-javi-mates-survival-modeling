"""Build final landmark result audit tables and thesis figures.

This script only reads final outputs. It does not train models and does not
modify final metric artifacts.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = [24, 48, 72]
SEEDS = [42, 123, 2026]
HORIZONS = list(range(1, 11))

AUDIT_DIR = ROOT / "outputs" / "results_audit"
FIG_DIR = ROOT / "outputs" / "figures" / "landmark_final"
BITMAP_DIR = ROOT / "Imagenes" / "Bitmap"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    group: str
    final_dir: str
    tuning_dir: str | None
    predictive: bool = True
    has_patient_predictions: bool = False


MODELS = [
    ModelSpec("coxph", "CoxPH", "static", "static/final/coxph", "static/tuning/coxph"),
    ModelSpec(
        "random_survival_forest",
        "RSF",
        "static",
        "static/final/random_survival_forest",
        "static/tuning/random_survival_forest",
    ),
    ModelSpec("deepsurv", "DeepSurv", "static", "static/final/deepsurv", "static/tuning/deepsurv"),
    ModelSpec(
        "logistic_hazard",
        "LogisticHazard",
        "static_discrete",
        "static/final/logistic_hazard",
        "static/tuning/logistic_hazard",
    ),
    ModelSpec("pchazard", "PCHazard", "static_discrete", "static/final/pchazard", "static/tuning/pchazard"),
    ModelSpec(
        "deephit_single",
        "DeepHit",
        "static_discrete",
        "static/final/deephit_single",
        "static/tuning/deephit_single",
    ),
    ModelSpec(
        "dynamic_deephit_faithful",
        "Dynamic-DeepHit",
        "dynamic",
        "dynamic_deephit_faithful",
        "dynamic_deephit_faithful",
        has_patient_predictions=True,
    ),
    ModelSpec(
        "dysurv_faithful",
        "DySurv temporal",
        "dynamic",
        "dysurv_faithful",
        "dysurv_faithful",
        has_patient_predictions=True,
    ),
    ModelSpec(
        "dysurv_static_faithful",
        "DySurv static",
        "static_control",
        "dysurv_static_faithful",
        "dysurv_static_faithful",
        has_patient_predictions=True,
    ),
]

KM_SPEC = ModelSpec(
    "kaplan_meier",
    "Kaplan-Meier",
    "descriptive",
    "static/final/kaplan_meier",
    None,
    predictive=False,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt_mean_std(mean: float, std: float | None, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    if abs(std) < 5 * 10 ** (-(digits + 2)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def metric(summary: dict[str, Any], name: str, suffix: str) -> float:
    metrics = summary.get("metrics", {})
    return as_float(
        summary.get(
            f"{name}_{suffix}",
            metrics.get(f"test_{name}_{suffix}", metrics.get(f"{name}_{suffix}")),
        )
    )


def best_hp_path(landmark: int, spec: ModelSpec) -> Path:
    assert spec.tuning_dir is not None
    return ROOT / "outputs" / f"landmark_{landmark}h" / spec.tuning_dir / "best_hyperparameters.json"


def tuning_results_path(landmark: int, spec: ModelSpec) -> Path:
    assert spec.tuning_dir is not None
    return ROOT / "outputs" / f"landmark_{landmark}h" / spec.tuning_dir / "tuning_results.csv"


def final_dir(landmark: int, spec: ModelSpec) -> Path:
    return ROOT / "outputs" / f"landmark_{landmark}h" / spec.final_dir


def summary_path(landmark: int, spec: ModelSpec) -> Path:
    return final_dir(landmark, spec) / "final_seed_summary.json"


def seed_results_path(landmark: int, spec: ModelSpec) -> Path:
    return final_dir(landmark, spec) / "final_seed_results.csv"


def metrics_path(landmark: int, spec: ModelSpec, seed: int) -> Path:
    if spec.has_patient_predictions:
        return final_dir(landmark, spec) / "final" / f"seed_{seed}" / "metrics" / "metrics.json"
    return final_dir(landmark, spec) / f"seed_{seed}" / "metrics" / "metrics.json"


def prediction_path(landmark: int, spec: ModelSpec, seed: int) -> Path:
    if spec.has_patient_predictions:
        return final_dir(landmark, spec) / "final" / f"seed_{seed}" / "predictions" / "test_survival_predictions.parquet"
    return final_dir(landmark, spec) / f"seed_{seed}" / "predictions" / "test_survival_predictions.parquet"


def horizon_csv_path(landmark: int, spec: ModelSpec, seed: int) -> Path:
    if spec.has_patient_predictions:
        return final_dir(landmark, spec) / "final" / f"seed_{seed}" / "metrics" / "horizon_c_index.csv"
    return final_dir(landmark, spec) / f"seed_{seed}" / "metrics" / spec.key / "horizon_c_index.csv"


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    hp_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    per_seed_rows: list[pd.DataFrame] = []

    for landmark in LANDMARKS:
        for spec in MODELS:
            summary_file = summary_path(landmark, spec)
            seed_file = seed_results_path(landmark, spec)
            best_file = best_hp_path(landmark, spec)
            tuning_file = tuning_results_path(landmark, spec)
            summary = read_json(summary_file)
            best = read_json(best_file)
            selected_best = best.get("selected", best)
            seed_df = pd.read_csv(seed_file) if seed_file.exists() else pd.DataFrame()
            tuning_df = pd.read_csv(tuning_file) if tuning_file.exists() else pd.DataFrame()

            availability_rows.append(
                {
                    "landmark_hours": landmark,
                    "model": spec.label,
                    "summary_exists": summary_file.exists(),
                    "seed_results_exists": seed_file.exists(),
                    "best_hyperparameters_exists": best_file.exists(),
                    "tuning_results_exists": tuning_file.exists(),
                    "n_seed_rows": len(seed_df),
                    "n_tuning_rows": len(tuning_df),
                    "expected_seeds_present": sorted(seed_df["seed"].dropna().astype(int).tolist()) == SEEDS
                    if "seed" in seed_df.columns
                    else False,
                    "patient_prediction_parquets_exist": all(
                        prediction_path(landmark, spec, seed).exists() for seed in SEEDS
                    ),
                }
            )

            if not summary:
                continue

            hp = selected_best.get("hyperparameters") or best.get("hyperparameters") or summary.get("selected_hyperparameters", {})
            if isinstance(hp, str):
                try:
                    hp = json.loads(hp)
                except json.JSONDecodeError:
                    hp = {"raw": hp}
            selected_config_id = (
                summary.get("selected_config_id")
                or summary.get("config_id")
                or selected_best.get("config_id")
                or best.get("config_id")
            )

            rows.append(
                {
                    "landmark_hours": landmark,
                    "model_key": spec.key,
                    "model": spec.label,
                    "model_group": spec.group,
                    "selected_config_id": selected_config_id,
                    "n_runs": summary.get("n_runs", len(seed_df)),
                    "seeds": ",".join(str(x) for x in seed_df.get("seed", pd.Series(dtype=int)).tolist()),
                    "ctd_antolini_mean": metric(summary, "ctd_antolini", "mean"),
                    "ctd_antolini_std": metric(summary, "ctd_antolini", "std"),
                    "harrell_c_index_mean": metric(summary, "harrell_c_index", "mean"),
                    "harrell_c_index_std": metric(summary, "harrell_c_index", "std"),
                    "harrell_c_index_final_risk_mean": metric(summary, "harrell_c_index_final_risk", "mean"),
                    "harrell_c_index_final_risk_std": metric(summary, "harrell_c_index_final_risk", "std"),
                    "mean_horizon_c_index_mean": metric(summary, "mean_horizon_c_index", "mean"),
                    "mean_horizon_c_index_std": metric(summary, "mean_horizon_c_index", "std"),
                    "ibs_mean": metric(summary, "ibs", "mean"),
                    "ibs_std": metric(summary, "ibs", "std"),
                    "ibll_mean": metric(summary, "ibll", "mean"),
                    "ibll_std": metric(summary, "ibll", "std"),
                    "nbll_mean": metric(summary, "nbll", "mean"),
                    "nbll_std": metric(summary, "nbll", "std"),
                    "risk10_std_mean": metric(summary, "std_risk10", "mean"),
                    "risk10_range_mean": metric(summary, "range_risk10", "mean"),
                    "tail_probability_mean": metric(summary, "tail_probability_mean", "mean"),
                    "collapse_suspected_any": bool(summary.get("collapse_suspected_any", False)),
                    "source_summary": str(summary_file.relative_to(ROOT)),
                    "source_seed_results": str(seed_file.relative_to(ROOT)),
                }
            )

            hp_rows.append(
                {
                    "landmark_hours": landmark,
                    "model": spec.label,
                    "selected_config_id": selected_config_id,
                    "selection_metric": best.get("selection_metric")
                    or best.get("selection_rule")
                    or best.get("metric")
                    or "validation_ctd_antolini",
                    "validation_ctd_antolini": as_float(selected_best.get("validation_ctd_antolini")),
                    "validation_mean_horizon_c_index": as_float(selected_best.get("validation_mean_horizon_c_index")),
                    "validation_ibs": as_float(selected_best.get("validation_ibs")),
                    "validation_ibll": as_float(selected_best.get("validation_ibll")),
                    "validation_nbll": as_float(selected_best.get("validation_nbll")),
                    "hyperparameters": json.dumps(hp, ensure_ascii=False, sort_keys=True),
                    "best_hyperparameters_path": str(best_file.relative_to(ROOT)),
                    "tuning_results_path": str(tuning_file.relative_to(ROOT)),
                }
            )

            if not seed_df.empty:
                seed_df = seed_df.copy()
                if "landmark_hours" in seed_df.columns:
                    seed_df["landmark_hours"] = landmark
                else:
                    seed_df.insert(0, "landmark_hours", landmark)
                if "model" in seed_df.columns:
                    seed_df["model"] = spec.label
                else:
                    seed_df.insert(1, "model", spec.label)
                per_seed_rows.append(seed_df)

    results = pd.DataFrame(rows)
    hps = pd.DataFrame(hp_rows)
    availability = pd.DataFrame(availability_rows)
    per_seed = pd.concat(per_seed_rows, ignore_index=True) if per_seed_rows else pd.DataFrame()
    return results, hps, availability, per_seed


def collect_cohorts() -> pd.DataFrame:
    rows = []
    for landmark in LANDMARKS:
        km_summary = read_json(summary_path(landmark, KM_SPEC))
        dynamic_audit = read_json(ROOT / "outputs" / f"landmark_{landmark}h" / "dynamic" / "audit" / f"dynamic_{landmark}h_data_audit.json")
        splits = dynamic_audit.get("splits", {})
        train_npz = splits.get("train", {})
        val_npz = splits.get("validation", {})
        test_npz = splits.get("test", {})
        source_max = [
            s.get("max_used_offset_minutes")
            for s in dynamic_audit.get("temporal_source_stats", {}).get("sources", [])
            if s.get("max_used_offset_minutes") is not None
        ]

        def n_events(split: dict[str, Any]) -> float:
            n = as_float(split.get("n_patients"))
            rate = as_float(split.get("event_rate"))
            return round(n * rate) if not pd.isna(n) and not pd.isna(rate) else math.nan

        rows.append(
            {
                "landmark_hours": landmark,
                "time_origin": f"{landmark}h ICU landmark",
                "evaluation_unit": "days since landmark",
                "prediction_horizon_days": 10,
                "inclusion_rule": f"adult ICU stays alive/at risk beyond {landmark}h landmark",
                "train_n": train_npz.get("n_patients"),
                "train_events": n_events(train_npz),
                "train_event_rate": train_npz.get("event_rate"),
                "val_n": val_npz.get("n_patients"),
                "val_events": n_events(val_npz),
                "val_event_rate": val_npz.get("event_rate"),
                "test_n": test_npz.get("n_patients"),
                "test_events": n_events(test_npz),
                "test_event_rate": test_npz.get("event_rate"),
                "n_static_features": dynamic_audit.get("n_static_features"),
                "n_temporal_features": dynamic_audit.get("n_temporal_features"),
                "sequence_length_hours": train_npz.get("X_seq_shape", [None, None, None])[1],
                "max_used_offset_minutes": max(source_max) if source_max else None,
                "cohort_source": str(summary_path(landmark, KM_SPEC).relative_to(ROOT)),
                "dynamic_audit_source": str(
                    (ROOT / "outputs" / f"landmark_{landmark}h" / "dynamic" / "audit" / f"dynamic_{landmark}h_data_audit.json").relative_to(ROOT)
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_horizons() -> pd.DataFrame:
    rows = []
    for landmark in LANDMARKS:
        for spec in MODELS:
            for seed in SEEDS:
                hci: dict[int, float] = {}
                hcsv = horizon_csv_path(landmark, spec, seed)
                source = metrics_path(landmark, spec, seed)
                if hcsv.exists():
                    hdf = pd.read_csv(hcsv)
                    if "split" in hdf.columns:
                        hdf = hdf[hdf["split"] == "test"]
                    h_col = "horizon_day" if "horizon_day" in hdf.columns else "horizon_days"
                    for _, row in hdf.iterrows():
                        hci[int(float(row[h_col]))] = as_float(row["c_index"])
                    source = hcsv
                else:
                    metrics = read_json(metrics_path(landmark, spec, seed))
                    test = metrics.get("splits", {}).get("test", metrics.get("test", metrics))
                    raw_hci = test.get("horizon_c_index", {})
                    hci = {int(float(k)): as_float(v) for k, v in raw_hci.items()}

                for h in HORIZONS:
                    value = hci.get(h)
                    rows.append(
                        {
                            "landmark_hours": landmark,
                            "model": spec.label,
                            "model_key": spec.key,
                            "horizon_days": h,
                            "seed": seed,
                            "c_index": as_float(value),
                            "source": str(source.relative_to(ROOT)),
                        }
                    )
    long = pd.DataFrame(rows)
    return long


def rank_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    disc_cols = [
        "ctd_antolini_mean",
        "mean_horizon_c_index_mean",
        "harrell_c_index_mean",
        "harrell_c_index_final_risk_mean",
    ]
    prob_cols = ["ibs_mean", "ibll_mean", "nbll_mean"]
    disc = results.copy()
    prob = results.copy()
    for col in disc_cols:
        disc[f"{col}_rank"] = disc.groupby("landmark_hours")[col].rank(ascending=False, method="min")
    for col in prob_cols:
        prob[f"{col}_rank"] = prob.groupby("landmark_hours")[col].rank(ascending=True, method="min")
    return disc, prob


def collect_predictions(landmark: int, spec: ModelSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    dfs = []
    validation = {"landmark_hours": landmark, "model": spec.label, "available": True}
    id_cols = ["patient_id", "duration_eval_days", "event_eval"]
    curve_cols = [f"survival_day_{h}" for h in range(0, 11)]
    stored_curve_cols = [f"survival_day_{h}" for h in range(1, 11)]

    for seed in SEEDS:
        path = prediction_path(landmark, spec, seed)
        if not path.exists():
            validation["available"] = False
            validation["missing_path"] = str(path.relative_to(ROOT))
            return pd.DataFrame(), validation
        df = pd.read_parquet(path)
        required = id_cols + stored_curve_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            validation["available"] = False
            validation["missing_columns"] = ",".join(missing)
            return pd.DataFrame(), validation
        df = df[required].copy()
        df["survival_day_0"] = 1.0
        df = df[id_cols + curve_cols].copy()
        df["seed"] = seed
        df["risk10"] = 1.0 - df["survival_day_10"]
        validation[f"seed_{seed}_n"] = len(df)
        validation[f"seed_{seed}_risk10_delta_max"] = float(
            np.nanmax(np.abs(df["risk10"].to_numpy() - (1.0 - df["survival_day_10"].to_numpy())))
        )
        dfs.append(df)

    base = dfs[0][id_cols].reset_index(drop=True)
    for seed, df in zip(SEEDS[1:], dfs[1:]):
        same_ids = base.equals(df[id_cols].reset_index(drop=True))
        validation[f"same_ids_seed_{seed}"] = bool(same_ids)
        if not same_ids:
            validation["available"] = False
            validation["id_mismatch_seed"] = seed
            return pd.DataFrame(), validation

    merged = base.copy()
    for col in curve_cols:
        merged[col] = np.mean([df[col].to_numpy(dtype=float) for df in dfs], axis=0)
    merged["risk10"] = 1.0 - merged["survival_day_10"]
    validation["risk10_oriented"] = bool(merged["risk10"].corr(1.0 - merged["survival_day_10"]) > 0.999999)
    validation["n_test"] = len(merged)
    return merged, validation


def km_curve(durations: np.ndarray, events: np.ndarray, timeline: np.ndarray) -> np.ndarray:
    durations = durations.astype(float)
    events = events.astype(int)
    survival = 1.0
    out = []
    for t in timeline:
        event_times = np.unique(durations[(events == 1) & (durations <= t)])
        survival = 1.0
        for et in event_times:
            at_risk = np.sum(durations >= et)
            observed = np.sum((durations == et) & (events == 1))
            if at_risk > 0:
                survival *= 1.0 - observed / at_risk
        out.append(survival)
    return np.asarray(out)


def make_risk_group_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    validations = []
    for landmark in LANDMARKS:
        for spec in [m for m in MODELS if m.has_patient_predictions]:
            pred, validation = collect_predictions(landmark, spec)
            validations.append(validation)
            if pred.empty:
                continue
            pred = pred.copy()
            pred["risk_group"] = pd.qcut(pred["risk10"], q=3, labels=["Low risk", "Medium risk", "High risk"])
            timeline = np.arange(0, 11, dtype=float)
            observed_by_group = {}
            for group, gdf in pred.groupby("risk_group", observed=True):
                observed = km_curve(gdf["duration_eval_days"].to_numpy(), gdf["event_eval"].to_numpy(), timeline)
                observed_by_group[str(group)] = float(observed[-1])
                for h in range(0, 11):
                    rows.append(
                        {
                            "landmark_hours": landmark,
                            "model": spec.label,
                            "risk_group": str(group),
                            "horizon_days": h,
                            "n": len(gdf),
                            "events": int(gdf["event_eval"].sum()),
                            "risk10_mean": float(gdf["risk10"].mean()),
                            "observed_km": float(observed[h]),
                            "predicted_survival_mean": float(gdf[f"survival_day_{h}"].mean()),
                        }
                    )
            validation["km_order_low_ge_medium_ge_high_day10"] = (
                observed_by_group.get("Low risk", math.nan)
                >= observed_by_group.get("Medium risk", math.nan)
                >= observed_by_group.get("High risk", math.nan)
            )
    return pd.DataFrame(rows), pd.DataFrame(validations)


def plot_ctd(results: pd.DataFrame, models: list[str], path_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    x = np.arange(len(LANDMARKS))
    width = 0.75 / len(models)
    for i, model in enumerate(models):
        sub = results[results["model"] == model].set_index("landmark_hours").reindex(LANDMARKS)
        xpos = x - 0.375 + width / 2 + i * width
        ax.bar(
            xpos,
            sub["ctd_antolini_mean"],
            width,
            yerr=sub["ctd_antolini_std"].replace(0, np.nan),
            capsize=2,
            label=model,
        )
    ax.set_xticks(x, [f"{h}h" for h in LANDMARKS])
    ax.set_ylabel("Test Ctd Antolini")
    ax.set_title(title)
    ax.set_ylim(0.62, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    save_fig(fig, path_base)


def plot_horizons(horizon_summary: pd.DataFrame, selected: dict[int, list[str]], path_base: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, landmark in zip(axes, LANDMARKS):
        for model in selected[landmark]:
            sub = horizon_summary[
                (horizon_summary["landmark_hours"] == landmark) & (horizon_summary["model"] == model)
            ].sort_values("horizon_days")
            if sub.empty:
                continue
            ax.plot(sub["horizon_days"], sub["c_index_mean"], marker="o", linewidth=1.8, label=model)
        ax.set_title(f"{landmark}h")
        ax.set_xlabel("Days from landmark")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("C-index@h")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_fig(fig, path_base)


def plot_km_groups(group_data: pd.DataFrame, models: list[str], path_base: Path, title: str) -> None:
    fig, axes = plt.subplots(len(models), len(LANDMARKS), figsize=(4.2 * len(LANDMARKS), 3.2 * len(models)), sharex=True, sharey=True)
    if len(models) == 1:
        axes = np.asarray([axes])
    colors = {"Low risk": "#2F7D32", "Medium risk": "#D18B00", "High risk": "#B23A48"}
    for row_i, model in enumerate(models):
        for col_i, landmark in enumerate(LANDMARKS):
            ax = axes[row_i, col_i]
            sub = group_data[(group_data["model"] == model) & (group_data["landmark_hours"] == landmark)]
            for group in ["Low risk", "Medium risk", "High risk"]:
                g = sub[sub["risk_group"] == group].sort_values("horizon_days")
                if g.empty:
                    continue
                ax.step(g["horizon_days"], g["observed_km"], where="post", color=colors[group], label=group)
            ax.set_title(f"{model} - {landmark}h")
            ax.set_ylim(0.45, 1.02)
            ax.grid(alpha=0.25)
            if row_i == len(models) - 1:
                ax.set_xlabel("Days from landmark")
            if col_i == 0:
                ax.set_ylabel("Observed KM survival")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_fig(fig, path_base)


def plot_km_vs_predicted(group_data: pd.DataFrame, models: list[str], landmarks: list[int], path_base: Path, title: str) -> None:
    fig, axes = plt.subplots(len(models), len(landmarks), figsize=(4.3 * len(landmarks), 3.2 * len(models)), sharex=True, sharey=True)
    if len(models) == 1:
        axes = np.asarray([axes])
    if len(landmarks) == 1:
        axes = axes.reshape(len(models), 1)
    colors = {"Low risk": "#2F7D32", "Medium risk": "#D18B00", "High risk": "#B23A48"}
    for row_i, model in enumerate(models):
        for col_i, landmark in enumerate(landmarks):
            ax = axes[row_i, col_i]
            sub = group_data[(group_data["model"] == model) & (group_data["landmark_hours"] == landmark)]
            for group in ["Low risk", "Medium risk", "High risk"]:
                g = sub[sub["risk_group"] == group].sort_values("horizon_days")
                if g.empty:
                    continue
                ax.step(g["horizon_days"], g["observed_km"], where="post", color=colors[group], linewidth=1.8)
                ax.plot(g["horizon_days"], g["predicted_survival_mean"], color=colors[group], linestyle="--", linewidth=1.8)
            ax.set_title(f"{model} - {landmark}h")
            ax.set_ylim(0.45, 1.02)
            ax.grid(alpha=0.25)
            if row_i == len(models) - 1:
                ax.set_xlabel("Days from landmark")
            if col_i == 0:
                ax.set_ylabel("Survival")
    solid = plt.Line2D([0], [0], color="black", linewidth=1.8, label="Observed KM")
    dashed = plt.Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label="Mean predicted")
    group_handles = [
        plt.Line2D([0], [0], color=colors[group], linewidth=2.2, label=group)
        for group in ["Low risk", "Medium risk", "High risk"]
    ]
    fig.legend(handles=[solid, dashed] + group_handles, loc="lower center", ncol=5)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_fig(fig, path_base)


def save_fig(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def copy_bitmap(name: str) -> None:
    src = FIG_DIR / name
    if src.exists():
        BITMAP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, BITMAP_DIR / name)


def build_tables(results: pd.DataFrame, hps: pd.DataFrame, horizons: pd.DataFrame, group_data: pd.DataFrame) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    disc, prob = rank_tables(results)

    table_a_cols = [
        "landmark_hours",
        "model",
        "ctd_antolini_mean",
        "ctd_antolini_std",
        "ctd_antolini_mean_rank",
        "mean_horizon_c_index_mean",
        "mean_horizon_c_index_std",
        "mean_horizon_c_index_mean_rank",
        "harrell_c_index_mean",
        "harrell_c_index_final_risk_mean",
        "selected_config_id",
        "n_runs",
    ]
    table_b_cols = [
        "landmark_hours",
        "model",
        "ibs_mean",
        "ibs_std",
        "ibs_mean_rank",
        "ibll_mean",
        "ibll_std",
        "ibll_mean_rank",
        "nbll_mean",
        "nbll_std",
        "selected_config_id",
        "n_runs",
    ]

    results.to_csv(AUDIT_DIR / "landmark_final_results_long.csv", index=False)
    disc[table_a_cols].to_csv(AUDIT_DIR / "table_a_discrimination.csv", index=False)
    prob[table_b_cols].to_csv(AUDIT_DIR / "table_b_probabilistic.csv", index=False)
    hps.to_csv(AUDIT_DIR / "selected_hyperparameters.csv", index=False)
    horizons.to_csv(AUDIT_DIR / "horizon_cindex_long.csv", index=False)
    group_data.to_csv(AUDIT_DIR / "km_risk_group_predicted_mean_data_24_48_72.csv", index=False)

    horizon_summary = (
        horizons.groupby(["landmark_hours", "model", "model_key", "horizon_days"], as_index=False)["c_index"]
        .agg(c_index_mean="mean", c_index_std="std")
        .sort_values(["landmark_hours", "model", "horizon_days"])
    )
    horizon_summary.to_csv(AUDIT_DIR / "horizon_cindex_summary.csv", index=False)


def build_figure_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(FIG_DIR.glob("*")):
        if path.suffix.lower() in {".png", ".pdf"}:
            rows.append({"figure": path.name, "path": str(path.relative_to(ROOT)), "exists": path.exists()})
    return pd.DataFrame(rows)


def collect_dysurv_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for landmark in LANDMARKS:
        for spec in [m for m in MODELS if m.key in {"dysurv_faithful", "dysurv_static_faithful"}]:
            for seed in SEEDS:
                metrics = read_json(metrics_path(landmark, spec, seed))
                collapse = metrics.get("selected_collapse") or metrics.get("metric_best_collapse") or {}
                row = {
                    "landmark_hours": landmark,
                    "model": spec.label,
                    "seed": seed,
                    "source": str(metrics_path(landmark, spec, seed).relative_to(ROOT)),
                }
                for key in [
                    "collapse_suspected",
                    "std_risk10",
                    "min_risk10",
                    "max_risk10",
                    "range_risk10",
                    "std_mu",
                    "active_units",
                    "kl_loss",
                    "kl_per_dim",
                    "reconstruction_loss",
                    "survival_loss",
                ]:
                    row[key] = collapse.get(key)
                rows.append(row)
    diag = pd.DataFrame(rows)
    summary_cols = ["landmark_hours", "model"]
    if not diag.empty:
        numeric = diag.drop(columns=["source"], errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in {"landmark_hours", "seed"}]
        summary = diag.groupby(summary_cols, as_index=False)[numeric].agg(["mean", "std"])
        summary.columns = ["_".join([c for c in col if c]) for col in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(AUDIT_DIR / "dysurv_diagnostics_summary.csv", index=False)
    diag.to_csv(AUDIT_DIR / "dysurv_diagnostics_per_seed.csv", index=False)
    return diag


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results, hps, availability, per_seed = collect_results()
    cohorts = collect_cohorts()
    horizons = collect_horizons()
    group_data, group_validations = make_risk_group_data()

    availability.to_csv(AUDIT_DIR / "availability_summary.csv", index=False)
    per_seed.to_csv(AUDIT_DIR / "final_seed_results_all_models.csv", index=False)
    cohorts.to_csv(AUDIT_DIR / "cohort_summary.csv", index=False)
    group_validations.to_csv(AUDIT_DIR / "km_prediction_validation.csv", index=False)
    build_tables(results, hps, horizons, group_data)
    collect_dysurv_diagnostics(results)

    horizon_summary = pd.read_csv(AUDIT_DIR / "horizon_cindex_summary.csv")
    main_ctd_models = ["CoxPH", "RSF", "LogisticHazard", "DySurv static", "Dynamic-DeepHit", "DySurv temporal"]
    all_models = [m.label for m in MODELS]
    plot_ctd(results, main_ctd_models, FIG_DIR / "ctd_landmark_main", "Ctd Antolini across landmarks")
    plot_ctd(results, all_models, FIG_DIR / "ctd_landmark_appendix_all_models", "Ctd Antolini across landmarks - all models")

    best_discrete = {}
    for landmark in LANDMARKS:
        sub = results[(results["landmark_hours"] == landmark) & (results["model_group"] == "static_discrete")]
        best_model = sub.sort_values("mean_horizon_c_index_mean", ascending=False).iloc[0]["model"]
        best_discrete[landmark] = best_model
    horizon_main = {
        landmark: list(dict.fromkeys(["CoxPH", "RSF", best_discrete[landmark], "DySurv static", "Dynamic-DeepHit", "DySurv temporal"]))
        for landmark in LANDMARKS
    }
    horizon_all = {landmark: all_models for landmark in LANDMARKS}
    plot_horizons(
        horizon_summary,
        horizon_main,
        FIG_DIR / "cindex_horizon_main_24_48_72",
        "C-index by horizon - selected models",
    )
    plot_horizons(
        horizon_summary,
        horizon_all,
        FIG_DIR / "cindex_horizon_appendix_all_models_24_48_72",
        "C-index by horizon - all models",
    )

    plot_km_groups(
        group_data,
        ["Dynamic-DeepHit", "DySurv temporal"],
        FIG_DIR / "km_risk_groups_dynamic_models_24_48_72",
        "Observed Kaplan-Meier survival by predicted risk tercile",
    )
    plot_km_groups(
        group_data,
        ["Dynamic-DeepHit", "DySurv temporal", "DySurv static"],
        FIG_DIR / "km_risk_groups_faithful_models_24_48_72",
        "Observed Kaplan-Meier survival by predicted risk tercile - faithful models",
    )
    plot_km_groups(
        group_data,
        ["DySurv temporal", "DySurv static"],
        FIG_DIR / "km_risk_groups_dysurv_vs_static_24_48_72",
        "Observed Kaplan-Meier survival by predicted risk tercile - DySurv variants",
    )

    plot_km_vs_predicted(
        group_data,
        ["Dynamic-DeepHit", "DySurv temporal"],
        [24, 72],
        FIG_DIR / "km_vs_predicted_survival_dynamic_dysurv_24_72",
        "Observed KM and mean predicted survival by risk tercile",
    )
    plot_km_vs_predicted(
        group_data,
        ["Dynamic-DeepHit", "DySurv temporal"],
        LANDMARKS,
        FIG_DIR / "km_vs_predicted_survival_dynamic_dysurv_24_48_72",
        "Observed KM and mean predicted survival by risk tercile - dynamic models",
    )
    plot_km_vs_predicted(
        group_data,
        ["DySurv temporal", "DySurv static"],
        LANDMARKS,
        FIG_DIR / "km_vs_predicted_survival_dysurv_static_vs_temporal_24_48_72",
        "Observed KM and mean predicted survival by risk tercile - DySurv variants",
    )

    for name in [
        "ctd_landmark_main.png",
        "ctd_landmark_appendix_all_models.png",
        "cindex_horizon_main_24_48_72.png",
        "cindex_horizon_appendix_all_models_24_48_72.png",
        "km_risk_groups_dynamic_models_24_48_72.png",
        "km_risk_groups_faithful_models_24_48_72.png",
        "km_risk_groups_dysurv_vs_static_24_48_72.png",
        "km_vs_predicted_survival_dynamic_dysurv_24_72.png",
        "km_vs_predicted_survival_dynamic_dysurv_24_48_72.png",
        "km_vs_predicted_survival_dysurv_static_vs_temporal_24_48_72.png",
    ]:
        copy_bitmap(name)

    inventory = build_figure_inventory()
    inventory.to_csv(AUDIT_DIR / "figure_inventory.csv", index=False)

    print(f"Audit artifacts written to {AUDIT_DIR}")
    print(f"Figures written to {FIG_DIR}")
    print("Best static discrete by mean horizon C-index:")
    for landmark, model in best_discrete.items():
        print(f"  {landmark}h: {model}")


if __name__ == "__main__":
    main()
