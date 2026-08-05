import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HORIZON_DAYS = 10.0
SEEDS = [42, 123, 2026]
RISK_GROUPS = ["Low risk", "Medium risk", "High risk"]
RISK_COLORS = {
    "Low risk": "#4C78A8",
    "Medium risk": "#F58518",
    "High risk": "#E45756",
}
MODEL_SPECS = {
    "dynamic_deephit": {
        "label": "Dynamic-DeepHit",
        "root": "dynamic_deephit_faithful",
        "kind": "dynamic",
    },
    "dysurv": {
        "label": "DySurv",
        "root": "dysurv_faithful",
        "kind": "dynamic",
    },
    "dysurv_static": {
        "label": "DySurv static",
        "root": "dysurv_static_faithful",
        "kind": "dynamic",
    },
}
STATIC_MODEL_KEYS = {
    "Random Survival Forest": "random_survival_forest",
    "DeepSurv": "deepsurv",
    "LogisticHazard": "logistic_hazard",
    "PCHazard": "pchazard",
    "DeepHit": "deephit_single",
}


def _prediction_path(landmark_hours, model_key, seed):
    root = MODEL_SPECS[model_key]["root"]
    return (
        PROJECT_ROOT
        / f"outputs/landmark_{landmark_hours}h/{root}/final/seed_{seed}/predictions/test_survival_predictions.parquet"
    )


def load_seed_averaged_predictions(landmark_hours, model_key):
    frames = []
    source_paths = []
    for seed in SEEDS:
        path = _prediction_path(landmark_hours, model_key, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing final test predictions: {path}")
        df = pd.read_parquet(path)
        required = {"patient_id", "duration_eval_days", "event_eval", "risk10"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if df["patient_id"].duplicated().any():
            raise ValueError(f"Duplicated patient_id values in {path}")
        df = df.copy()
        df["seed"] = seed
        frames.append(df)
        source_paths.append(str(path.relative_to(PROJECT_ROOT)))

    base = frames[0][["patient_id", "duration_eval_days", "event_eval"]].sort_values("patient_id").reset_index(drop=True)
    for df in frames[1:]:
        check = df[["patient_id", "duration_eval_days", "event_eval"]].sort_values("patient_id").reset_index(drop=True)
        if not base["patient_id"].equals(check["patient_id"]):
            raise ValueError(f"Patient order/set mismatch across seeds for {landmark_hours}h {model_key}")
        if not np.allclose(base["duration_eval_days"], check["duration_eval_days"]):
            raise ValueError(f"Observed duration mismatch across seeds for {landmark_hours}h {model_key}")
        if not base["event_eval"].equals(check["event_eval"]):
            raise ValueError(f"Event indicator mismatch across seeds for {landmark_hours}h {model_key}")

    merged = base.copy()
    risk_by_seed = []
    for df in frames:
        risk_by_seed.append(df.set_index("patient_id")["risk10"].rename(f"risk10_seed_{int(df['seed'].iloc[0])}"))
    risk_df = pd.concat(risk_by_seed, axis=1).reset_index()
    merged = merged.merge(risk_df, on="patient_id", how="inner")
    merged["risk10"] = merged[[f"risk10_seed_{seed}" for seed in SEEDS]].mean(axis=1)

    survival_cols = [f"survival_day_{day}" for day in range(1, 11) if f"survival_day_{day}" in frames[0].columns]
    for col in survival_cols:
        surv_by_seed = [df.set_index("patient_id")[col].rename(f"{col}_seed_{int(df['seed'].iloc[0])}") for df in frames]
        surv_df = pd.concat(surv_by_seed, axis=1).reset_index()
        merged = merged.merge(surv_df, on="patient_id", how="inner")
        merged[col] = merged[[f"{col}_seed_{seed}" for seed in SEEDS]].mean(axis=1)

    merged["duration_eval_days"] = merged["duration_eval_days"].clip(lower=0.0, upper=HORIZON_DAYS)
    merged["event_eval"] = merged["event_eval"].astype(int)
    merged["model"] = MODEL_SPECS[model_key]["label"]
    merged["model_key"] = model_key
    merged["landmark_hours"] = int(landmark_hours)
    merged.attrs["source_paths"] = source_paths
    return merged


def assign_risk_groups(df):
    out = df.copy()
    # Rank before qcut to keep terciles stable even when risks have ties.
    out["_risk_rank"] = out["risk10"].rank(method="first")
    out["risk_group"] = pd.qcut(out["_risk_rank"], q=3, labels=RISK_GROUPS)
    return out.drop(columns=["_risk_rank"])


def km_curve(group_df):
    durations = group_df["duration_eval_days"].to_numpy(dtype=float)
    events = group_df["event_eval"].to_numpy(dtype=int)
    event_times = np.sort(np.unique(durations[(events == 1) & (durations <= HORIZON_DAYS)]))
    survival = 1.0
    rows = [
        {
            "time_day": 0.0,
            "km_survival": 1.0,
            "n_at_risk": int(len(group_df)),
            "n_events": 0,
        }
    ]
    for time in event_times:
        n_at_risk = int(np.sum(durations >= time))
        n_events = int(np.sum((durations == time) & (events == 1)))
        if n_at_risk <= 0:
            continue
        survival *= 1.0 - (n_events / n_at_risk)
        rows.append(
            {
                "time_day": float(time),
                "km_survival": float(survival),
                "n_at_risk": n_at_risk,
                "n_events": n_events,
            }
        )
    if rows[-1]["time_day"] < HORIZON_DAYS:
        rows.append(
            {
                "time_day": HORIZON_DAYS,
                "km_survival": float(survival),
                "n_at_risk": int(np.sum(durations >= HORIZON_DAYS)),
                "n_events": 0,
            }
        )
    return pd.DataFrame(rows)


def build_km_plot_data(predictions):
    rows = []
    validation_rows = []
    for (landmark_hours, model), model_df in predictions.groupby(["landmark_hours", "model"], sort=False):
        final_survival = {}
        for risk_group in RISK_GROUPS:
            group_df = model_df[model_df["risk_group"].eq(risk_group)]
            curve = km_curve(group_df)
            final_survival[risk_group] = float(curve[curve["time_day"].le(HORIZON_DAYS)]["km_survival"].iloc[-1])
            for row in curve.itertuples(index=False):
                rows.append(
                    {
                        "landmark_hours": int(landmark_hours),
                        "model": model,
                        "risk_group": risk_group,
                        "time_day": row.time_day,
                        "km_survival": row.km_survival,
                        "n_at_risk": row.n_at_risk,
                        "n_events": row.n_events,
                    }
                )
        validation_rows.append(
            {
                "landmark_hours": int(landmark_hours),
                "model": model,
                "low_survival_10d": final_survival["Low risk"],
                "medium_survival_10d": final_survival["Medium risk"],
                "high_survival_10d": final_survival["High risk"],
                "risk_order_valid": final_survival["Low risk"] >= final_survival["Medium risk"] >= final_survival["High risk"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def build_predicted_group_data(predictions):
    survival_cols = [f"survival_day_{day}" for day in range(1, 11)]
    rows = []
    for (landmark_hours, model, risk_group), group_df in predictions.groupby(
        ["landmark_hours", "model", "risk_group"], sort=False, observed=False
    ):
        rows.append(
            {
                "landmark_hours": int(landmark_hours),
                "model": model,
                "risk_group": risk_group,
                "time_day": 0.0,
                "mean_predicted_survival": 1.0,
            }
        )
        for day in range(1, 11):
            col = f"survival_day_{day}"
            if col not in group_df.columns:
                continue
            rows.append(
                {
                    "landmark_hours": int(landmark_hours),
                    "model": model,
                    "risk_group": risk_group,
                    "time_day": float(day),
                    "mean_predicted_survival": float(group_df[col].mean()),
                }
            )
    return pd.DataFrame(rows)


def select_best_static_by_landmark(landmarks):
    selections = {}
    for landmark_hours in landmarks:
        scores = []
        for label, key in STATIC_MODEL_KEYS.items():
            path = PROJECT_ROOT / f"outputs/landmark_{landmark_hours}h/static/final/{key}/final_seed_results.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "test_ctd_antolini" in df.columns:
                scores.append((label, key, float(df["test_ctd_antolini"].mean()), "test_ctd_antolini"))
            elif "test_mean_horizon_c_index" in df.columns:
                scores.append((label, key, float(df["test_mean_horizon_c_index"].mean()), "test_mean_horizon_c_index"))
        if scores:
            selections[int(landmark_hours)] = max(scores, key=lambda item: item[2])
    return selections


def static_prediction_availability(best_static):
    rows = []
    for landmark_hours, (label, key, score, criterion) in best_static.items():
        final_dir = PROJECT_ROOT / f"outputs/landmark_{landmark_hours}h/static/final/{key}"
        full_predictions = sorted(final_dir.glob("seed_*/predictions/**/test_survival_predictions.parquet"))
        example_curves = sorted(final_dir.glob("seed_*/predictions/**/*.csv"))
        rows.append(
            {
                "landmark_hours": landmark_hours,
                "best_static_model": label,
                "selected_by": criterion,
                "selection_score": score,
                "full_test_prediction_files": len(full_predictions),
                "example_prediction_files": len(example_curves),
                "can_plot_risk_terciles": bool(full_predictions),
            }
        )
    return pd.DataFrame(rows)


def _plot_panel(ax, km_df, model, landmark_hours, predicted_df=None):
    panel = km_df[(km_df["model"].eq(model)) & (km_df["landmark_hours"].eq(landmark_hours))]
    if panel.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    for group in RISK_GROUPS:
        curve = panel[panel["risk_group"].eq(group)].sort_values("time_day")
        ax.step(
            curve["time_day"],
            curve["km_survival"],
            where="post",
            label=group,
            color=RISK_COLORS[group],
            linewidth=2.0,
        )
        if predicted_df is not None:
            pred = predicted_df[
                (predicted_df["model"].eq(model))
                & (predicted_df["landmark_hours"].eq(landmark_hours))
                & (predicted_df["risk_group"].eq(group))
            ].sort_values("time_day")
            if not pred.empty:
                ax.plot(
                    pred["time_day"],
                    pred["mean_predicted_survival"],
                    color=RISK_COLORS[group],
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.8,
                )
    ax.set_title(f"{model}\nLandmark {landmark_hours}h")
    ax.set_xlim(0, HORIZON_DAYS)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days from landmark")
    ax.grid(True, alpha=0.25, linewidth=0.7)


def _save_figure(fig, output_base):
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def plot_grid(km_df, models, landmarks, output_base, predicted_df=None, title=None):
    fig, axes = plt.subplots(len(models), len(landmarks), figsize=(5.7 * len(landmarks), 3.6 * len(models)), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(models), len(landmarks))
    for i, model in enumerate(models):
        for j, landmark_hours in enumerate(landmarks):
            _plot_panel(axes[i, j], km_df, model, landmark_hours, predicted_df=predicted_df)
            if j == 0:
                axes[i, j].set_ylabel("Survival" if predicted_df is not None else "Observed survival")
    bottom = 0.06
    if predicted_df is not None:
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
        fig.text(
            0.5,
            0.115,
            "Solid: observed Kaplan-Meier; dashed: mean predicted survival",
            ha="center",
            va="center",
            fontsize=10,
            color="#444444",
        )
        bottom = 0.20
    else:
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return _save_figure(fig, output_base)


def copy_for_latex(paths, latex_dir):
    latex_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        if path.suffix.lower() == ".png":
            target = latex_dir / path.name
            shutil.copy2(path, target)
            copied.append(target)
    return copied


def parse_args():
    parser = argparse.ArgumentParser(description="Plot observed KM curves by predicted risk terciles.")
    parser.add_argument("--landmarks", nargs="+", type=int, default=[48, 72])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dynamic_deephit", "dysurv", "dysurv_static"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument("--include-predicted", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    landmarks = list(dict.fromkeys(args.landmarks))
    frames = []
    source_paths = []
    for landmark_hours in landmarks:
        for model_key in args.models:
            df = assign_risk_groups(load_seed_averaged_predictions(landmark_hours, model_key))
            source_paths.extend(df.attrs.get("source_paths", []))
            frames.append(df)
    if not frames:
        raise ValueError("No prediction files loaded.")
    predictions = pd.concat(frames, ignore_index=True)
    km_df, validation_df = build_km_plot_data(predictions)
    predicted_df = build_predicted_group_data(predictions)
    best_static = select_best_static_by_landmark(landmarks)
    static_availability_df = static_prediction_availability(best_static)

    output_dir = PROJECT_ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    km_path = output_dir / "km_risk_group_plot_data.csv"
    km_df.to_csv(km_path, index=False)
    predicted_path = output_dir / "km_risk_group_predicted_mean_data.csv"
    predicted_df.to_csv(predicted_path, index=False)
    validation_path = output_dir / "km_risk_group_validation.csv"
    validation_df.to_csv(validation_path, index=False)
    static_availability_path = output_dir / "km_risk_group_static_availability.csv"
    static_availability_df.to_csv(static_availability_path, index=False)

    outputs = []
    outputs.extend(
        plot_grid(
            km_df,
            ["Dynamic-DeepHit"],
            landmarks,
            output_dir / "km_risk_groups_dynamic_deephit_48_72",
            title="Observed survival by Dynamic-DeepHit risk tercile",
        )
    )
    outputs.extend(
        plot_grid(
            km_df,
            ["Dynamic-DeepHit", "DySurv"],
            landmarks,
            output_dir / "km_risk_groups_dynamic_models_48_72",
            title="Observed survival by predicted risk tercile: dynamic models",
        )
    )
    outputs.extend(
        plot_grid(
            km_df,
            ["DySurv", "DySurv static"],
            landmarks,
            output_dir / "km_risk_groups_dysurv_vs_static_48_72",
            title="Observed survival by predicted risk tercile: DySurv temporal vs static",
        )
    )
    if args.include_predicted:
        outputs.extend(
            plot_grid(
                km_df,
                ["Dynamic-DeepHit"],
                landmarks,
                output_dir / "km_vs_predicted_survival_dynamic_deephit_48_72",
                predicted_df=predicted_df,
                title="Observed KM and mean predicted survival by risk tercile",
            )
        )
        outputs.extend(
            plot_grid(
                km_df,
                ["DySurv"],
                landmarks,
                output_dir / "km_vs_predicted_survival_dysurv_48_72",
                predicted_df=predicted_df,
                title="Observed KM and mean predicted survival by DySurv risk tercile",
            )
        )
        outputs.extend(
            plot_grid(
                km_df,
                ["DySurv static"],
                landmarks,
                output_dir / "km_vs_predicted_survival_dysurv_static_48_72",
                predicted_df=predicted_df,
                title="Observed KM and mean predicted survival by DySurv static risk tercile",
            )
        )

    copied = copy_for_latex([path for path in outputs if path.suffix.lower() == ".png"], PROJECT_ROOT / "Imagenes/Bitmap")

    print("Loaded final test prediction sources:")
    for path in sorted(set(source_paths)):
        print("  ", path)
    print("Wrote:", km_path)
    print("Wrote:", predicted_path)
    print("Wrote:", validation_path)
    print("Wrote:", static_availability_path)
    for path in outputs + copied:
        print("Wrote:", path)
    print("Best static selections:")
    for landmark_hours, (label, _key, score, criterion) in best_static.items():
        print(f"  {landmark_hours}h: {label} by {criterion}={score:.6f}")
    invalid = validation_df[~validation_df["risk_order_valid"]]
    if invalid.empty:
        print("Risk-group ordering check passed for all plotted model/landmark pairs.")
    else:
        print("WARNING: risk-group ordering check failed for:")
        print(invalid.to_string(index=False))
    unavailable = static_availability_df[~static_availability_df["can_plot_risk_terciles"]]
    if not unavailable.empty:
        print("WARNING: best static conventional models were not plotted because full test prediction files were unavailable:")
        print(unavailable.to_string(index=False))


if __name__ == "__main__":
    main()
