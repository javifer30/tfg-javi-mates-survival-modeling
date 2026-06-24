import argparse
import json
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


HORIZONS = list(range(1, 11))

STATIC_MODELS = {
    "coxph": "CoxPH",
    "deepsurv": "DeepSurv",
    "logistic_hazard": "LogisticHazard",
    "pchazard": "PCHazard",
    "deephit_single": "DeepHit",
}
DYNAMIC_MODELS = {
    "dysurv_static_faithful": "DySurv static",
    "dynamic_deephit_faithful": "Dynamic-DeepHit",
    "dysurv_faithful": "DySurv",
}
CONVENTIONAL_STATIC = ["DeepSurv", "LogisticHazard", "PCHazard", "DeepHit"]
MAIN_MODEL_ORDER = ["CoxPH", "Best static", "DySurv static", "Dynamic-DeepHit", "DySurv"]
APPENDIX_MODEL_ORDER = [
    "CoxPH",
    "DeepSurv",
    "LogisticHazard",
    "PCHazard",
    "DeepHit",
    "DySurv static",
    "Dynamic-DeepHit",
    "DySurv",
]
STYLE = {
    "CoxPH": {"color": "#222222", "marker": "o", "linestyle": "-"},
    "Best static": {"color": "#4C78A8", "marker": "s", "linestyle": "-"},
    "DeepSurv": {"color": "#4C78A8", "marker": "s", "linestyle": "-"},
    "LogisticHazard": {"color": "#72B7B2", "marker": "^", "linestyle": "-."},
    "PCHazard": {"color": "#9D755D", "marker": "D", "linestyle": ":"},
    "DeepHit": {"color": "#59A14F", "marker": "v", "linestyle": "--"},
    "DySurv static": {"color": "#F58518", "marker": "P", "linestyle": "--"},
    "Dynamic-DeepHit": {"color": "#54A24B", "marker": "X", "linestyle": "-"},
    "DySurv": {"color": "#E45756", "marker": "*", "linestyle": "-"},
}


def _read_static_seed_rows(landmark_hours, model_key, model_label):
    model_dir = PROJECT_ROOT / f"outputs/landmark_{landmark_hours}h/static/final/{model_key}"
    rows = []
    for horizon_path in sorted(model_dir.glob(f"seed_*/metrics/{model_key}/horizon_c_index.csv")):
        seed = int(horizon_path.parts[-4].replace("seed_", ""))
        df = pd.read_csv(horizon_path)
        test = df[df["split"].eq("test")].copy()
        for row in test.itertuples(index=False):
            rows.append(
                {
                    "landmark_hours": landmark_hours,
                    "model": model_label,
                    "model_key": model_key,
                    "seed": seed,
                    "horizon_day": int(row.horizon_day),
                    "c_index": float(row.c_index),
                    "source_path": str(horizon_path.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def _read_dynamic_seed_rows(landmark_hours, model_key, model_label):
    final_dir = PROJECT_ROOT / f"outputs/landmark_{landmark_hours}h/{model_key}/final"
    rows = []
    for metrics_path in sorted(final_dir.glob("seed_*/metrics/metrics.json")):
        seed = int(metrics_path.parts[-3].replace("seed_", ""))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        horizon_c_index = metrics.get("splits", {}).get("test", {}).get("horizon_c_index", {})
        for horizon, value in horizon_c_index.items():
            rows.append(
                {
                    "landmark_hours": landmark_hours,
                    "model": model_label,
                    "model_key": model_key,
                    "seed": seed,
                    "horizon_day": int(float(horizon)),
                    "c_index": float(value),
                    "source_path": str(metrics_path.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def collect_seed_rows(landmarks):
    rows = []
    for landmark_hours in landmarks:
        for model_key, model_label in STATIC_MODELS.items():
            if model_key == "kaplan_meier":
                continue
            rows.extend(_read_static_seed_rows(landmark_hours, model_key, model_label))
        for model_key, model_label in DYNAMIC_MODELS.items():
            rows.extend(_read_dynamic_seed_rows(landmark_hours, model_key, model_label))
    return pd.DataFrame(rows)


def validate_seed_rows(seed_df):
    if seed_df.empty:
        raise ValueError("No final horizon C-index rows were found.")
    problems = []
    for (landmark_hours, model), group in seed_df.groupby(["landmark_hours", "model"]):
        horizons = sorted(group["horizon_day"].unique().tolist())
        if horizons != HORIZONS:
            problems.append(f"{landmark_hours}h {model}: horizons={horizons}")
    if problems:
        raise ValueError("Expected horizons 1--10 for every final model. Problems: " + "; ".join(problems))
    if "Kaplan" in set(seed_df["model"]):
        raise ValueError("Kaplan-Meier must not be included in predictive horizon figures.")


def aggregate_seed_rows(seed_df):
    agg = (
        seed_df.groupby(["landmark_hours", "model", "horizon_day"], as_index=False)
        .agg(
            mean_c_index=("c_index", "mean"),
            std_c_index=("c_index", lambda s: float(s.std(ddof=0)) if len(s) > 1 else np.nan),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["landmark_hours", "model", "horizon_day"])
    )
    return agg


def select_best_static_by_landmark(seed_df):
    selections = {}
    for landmark_hours in sorted(seed_df["landmark_hours"].unique()):
        scores = []
        for model_label in CONVENTIONAL_STATIC:
            result_path = (
                PROJECT_ROOT
                / f"outputs/landmark_{landmark_hours}h/static/final/{_model_key(model_label)}/final_seed_results.csv"
            )
            if result_path.exists():
                result_df = pd.read_csv(result_path)
                if "test_ctd_antolini" in result_df.columns:
                    scores.append((model_label, float(result_df["test_ctd_antolini"].mean()), "test_ctd_antolini"))
                    continue
            subset = seed_df[(seed_df["landmark_hours"].eq(landmark_hours)) & (seed_df["model"].eq(model_label))]
            if not subset.empty:
                scores.append((model_label, float(subset["c_index"].mean()), "test_mean_horizon_c_index"))
        if not scores:
            raise ValueError(f"Could not select best static model for landmark {landmark_hours}h.")
        selections[int(landmark_hours)] = max(scores, key=lambda item: item[1])
    return selections


def _model_key(model_label):
    for key, label in STATIC_MODELS.items():
        if label == model_label:
            return key
    raise KeyError(model_label)


def _panel_y_limits(plot_df):
    y = plot_df["mean_c_index"].dropna()
    if y.empty:
        return 0.60, 0.86
    lower = max(0.0, min(0.60, float(y.min()) - 0.015))
    upper = min(1.0, max(0.86, float(y.max()) + 0.015))
    return lower, upper


def _plot_panel(ax, plot_df, landmark_hours, model_order, best_static=None, show_bands=False):
    panel = plot_df[plot_df["landmark_hours"].eq(landmark_hours)]
    if panel.empty:
        ax.set_visible(False)
        return
    for model in model_order:
        source_model = model
        label = model
        if model == "Best static":
            if best_static is None:
                continue
            source_model = best_static[0]
            label = "Best static"
        line = panel[panel["model"].eq(source_model)].sort_values("horizon_day")
        if line.empty:
            continue
        style = STYLE[model if model == "Best static" else source_model]
        x = line["horizon_day"].to_numpy(dtype=float)
        y = line["mean_c_index"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4.5,
        )
        if show_bands and line["std_c_index"].notna().any():
            std = line["std_c_index"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - std, y + std, color=style["color"], alpha=0.10, linewidth=0)
    ax.set_title(f"Landmark {landmark_hours}h")
    if best_static is not None:
        ax.text(
            0.98,
            0.04,
            f"Best static: {best_static[0]}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    ax.set_xlabel("Horizon from landmark (days)")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_ylim(*_panel_y_limits(panel))


def _save_figure(fig, output_base):
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def plot_main(agg_df, landmarks, best_static, output_dir):
    fig, axes = plt.subplots(1, len(landmarks), figsize=(6.6 * len(landmarks), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, landmark_hours in zip(axes, landmarks):
        _plot_panel(
            ax,
            agg_df,
            landmark_hours,
            MAIN_MODEL_ORDER,
            best_static=best_static.get(int(landmark_hours)),
            show_bands=True,
        )
    axes[0].set_ylabel("Test horizon C-index")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Test C-index by prediction horizon", y=1.02)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return _save_figure(fig, output_dir / "cindex_horizon_main_48_72")


def plot_appendix(agg_df, landmarks, output_dir):
    fig, axes = plt.subplots(1, len(landmarks), figsize=(6.6 * len(landmarks), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, landmark_hours in zip(axes, landmarks):
        _plot_panel(ax, agg_df, landmark_hours, APPENDIX_MODEL_ORDER, show_bands=False)
    axes[0].set_ylabel("Test horizon C-index")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Test C-index by prediction horizon: all final models", y=1.02)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return _save_figure(fig, output_dir / "cindex_horizon_appendix_all_models")


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
    parser = argparse.ArgumentParser(description="Plot final landmark test C-index by horizon.")
    parser.add_argument("--landmarks", nargs="+", type=int, default=[48, 72])
    parser.add_argument("--include-24", action="store_true", help="Also include 24h if final outputs exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    landmarks = list(dict.fromkeys(([24] if args.include_24 else []) + args.landmarks))
    seed_df = collect_seed_rows(landmarks)
    available_landmarks = sorted(seed_df["landmark_hours"].unique().astype(int).tolist())
    if not available_landmarks:
        raise ValueError(f"No final landmark outputs found for requested landmarks: {landmarks}")
    validate_seed_rows(seed_df)
    agg_df = aggregate_seed_rows(seed_df)

    output_dir = PROJECT_ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cindex_horizon_plot_data.csv"
    agg_df[["landmark_hours", "model", "horizon_day", "mean_c_index", "std_c_index", "n_seeds"]].to_csv(
        csv_path, index=False
    )

    main_landmarks = [landmark for landmark in [48, 72] if landmark in available_landmarks]
    if main_landmarks != [48, 72]:
        raise ValueError("The main figure requires final 48h and 72h outputs.")
    best_static = select_best_static_by_landmark(seed_df[seed_df["landmark_hours"].isin(main_landmarks)])
    main_paths = plot_main(agg_df, main_landmarks, best_static, output_dir)
    appendix_paths = plot_appendix(agg_df, available_landmarks, output_dir)
    copied = copy_for_latex([main_paths[0], appendix_paths[0]], PROJECT_ROOT / "Imagenes/Bitmap")

    print("Wrote:", csv_path)
    for path in [*main_paths, *appendix_paths, *copied]:
        print("Wrote:", path)
    print("Best static selections:")
    for landmark_hours in main_landmarks:
        model, score, criterion = best_static[landmark_hours]
        print(f"  {landmark_hours}h: {model} by {criterion}={score:.6f}")
    missing = sorted(set(landmarks) - set(available_landmarks))
    if missing:
        print("Skipped landmarks without final horizon outputs:", missing)


if __name__ == "__main__":
    main()
