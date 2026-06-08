"""Kaplan-Meier descriptive analysis for the static TFG cohort."""

from pathlib import Path

import pandas as pd

from src.data.static_dataset import EVENT_COL, TIME_COL
from src.models.static_common import load_static_splits, model_metrics_dir, save_json


def train_kaplan_meier(config, logger):
    try:
        from lifelines import KaplanMeierFitter
    except ImportError as exc:
        raise ImportError("lifelines is required for Kaplan-Meier analysis") from exc

    paths = config["paths"]
    _, _, test = load_static_splits(paths)
    df = test if config.get("use_test_split", True) else pd.concat(load_static_splits(paths), ignore_index=True)

    kmf = KaplanMeierFitter()
    kmf.fit(df[TIME_COL], event_observed=df[EVENT_COL], label="Kaplan-Meier")

    curve = kmf.survival_function_.reset_index()
    curve_path = Path(paths["predictions_dir"]) / "kaplan_meier_survival_curve.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)

    metrics = {
        "model": "kaplan_meier",
        "n_patients": int(len(df)),
        "events": int(df[EVENT_COL].sum()),
        "censored": int(len(df) - df[EVENT_COL].sum()),
        "median_survival_time": None if pd.isna(kmf.median_survival_time_) else float(kmf.median_survival_time_),
    }
    metrics_dir = model_metrics_dir(paths, "kaplan_meier")
    save_json(metrics, metrics_dir / "kaplan_meier_metrics.json")

    try:
        import matplotlib.pyplot as plt

        ax = kmf.plot_survival_function()
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival probability")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(Path(paths["figures_dir"]) / "kaplan_meier_survival.png", dpi=160)
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib is not installed; Kaplan-Meier figure was not created")

    logger.info("Kaplan-Meier descriptive outputs saved")
    return metrics
