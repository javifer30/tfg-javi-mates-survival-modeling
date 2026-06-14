import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_exclusions(values):
    exclusions = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid exclusion '{value}'. Use format model:seed, for example dysurv:123.")
        model, seed = value.split(":", 1)
        exclusions.add((model, seed))
    return exclusions


def _selection_path(predictions_dir, model):
    return predictions_dir / f"example_survival_selection_{model}.csv"


def _plot_one_run(model, seed_dir, output_dir):
    predictions_dir = seed_dir / "predictions"
    curves_path = predictions_dir / "survival_curve_examples.csv"
    selection_path = _selection_path(predictions_dir, model)
    if not curves_path.exists() or not selection_path.exists():
        return None

    curves = pd.read_csv(curves_path)
    selection = pd.read_csv(selection_path)
    patient_cols = [col for col in curves.columns if col != "time_days"]
    if not patient_cols:
        return None

    n_cols = min(3, len(patient_cols))
    n_rows = (len(patient_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    selection = selection.copy()
    selection["column_index"] = selection["column_index"].astype(str)
    selection_by_col = selection.set_index("column_index")

    for idx, patient_col in enumerate(patient_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.plot(curves["time_days"], curves[patient_col], marker="o", linewidth=2)

        if patient_col in selection_by_col.index:
            patient = selection_by_col.loc[patient_col]
            duration = float(patient["duration_eval_days"])
            event = int(patient["event_eval"])
            risk = float(patient["risk_at_10d"])
            ax.axvline(duration, color="0.35", linestyle="--", linewidth=1)
            status = "event" if event == 1 else "censored"
            ax.set_title(f"id={patient_col}, {status}, obs={duration:.2f}d, risk10={risk:.3f}", fontsize=10)
        else:
            ax.set_title(f"id={patient_col}", fontsize=10)

        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Days after 72h landmark")
        ax.set_ylabel("Predicted survival S(t)")

    for idx in range(len(patient_cols), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    seed = seed_dir.name.replace("seed_", "")
    fig.suptitle(f"{model} survival curve examples, seed {seed}", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"survival_curves_{model}_seed_{seed}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot dynamic_72h example survival curves from final prediction CSVs."
    )
    parser.add_argument("--outputs-dir", default="outputs/dynamic_72h_bis")
    parser.add_argument("--figures-dir", default="outputs/figures/dynamic_72h_bis")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude a model seed using model:seed, for example --exclude dysurv:123.",
    )
    args = parser.parse_args()

    final_root = Path(args.outputs_dir) / "final"
    figures_dir = Path(args.figures_dir)
    exclusions = _parse_exclusions(args.exclude)

    written = []
    for model_dir in sorted(final_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for seed_dir in sorted(model_dir.glob("seed_*")):
            seed = seed_dir.name.replace("seed_", "")
            if (model, seed) in exclusions:
                continue
            out_path = _plot_one_run(model, seed_dir, figures_dir)
            if out_path is not None:
                written.append(out_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
