import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_dysurv_faithful_tuning_impl import build_run_config
from src.models.landmark_dynamic.common import load_yaml
from src.models.landmark_dynamic.train_dysurv_faithful import train_dysurv_faithful
from src.utils.logger import get_logger


def initialize_outputs(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholders = {
        "tuning_results.csv": ["config_id", "status", "validation_ctd_antolini", "validation_ibll", "collapse_suspected"],
        "final_seed_results.csv": ["seed", "status", "test_ctd_antolini", "test_ibll", "collapse_suspected"],
    }
    for name, columns in placeholders.items():
        path = output_dir / name
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
    for name in ["best_hyperparameters.json", "final_seed_summary.json"]:
        path = output_dir / name
        if not path.exists():
            path.write_text(json.dumps({"status": "not_run"}, indent=2), encoding="utf-8")


def write_readme(output_dir: Path):
    content = """# DySurv Faithful 72h Outputs

This folder is isolated from the previous `dynamic_landmark` DySurv outputs.

- `tuning_results.csv`: validation-only candidates and collapse diagnostics.
- `best_hyperparameters.json`: selected non-collapsed candidate when available.
- `final_seed_results.csv`: final seeds 42, 123 and 2026 after validation selection.
- `final_seed_summary.json`: aggregate final metrics and collapsed-seed list.
- `tuning/`: per-candidate configs, checkpoints, epoch metrics and validation predictions.
- `final/`: per-seed validation/test predictions and checkpoints.
- `dysurv_faithful_audit_report.md`: implementation and collapse audit.

Empty result files with `status: not_run` are placeholders, not experiment results.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run_tiny_overfit(base: dict, device: str):
    params = dict(base["audit"]["tiny_overfit"])
    sample_size = int(params.pop("sample_size"))
    params["patience"] = int(params["epochs"])
    params["metrics_every_n_epochs"] = 1
    output_dir = Path(base["paths"]["outputs_dir"]) / "audit" / "tiny_overfit"
    config = build_run_config(base, "tiny_overfit", params, output_dir, int(base["tuning"]["seed"]), sample_size, device, include_test=False)
    return train_dysurv_faithful(config, get_logger("audit_dysurv_faithful_72h"))


def _risk_summary(output_dir: Path, split: str):
    files = sorted(output_dir.glob(f"final/seed_*/predictions/{split}_survival_predictions.parquet"))
    rows = []
    for path in files:
        frame = pd.read_parquet(path, columns=["risk10"])
        rows.append({
            "seed": int(path.parts[-3].split("_")[-1]),
            "min": float(frame["risk10"].min()),
            "max": float(frame["risk10"].max()),
            "mean": float(frame["risk10"].mean()),
            "std": float(frame["risk10"].std(ddof=0)),
            "range": float(frame["risk10"].max() - frame["risk10"].min()),
            "unique_rounded_6": int(frame["risk10"].round(6).nunique()),
        })
    return rows


def write_report(base: dict, tiny_metrics=None):
    output_dir = Path(base["paths"]["outputs_dir"])
    tuning = pd.read_csv(output_dir / "tuning_results.csv") if (output_dir / "tuning_results.csv").exists() else pd.DataFrame()
    smoke_path = output_dir / "smoke" / "tuning_results.csv"
    smoke = pd.read_csv(smoke_path) if smoke_path.exists() else pd.DataFrame()
    final = pd.read_csv(output_dir / "final_seed_results.csv") if (output_dir / "final_seed_results.csv").exists() else pd.DataFrame()
    tiny_path = output_dir / "audit" / "tiny_overfit" / "metrics" / "epoch_metrics.csv"
    tiny_history = pd.read_csv(tiny_path) if tiny_path.exists() else pd.DataFrame()
    lines = [
        "# DySurv Faithful 72h Audit Report",
        "",
        "## 1. Differences from original DySurv",
        "",
        "- Preserved: 72-step LSTM encoder, 3x/5x/3x MLP capacity, latent dimension 20, variational sampling, LogisticHazard head and recurrent decoder.",
        "- Adapted: the target duration is not appended to the input and does not condition the decoder, preventing outcome leakage.",
        "- Adapted: the 10-day target is measured from the selected landmark using the existing patient-level train/validation/test splits.",
        "",
        "## 2. Differences from previous TFG DySurv pipeline",
        "",
        "- Input excludes observation masks as channels.",
        "- Reconstruction excludes masks and repeated static variables.",
        "- Decoder is recurrent rather than a single MLP producing `72 * input_dim` values.",
        "- Encoder/survival MLP defaults are `[294, 490, 294]` with latent dimension 20.",
        "- KL warm-up, collapse diagnostics, checkpoints and full patient predictions are mandatory.",
        "",
        "## 3. Why masks are not explicit inputs",
        "",
        "The faithful primary experiment tests clinical trajectories after DySurv-like within-patient imputation. `M_seq` is retained only to weight reconstruction and audit observed values; it cannot become an artificial predictive channel.",
        "",
        "## 4. Reconstruction target",
        "",
        "The recurrent decoder reconstructs only imputed temporal clinical variables. Static variables may enter the encoder at each timestep but are not reconstruction targets, so they cannot dominate MSE through repetition.",
        "",
        "## 5. Collapse diagnostics",
        "",
        f"- Tuning candidates recorded: {len(tuning)}.",
        f"- Final seeds recorded: {len(final)}.",
    ]
    if not tuning.empty and "collapse_suspected" in tuning:
        lines.append(f"- Tuning candidates flagged: {int(tuning['collapse_suspected'].fillna(True).astype(bool).sum())}.")
    if not final.empty and "collapse_suspected" in final:
        lines.append(f"- Final seeds flagged: {int(final['collapse_suspected'].fillna(True).astype(bool).sum())}.")
    if not smoke.empty:
        best_smoke = smoke.sort_values(["validation_ctd_antolini", "validation_ibll"], ascending=[False, True]).iloc[0]
        lines.extend([
            "",
            "### Smoke run",
            "",
            f"- Candidates: {len(smoke)}; sample size: {int(best_smoke['sample_size'])} per split.",
            f"- Selected validation Ctd: {best_smoke['validation_ctd_antolini']:.6f}.",
            f"- Selected validation IBLL: {best_smoke['validation_ibll']:.6f}.",
            f"- Selected validation risk10 std: {best_smoke['validation_std_risk10']:.6f}.",
            f"- Selected collapse flag: {bool(best_smoke['collapse_suspected'])}.",
            "- Smoke artifacts are isolated under `smoke/` and cannot be used by the final-seed script.",
        ])
    lines.extend(["", "## 6. Tiny-overfit result", ""])
    if not tiny_history.empty:
        first = tiny_history.iloc[0]
        last = tiny_history.iloc[-1]
        lines.extend([
            f"- Epochs completed: {len(tiny_history)}.",
            f"- Train survival loss: {first['train_survival_loss']:.6f} -> {last['train_survival_loss']:.6f}.",
            f"- Final train risk10 std: {last['train_std_risk10']:.6f}.",
            f"- Final validation risk10 std: {last['validation_std_risk10']:.6f}.",
            f"- Final validation collapse flag: {bool(last['collapse_suspected'])}.",
        ])
    else:
        lines.append("Not run. Execute this script with `--run-tiny-overfit` after preparing the dataset.")
    lines.extend(["", "## 7. Risk10 distribution", ""])
    for split in ["validation", "test"]:
        rows = _risk_summary(output_dir, split)
        if not rows:
            lines.append(f"- {split}: no final predictions yet.")
        for row in rows:
            lines.append(f"- {split}, seed {row['seed']}: mean={row['mean']:.6f}, std={row['std']:.6f}, range={row['range']:.6f}, unique6={row['unique_rounded_6']}.")
    lines.extend([
        "",
        "## 8. Example curves",
        "",
        "Each completed run stores low-risk, high-risk, early-event and long-censored examples under its `predictions/` directory.",
        "",
        "## 9. Validity decision",
        "",
    ])
    if final.empty:
        lines.append("Pending. Full validation-only tuning and three final seeds have not been completed.")
    elif "collapse_suspected" in final and final["collapse_suspected"].fillna(True).astype(bool).any():
        lines.append("Not yet valid for unconditional final reporting: at least one final seed is flagged for collapse and requires review.")
    else:
        lines.append("No automatic collapse criterion was triggered. Final validity still requires metric and curve review.")
    (output_dir / "dysurv_faithful_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit the DySurv faithful 72h pipeline and outputs.")
    parser.add_argument("--config", default="configs/landmark_dysurv_faithful.yaml")
    parser.add_argument("--run-tiny-overfit", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    base = load_yaml(args.config)
    output_dir = Path(base["paths"]["outputs_dir"])
    initialize_outputs(output_dir)
    write_readme(output_dir)
    tiny_metrics = run_tiny_overfit(base, args.device) if args.run_tiny_overfit else None
    write_report(base, tiny_metrics)
    print(json.dumps({"output_dir": str(output_dir), "tiny_overfit_run": bool(args.run_tiny_overfit)}, indent=2))


if __name__ == "__main__":
    main()
