import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_dynamic_deephit_faithful_tuning_impl import build_run_config
from src.models.landmark_dynamic.common import load_yaml
from src.models.landmark_dynamic.train_dynamic_deephit_faithful import train_dynamic_deephit_faithful
from src.utils.logger import get_logger


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


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
    content = """# Dynamic-DeepHit Faithful 72h Outputs

This folder is isolated from the previous `dynamic_landmark` outputs and uses the
same prepared train/validation/test dataset as `dysurv_faithful_72h`.

- `tuning_results.csv`: validation-only candidates and collapse diagnostics.
- `best_hyperparameters.json`: selected non-collapsed candidate when available.
- `final_seed_results.csv`: final seeds 42, 123 and 2026.
- `final_seed_summary.json`: aggregate final metrics and collapsed-seed list.
- `tuning/`: per-candidate configs, checkpoints, epoch metrics and predictions.
- `final/`: per-seed validation/test predictions and checkpoints.
- `dynamic_deephit_faithful_audit_report.md`: implementation and probability audit.

Empty result files with `status: not_run` are placeholders, not experiment results.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run_tiny_overfit(base: dict, device: str):
    params = dict(base["audit"]["tiny_overfit"])
    sample_size = int(params.pop("sample_size"))
    output_dir = Path(base["paths"]["outputs_dir"]) / "audit" / "tiny_overfit"
    config = build_run_config(base, "tiny_overfit", params, output_dir, int(base["tuning"]["seed"]), sample_size, device, include_test=False)
    return train_dynamic_deephit_faithful(config, get_logger("audit_dynamic_deephit_faithful_72h"))


def _risk_summary(output_dir: Path, split: str):
    files = sorted(output_dir.glob(f"final/seed_*/predictions/{split}_survival_predictions.parquet"))
    rows = []
    for path in files:
        frame = pd.read_parquet(path, columns=["risk10", "tail_probability"])
        rows.append({
            "seed": int(path.parts[-3].split("_")[-1]),
            "mean": float(frame["risk10"].mean()),
            "std": float(frame["risk10"].std(ddof=0)),
            "range": float(frame["risk10"].max() - frame["risk10"].min()),
            "unique_rounded_6": int(frame["risk10"].round(6).nunique()),
            "mean_tail": float(frame["tail_probability"].mean()),
        })
    return rows


def write_report(base: dict):
    output_dir = Path(base["paths"]["outputs_dir"])
    tuning = pd.read_csv(output_dir / "tuning_results.csv") if (output_dir / "tuning_results.csv").exists() else pd.DataFrame()
    smoke_path = output_dir / "smoke" / "tuning_results.csv"
    smoke = pd.read_csv(smoke_path) if smoke_path.exists() else pd.DataFrame()
    final = pd.read_csv(output_dir / "final_seed_results.csv") if (output_dir / "final_seed_results.csv").exists() else pd.DataFrame()
    tiny_path = output_dir / "audit" / "tiny_overfit" / "metrics" / "epoch_metrics.csv"
    tiny = pd.read_csv(tiny_path) if tiny_path.exists() else pd.DataFrame()
    lines = [
        "# Dynamic-DeepHit Faithful 72h Audit Report",
        "",
        "## 1. Relation to original Dynamic-DeepHit",
        "",
        "- Preserved: LSTM embedding, next-step longitudinal network, temporal attention, cause-specific network, PMF NLL and pairwise ranking loss.",
        "- Preserved: one risk and a softmax event-time distribution.",
        "- Adapted: an explicit tail category represents survival beyond day 10.",
        "- Adapted: targets are relative to the 72-hour landmark and use the fixed TFG patient splits.",
        "",
        "## 2. Shared practices with DySurv faithful",
        "",
        "- Uses `data/processed/dysurv_faithful_72h/` without creating new splits.",
        "- Uses temporal clinical variables plus repeated standardized static variables; masks are not input channels.",
        "- Test is locked during tuning; smoke outputs are isolated.",
        "- Saves best/last checkpoints, config snapshots, epoch metrics, complete patient predictions and curve examples.",
        "",
        "## 3. Longitudinal target",
        "",
        "The longitudinal auxiliary head predicts only the next temporal clinical vector. Repeated static variables can inform the encoder and attention but are excluded from longitudinal MSE to prevent a trivial repeated-static objective.",
        "",
        "## 4. Probability and collapse diagnostics",
        "",
        f"- Tuning candidates recorded: {len(tuning)}.",
        f"- Final seeds recorded: {len(final)}.",
    ]
    if not tuning.empty and "collapse_suspected" in tuning:
        lines.append(f"- Tuning candidates flagged: {int(tuning['collapse_suspected'].fillna(True).astype(bool).sum())}.")
    if not smoke.empty:
        selected = smoke.sort_values(["validation_ctd_antolini", "validation_ibll"], ascending=[False, True]).iloc[0]
        lines.extend([
            "",
            "### Smoke run",
            "",
            f"- Candidates: {len(smoke)}; sample size: {int(selected['sample_size'])} per split.",
            f"- Validation Ctd: {selected['validation_ctd_antolini']:.6f}.",
            f"- Validation IBLL: {selected['validation_ibll']:.6f}.",
            f"- Validation risk10 std: {selected['validation_std_risk10']:.6f}.",
            f"- Mean tail probability: {selected['validation_mean_tail_probability']:.6f}.",
            f"- Collapse flag: {_as_bool(selected['collapse_suspected'])}.",
        ])
    lines.extend(["", "## 5. Tiny-overfit result", ""])
    if tiny.empty:
        lines.append("Not run. Execute this script with `--run-tiny-overfit`.")
    else:
        first, last = tiny.iloc[0], tiny.iloc[-1]
        lines.extend([
            f"- Epochs completed: {len(tiny)}.",
            f"- Train total loss: {first['train_total_loss']:.6f} -> {last['train_total_loss']:.6f}.",
            f"- Train NLL: {first['train_nll_loss']:.6f} -> {last['train_nll_loss']:.6f}.",
            f"- Final train risk10 std: {last['train_std_risk10']:.6f}.",
            f"- Final validation risk10 std: {last['validation_std_risk10']:.6f}.",
            f"- Final validation collapse flag: {_as_bool(last['collapse_suspected'])}.",
        ])
    lines.extend(["", "## 6. Final risk10 and tail distributions", ""])
    for split in ["validation", "test"]:
        rows = _risk_summary(output_dir, split)
        if not rows:
            lines.append(f"- {split}: no final predictions yet.")
        for row in rows:
            lines.append(f"- {split}, seed {row['seed']}: risk mean={row['mean']:.6f}, std={row['std']:.6f}, range={row['range']:.6f}, unique6={row['unique_rounded_6']}, tail mean={row['mean_tail']:.6f}.")
    lines.extend([
        "",
        "## 7. Validity decision",
        "",
        "Pending full validation-only tuning and final seeds." if final.empty else "Review final metrics, curves and collapse flags before thesis use.",
    ])
    (output_dir / "dynamic_deephit_faithful_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit Dynamic-DeepHit faithful 72h outputs.")
    parser.add_argument("--config", default="configs/landmark_dynamic_deephit_faithful.yaml")
    parser.add_argument("--run-tiny-overfit", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    base = load_yaml(args.config)
    output_dir = Path(base["paths"]["outputs_dir"])
    initialize_outputs(output_dir)
    write_readme(output_dir)
    if args.run_tiny_overfit:
        run_tiny_overfit(base, args.device)
    write_report(base)
    print(json.dumps({"output_dir": str(output_dir), "tiny_overfit_run": bool(args.run_tiny_overfit)}, indent=2))


if __name__ == "__main__":
    main()
