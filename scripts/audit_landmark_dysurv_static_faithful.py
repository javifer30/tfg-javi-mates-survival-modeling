import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_dysurv_static_faithful_tuning_impl import build_run_config
from src.models.landmark_dynamic.common import load_yaml
from src.models.landmark_dynamic.train_dysurv_static_faithful import train_dysurv_static_faithful
from src.utils.logger import get_logger


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def initialize_outputs(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tuning").mkdir(exist_ok=True)
    (output_dir / "final").mkdir(exist_ok=True)
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
    content = """# DySurv Static Faithful 72h Outputs

This pipeline uses only `X_static` from the exact dataset and patient splits
prepared for `dysurv_faithful_72h`. It is isolated from all temporal pipelines.

- `tuning_results.csv`: validation-only candidates and collapse diagnostics.
- `best_hyperparameters.json`: reviewed validation selection.
- `final_seed_results.csv`: final seeds 42, 123 and 2026.
- `final_seed_summary.json`: final aggregate metrics.
- `tuning/`: per-candidate snapshots, checkpoints, metrics and validation predictions.
- `final/`: per-seed validation/test artifacts.
- `dysurv_static_faithful_audit_report.md`: implementation and validity audit.

Placeholder files with `status: not_run` are not experiment results.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_dataset_identity(base: dict):
    dataset_dir = Path(base["paths"]["prepared_dataset_dir"])
    output_dir = Path(base["paths"]["outputs_dir"]) / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_dir": str(dataset_dir),
        "comparison_basis": "Both static and temporal faithful pipelines read these exact split files.",
        "splits": {},
    }
    for split, filename in base["data"]["source_split_files"].items():
        with np.load(dataset_dir / filename) as data:
            patient_ids = np.asarray(data["patient_ids"]).astype(str)
            durations = np.asarray(data["duration_eval_days"], dtype="float32")
            events = np.asarray(data["event_eval"], dtype="int64")
            static_shape = list(data["X_static"].shape)
        identity = "\n".join(patient_ids.tolist()).encode("utf-8")
        target = np.column_stack([durations, events]).tobytes()
        summary["splits"][split] = {
            "source_file": filename,
            "n_patients": int(len(patient_ids)),
            "x_static_shape": static_shape,
            "patient_id_order_sha256": hashlib.sha256(identity).hexdigest(),
            "target_order_sha256": hashlib.sha256(target).hexdigest(),
        }
    (output_dir / "dataset_identity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_tiny_overfit(base: dict, device: str):
    params = dict(base["audit"]["tiny_overfit"])
    sample_size = int(params.pop("sample_size"))
    params["patience"] = int(params["epochs"])
    output_dir = Path(base["paths"]["outputs_dir"]) / "audit" / "tiny_overfit"
    config = build_run_config(base, "tiny_overfit", params, output_dir, int(base["tuning"]["seed"]), sample_size, device, False)
    return train_dysurv_static_faithful(config, get_logger("audit_dysurv_static_faithful_72h"))


def _risk_summary(output_dir: Path, split: str):
    rows = []
    for path in sorted(output_dir.glob(f"final/seed_*/predictions/{split}_survival_predictions.parquet")):
        frame = pd.read_parquet(path, columns=["risk10"])
        rows.append({
            "seed": int(path.parts[-3].split("_")[-1]),
            "mean": float(frame["risk10"].mean()),
            "std": float(frame["risk10"].std(ddof=0)),
            "range": float(frame["risk10"].max() - frame["risk10"].min()),
            "unique6": int(frame["risk10"].round(6).nunique()),
        })
    return rows


def write_report(base: dict, identity: dict):
    output_dir = Path(base["paths"]["outputs_dir"])
    tuning = pd.read_csv(output_dir / "tuning_results.csv")
    smoke_path = output_dir / "smoke" / "tuning_results.csv"
    smoke = pd.read_csv(smoke_path) if smoke_path.exists() else pd.DataFrame()
    final = pd.read_csv(output_dir / "final_seed_results.csv")
    tiny_path = output_dir / "audit" / "tiny_overfit" / "metrics" / "epoch_metrics.csv"
    tiny = pd.read_csv(tiny_path) if tiny_path.exists() else pd.DataFrame()
    lines = [
        "# DySurv Static Faithful 72h Audit Report",
        "",
        "## 1. Executive summary",
        "",
        "This pipeline compares a static MLP-VAE DySurv against temporal faithful DySurv using the same selected landmark cohort, patient splits, targets, horizon and metrics. Full tuning and final validity remain pending unless results are listed below.",
        "",
        "## 2. Original static DySurv notebooks",
        "",
        "The final `DySurv` sections of GBSG, METABRIC, SUPPORT, NWTCO, SAC3 and SAC_ADMIN share the same structure: a ReLU encoder `F -> 3F -> 5F -> 3F`, separate latent mean/log-variance layers, latent dimension 20, a decoder `20 -> 3F -> 5F -> 3F -> F` without intermediate activations, a ReLU LogisticHazard head, 10 durations, MSE reconstruction and VAE KL.",
        "",
        "Repeated notebook issues are corrected rather than copied: the encoder has a missing parenthesis; `Loss(0.5)` conflicts with later vector indexing; prediction samples a random latent even at evaluation; and summed KL divided by 10 changes scale with batch size. SAC_ADMIN additionally uses learning rate 0.01 and batch size 512, while the other five use 0.001 and 256. Target column names vary by benchmark.",
        "",
        "## 3. Clean implementation decisions",
        "",
        "- Encoder and survival hidden layers retain ReLU and the original 3x/5x/3x widths.",
        "- Decoder activation defaults to `none`, matching the notebook code; `relu` remains an explicit non-primary option.",
        "- Evaluation uses latent mean `mu` deterministically; stochastic reparameterization is training-only.",
        "- KL is averaged per patient, making its scale independent of batch size.",
        "- Loss weights are explicit and sum to one; exact author weights are not claimed because the notebooks are inconsistent.",
        "",
        "## 4. Difference from temporal faithful DySurv",
        "",
        "Static faithful DySurv receives only `X_static [N,F]`, has no LSTM, no temporal values, no masks and no repeated static sequence. Its MLP decoder reconstructs `X_static`. Temporal faithful DySurv encodes the 72-step clinical trajectory plus repeated static covariates and reconstructs only temporal clinical variables with a recurrent decoder.",
        "",
        "## 5. Dataset and leakage checks",
        "",
        "The loader reads `X_static`, patient IDs, durations and events directly from `data/processed/dysurv_faithful_72h/{train,val,test}_dynamic_landmark.npz`. It does not create splits or load `X_seq`/`M_seq` as model inputs. Static features are the existing train-standardized 28-column representation.",
        f"Patient counts are train={identity['splits']['train']['n_patients']}, validation={identity['splits']['validation']['n_patients']} and test={identity['splits']['test']['n_patients']}. Ordered patient-ID and target hashes are stored in `audit/dataset_identity.json`.",
        "",
        "## 6. Architecture and loss",
        "",
        "Default architecture: `28 -> 84 -> 140 -> 84 -> (mu,logvar:20)`, decoder `20 -> 84 -> 140 -> 84 -> 28`, and survival head `20 -> 84 -> 140 -> 84 -> 10`. Total loss is `w_surv*NLL + w_recon*MSE + effective_w_kl*KL`; KL warm-up reduces early posterior pressure while the reconstruction and survival representations form.",
        "",
        "## 7. Tiny-overfit",
        "",
    ]
    if tiny.empty:
        lines.append("Not run.")
    else:
        first, last = tiny.iloc[0], tiny.iloc[-1]
        lines.extend([
            f"- Epochs: {len(tiny)}.",
            f"- Train survival loss: {first['train_survival_loss']:.6f} -> {last['train_survival_loss']:.6f}.",
            f"- Train reconstruction loss: {first['train_reconstruction_loss']:.6f} -> {last['train_reconstruction_loss']:.6f}.",
            f"- Final train risk10 std: {last['train_std_risk10']:.6f}.",
            f"- Final validation collapse flag: {_as_bool(last['collapse_suspected'])}.",
        ])
    lines.extend(["", "## 8. Smoke run", ""])
    if smoke.empty:
        lines.append("Not run.")
    else:
        selected = smoke.sort_values(["validation_ctd_antolini", "validation_ibll"], ascending=[False, True]).iloc[0]
        lines.extend([
            f"- Candidates: {len(smoke)}; sample size: {int(selected['sample_size'])}.",
            f"- Validation Ctd: {selected['validation_ctd_antolini']:.6f}.",
            f"- Validation IBS: {selected['validation_ibs']:.6f}.",
            f"- Validation IBLL: {selected['validation_ibll']:.6f}.",
            f"- Validation risk10 std: {selected['validation_std_risk10']:.6f}.",
            f"- Collapse flag: {_as_bool(selected['collapse_suspected'])}.",
        ])
    lines.extend([
        "",
        "## 9. Tuning and collapse diagnostics",
        "",
        f"- Full tuning candidates recorded: {len(tuning)}.",
        f"- Final seeds recorded: {len(final)}.",
    ])
    if not tuning.empty and "collapse_suspected" in tuning:
        lines.append(f"- Full tuning candidates flagged: {sum(_as_bool(value) for value in tuning['collapse_suspected'])}.")
    lines.extend(["", "## 10. Risk10 distributions and example curves", ""])
    for split in ["validation", "test"]:
        rows = _risk_summary(output_dir, split)
        if not rows:
            lines.append(f"- {split}: no final predictions yet.")
        for row in rows:
            lines.append(f"- {split}, seed {row['seed']}: mean={row['mean']:.6f}, std={row['std']:.6f}, range={row['range']:.6f}, unique6={row['unique6']}.")
    lines.append("Each completed run stores low/high-risk, early-event and long-censored curve examples under `predictions/`.")
    lines.extend(["", "## 11. Validity and recommendation", ""])
    if final.empty:
        lines.append("Pending full validation-only tuning, curve review and final seeds. Do not include this model in final comparative results yet.")
    elif any(_as_bool(value) for value in final["collapse_suspected"]):
        lines.append("At least one final seed is flagged for collapse; review before thesis inclusion.")
    else:
        lines.append("No automatic collapse flag was triggered. Compare final metrics and curves directly with temporal faithful DySurv before deciding whether the VAE adds value with static inputs.")
    (output_dir / "dysurv_static_faithful_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit static-only DySurv faithful 72h outputs.")
    parser.add_argument("--config", default="configs/landmark_dysurv_static_faithful.yaml")
    parser.add_argument("--run-tiny-overfit", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    base = load_yaml(args.config)
    output_dir = Path(base["paths"]["outputs_dir"])
    initialize_outputs(output_dir)
    write_readme(output_dir)
    identity = write_dataset_identity(base)
    if args.run_tiny_overfit:
        run_tiny_overfit(base, args.device)
    write_report(base, identity)
    print(json.dumps({"output_dir": str(output_dir), "tiny_overfit_run": bool(args.run_tiny_overfit)}, indent=2))


if __name__ == "__main__":
    main()
