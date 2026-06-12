import argparse
import copy
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dynamic_72h.common import expand_grid, load_yaml, metric, save_json, timestamp_utc
from src.models.dynamic_72h.train import train_dynamic_72h_model
from src.utils.logger import get_logger


def _model_list(config, requested):
    available = config.get("models", {})
    models = requested or list(available)
    unknown = [model for model in models if model not in available]
    if unknown:
        raise ValueError(f"Unknown dynamic_72h model(s): {unknown}")
    return models


def _run_config(tuning_config, model_name, config_id, hyperparameters, output_dir, include_test, seed, sample_size=None, device="auto"):
    return {
        "seed": int(seed),
        "device": device,
        "phase": "final" if include_test else "tuning",
        "include_test": bool(include_test),
        "sample_size": sample_size,
        "experiment": copy.deepcopy(tuning_config["experiment"]),
        "paths": {
            "dataset_dir": tuning_config["paths"]["dataset_dir"],
            "output_dir": str(output_dir),
            "audit_dir": str(Path(tuning_config["paths"].get("outputs_dir", "outputs/dynamic_72h")) / "audit" / model_name / config_id / f"seed_{seed}"),
        },
        "data": copy.deepcopy(tuning_config["data"]),
        "evaluation": copy.deepcopy(tuning_config["evaluation"]),
        "model": {"name": model_name, "params": copy.deepcopy(hyperparameters)},
        "run": {
            "model": model_name,
            "config_id": config_id,
            "hyperparameters": copy.deepcopy(hyperparameters),
            "seed": int(seed),
            "timestamp": timestamp_utc(),
        },
    }


def tuning_row(model_name, config_id, hyperparameters, seed, status, output_dir, metrics=None, error=None):
    metrics = metrics or {}
    return {
        "timestamp": timestamp_utc(),
        "model": model_name,
        "config_id": config_id,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "seed": int(seed),
        "status": status,
        "error": error,
        "validation_ctd_antolini": metric(metrics, "validation", "ctd_antolini"),
        "validation_ibs": metric(metrics, "validation", "ibs"),
        "validation_ibll": metric(metrics, "validation", "ibll"),
        "validation_nbll": metric(metrics, "validation", "nbll"),
        "validation_mean_horizon_c_index": metric(metrics, "validation", "mean_horizon_c_index"),
        "test_metrics_recorded": "test" in metrics.get("splits", {}),
        "output_dir": str(output_dir),
        "metrics_path": str(Path(output_dir) / "metrics" / "metrics.json"),
    }


def select_best_row(rows):
    candidates = [row for row in rows if row["status"] == "completed" and not math.isnan(row["validation_ctd_antolini"])]
    if not candidates:
        raise ValueError("No successful dynamic_72h tuning candidate with finite validation Ctd")
    return sorted(
        candidates,
        key=lambda row: (
            -row["validation_ctd_antolini"],
            row["validation_ibll"] if not math.isnan(row["validation_ibll"]) else math.inf,
            row["validation_ibs"] if not math.isnan(row["validation_ibs"]) else math.inf,
            row["config_id"],
        ),
    )[0]


def tune_models(config_path, requested_models=None, dry_run=False, max_runs=None, sample_size=None, device="auto", force=False):
    logger = get_logger("tune_dynamic_72h_models")
    tuning_config = load_yaml(config_path)
    seed = int(tuning_config["tuning"].get("seed", 42))
    output_root = Path(tuning_config["paths"].get("outputs_dir", "outputs/dynamic_72h")) / "tuning"
    planned = []
    run_count = 0
    for model_name in _model_list(tuning_config, requested_models):
        rows = []
        for index, hyperparameters in enumerate(expand_grid(tuning_config["models"][model_name].get("grid", {})), start=1):
            if max_runs is not None and run_count >= int(max_runs):
                break
            if model_name == "dynamic_deephit" and float(hyperparameters.get("alpha", 0.0)) + float(hyperparameters.get("beta", 0.0)) > 1.0:
                continue
            config_id = f"{model_name}_cfg_{index:03d}"
            run_dir = output_root / model_name / config_id / f"seed_{seed}"
            run_config = _run_config(tuning_config, model_name, config_id, hyperparameters, run_dir, False, seed, sample_size, device)
            planned.append({"model": model_name, "config_id": config_id, "seed": seed, "output_dir": str(run_dir)})
            run_count += 1
            if dry_run:
                continue
            if run_dir.exists() and not force:
                logger.info("Reusing/overwriting existing dynamic_72h run directory contents: %s", run_dir)
            try:
                metrics = train_dynamic_72h_model(run_config, logger)
                rows.append(tuning_row(model_name, config_id, hyperparameters, seed, "completed", run_dir, metrics=metrics))
            except Exception as exc:
                logger.warning("dynamic_72h tuning candidate failed: %s %s: %s", model_name, config_id, exc)
                rows.append(tuning_row(model_name, config_id, hyperparameters, seed, "failed", run_dir, error=repr(exc)))
        if rows:
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(model_dir / "tuning_results.csv", index=False)
            best = select_best_row(rows)
            save_json(model_dir / "best_hyperparameters.json", best)
    return planned


def main():
    parser = argparse.ArgumentParser(description="Validation-only tuning for dynamic_72h models.")
    parser.add_argument("--config", default="configs/dynamic_72h_tuning.yaml")
    parser.add_argument("--model", "--models", dest="models", nargs="*", choices=["dysurv", "dynamic_deephit"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    planned = tune_models(args.config, args.models, args.dry_run, args.max_runs, args.sample_size, args.device, args.force)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()

