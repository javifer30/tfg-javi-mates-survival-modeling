import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tune_static_models import (
    metric_value,
    prepare_run_config,
    run_training,
    save_config_snapshot,
    timestamp_utc,
)
from src.models.static_common import save_json
from src.utils.config import load_yaml
from src.utils.logger import get_logger


FINAL_SEEDS = [42, 123, 2026]


def _requested_models(tuning_config, requested):
    available = tuning_config.get("models", {})
    models = requested or list(available)
    unknown = [model for model in models if model not in available]
    if unknown:
        raise ValueError(f"Unknown final static model(s): {unknown}")
    return models


def _load_best_hyperparameters(tuning_output_dir, model_name):
    path = Path(tuning_output_dir) / model_name / "best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing selected tuning result for {model_name}: {path}. "
            "Run scripts/tune_static_models.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["config_id"], json.loads(data["hyperparameters"])


def _validate_final_seeds(seeds):
    if [int(seed) for seed in seeds] != FINAL_SEEDS:
        raise ValueError(f"Final static seeds must be exactly {FINAL_SEEDS}")


def final_summary_row(model_name, config_id, hyperparameters, seed, metrics, run_dir):
    return {
        "timestamp": timestamp_utc(),
        "model": model_name,
        "selected_config_id": config_id,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "seed": int(seed),
        "validation_ctd_antolini": metric_value(metrics, "validation", "ctd_antolini"),
        "validation_ibll": metric_value(metrics, "validation", "ibll"),
        "validation_nbll": metric_value(metrics, "validation", "nbll"),
        "validation_ibs": metric_value(metrics, "validation", "ibs"),
        "validation_mean_horizon_c_index": metric_value(metrics, "validation", "mean_horizon_c_index"),
        "test_ctd_antolini": metric_value(metrics, "test", "ctd_antolini"),
        "test_ibll": metric_value(metrics, "test", "ibll"),
        "test_nbll": metric_value(metrics, "test", "nbll"),
        "test_ibs": metric_value(metrics, "test", "ibs"),
        "test_mean_horizon_c_index": metric_value(metrics, "test", "mean_horizon_c_index"),
        "metrics_path": str(Path(run_dir) / "metrics" / model_name / f"{model_name}_metrics.json"),
        "config_snapshot_path": str(Path(run_dir) / "config_snapshot.yaml"),
        "output_dir": str(run_dir),
        "selection_split": "validation",
    }


def run_final_static_seeds(config_path, requested_models=None, dry_run=False):
    logger = get_logger("run_final_static_seeds")
    tuning_config = load_yaml(config_path)
    seeds = tuning_config.get("final", {}).get("seeds", FINAL_SEEDS)
    _validate_final_seeds(seeds)
    tuning_output_dir = tuning_config.get("tuning", {}).get("output_dir", "outputs/tuning")
    output_root = tuning_config.get("final", {}).get("output_dir", "outputs/final_static")
    save_predictions = bool(tuning_config.get("final", {}).get("save_predictions", True))
    save_models = bool(tuning_config.get("final", {}).get("save_models", False))
    save_checkpoints = bool(tuning_config.get("final", {}).get("save_checkpoints", False))
    planned = []

    for model_name in _requested_models(tuning_config, requested_models):
        model_cfg = tuning_config["models"][model_name]
        base_config = load_yaml(model_cfg["base_config"])
        selected_config_id, hyperparameters = _load_best_hyperparameters(tuning_output_dir, model_name)
        rows = []
        for seed in seeds:
            run_config, run_dir = prepare_run_config(
                copy.deepcopy(base_config),
                model_name,
                selected_config_id,
                hyperparameters,
                int(seed),
                output_root,
                phase="final_static",
                include_test=True,
                save_predictions=save_predictions,
                save_models=save_models,
                save_checkpoints=save_checkpoints,
            )
            planned.append({"model": model_name, "seed": int(seed), "selected_config_id": selected_config_id, "output_dir": str(run_dir)})
            if dry_run:
                continue
            save_config_snapshot(run_config, run_dir)
            metrics = run_training(run_config, logger)
            rows.append(final_summary_row(model_name, selected_config_id, hyperparameters, int(seed), metrics, run_dir))

        if rows:
            model_output_dir = Path(output_root) / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(model_output_dir / "final_seed_results.csv", index=False)
            save_json({"model": model_name, "selected_config_id": selected_config_id, "seeds": FINAL_SEEDS}, model_output_dir / "final_seed_summary.json")
    return planned


def main():
    parser = argparse.ArgumentParser(description="Run final static survival models with the selected validation hyperparameters.")
    parser.add_argument("--config", default="configs/static_tuning.yaml")
    parser.add_argument("--models", nargs="*", help="Subset of models to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print/validate planned final runs without training.")
    args = parser.parse_args()
    planned = run_final_static_seeds(args.config, requested_models=args.models, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
