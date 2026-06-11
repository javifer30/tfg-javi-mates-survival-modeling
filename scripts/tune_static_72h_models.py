import argparse
import copy
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.static_72h_pycox import train_static_72h_model
from src.utils.config import load_yaml, resolve_path
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def timestamp_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def expand_grid(grid):
    if not grid:
        return [{}]
    keys = list(grid)
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def _resolve_config_paths(config):
    for key, value in config.get("paths", {}).items():
        if isinstance(value, str):
            config["paths"][key] = str(resolve_path(value))
    return config


def _model_list(config, requested):
    available = config.get("models", {})
    models = requested or list(available)
    unknown = [model for model in models if model not in available]
    if unknown:
        raise ValueError(f"Unknown static_72h model(s): {unknown}")
    return models


def _run_config(tuning_config, model_name, config_id, hyperparameters, output_dir, include_test, seed):
    run_config = {
        "seed": int(seed),
        "experiment": copy.deepcopy(tuning_config["experiment"]),
        "paths": {
            "processed_dir": tuning_config["paths"]["processed_dir"],
            "outputs_dir": str(output_dir),
        },
        "model": {"name": model_name, **copy.deepcopy(hyperparameters)},
        "evaluation": {
            "splits": ["train", "validation", "test"] if include_test else ["train", "validation"],
            "allow_test_metrics": bool(include_test),
            "evaluation_time_grid": copy.deepcopy(tuning_config["evaluation"]["evaluation_time_grid"]),
            "horizon_times": copy.deepcopy(tuning_config["evaluation"]["horizon_times"]),
            "save_example_curves": bool(include_test and tuning_config["evaluation"].get("save_example_curves", True)),
            "n_example_patients": int(tuning_config["evaluation"].get("n_example_patients", 9)),
        },
        "run": {
            "phase": "final" if include_test else "tuning",
            "model": model_name,
            "config_id": config_id,
            "hyperparameters": copy.deepcopy(hyperparameters),
            "seed": int(seed),
            "timestamp": timestamp_utc(),
        },
    }
    return run_config


def save_config_snapshot(run_config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_snapshot.yaml"
    path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return path


def metric(metrics, split_name, metric_name):
    value = metrics.get("splits", {}).get(split_name, {}).get(metric_name)
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


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
        "config_snapshot_path": str(Path(output_dir) / "config_snapshot.yaml"),
        "metrics_path": str(Path(output_dir) / "metrics" / model_name / f"{model_name}_metrics.json"),
    }


def select_best_row(rows):
    candidates = [row for row in rows if row["status"] == "completed" and not math.isnan(row["validation_ctd_antolini"])]
    if not candidates:
        raise ValueError("No successful tuning candidate with finite validation Ctd")
    return sorted(
        candidates,
        key=lambda row: (
            -row["validation_ctd_antolini"],
            row["validation_ibll"] if not math.isnan(row["validation_ibll"]) else math.inf,
            row["validation_ibs"] if not math.isnan(row["validation_ibs"]) else math.inf,
            row["config_id"],
        ),
    )[0]


def tune_models(config_path, requested_models=None, dry_run=False, max_runs=None):
    logger = get_logger("tune_static_72h_models")
    tuning_config = load_yaml(config_path)
    seed = int(tuning_config["tuning"].get("seed", 42))
    output_root = Path(tuning_config["paths"].get("outputs_dir", "outputs/static_72h")) / "tuning"
    planned = []
    run_count = 0

    for model_name in _model_list(tuning_config, requested_models):
        rows = []
        for index, hyperparameters in enumerate(expand_grid(tuning_config["models"][model_name].get("grid", {})), start=1):
            if max_runs is not None and run_count >= int(max_runs):
                break
            config_id = f"{model_name}_cfg_{index:03d}"
            run_dir = output_root / model_name / config_id / f"seed_{seed}"
            run_config = _run_config(tuning_config, model_name, config_id, hyperparameters, run_dir, include_test=False, seed=seed)
            planned.append({"model": model_name, "config_id": config_id, "seed": seed, "output_dir": str(run_dir)})
            run_count += 1
            if dry_run:
                continue
            save_config_snapshot(run_config, run_dir)
            try:
                resolved = _resolve_config_paths(copy.deepcopy(run_config))
                set_seed(seed)
                metrics = train_static_72h_model(resolved, logger)
                rows.append(tuning_row(model_name, config_id, hyperparameters, seed, "completed", run_dir, metrics=metrics))
            except Exception as exc:
                logger.warning("static_72h tuning candidate failed: %s %s: %s", model_name, config_id, exc)
                rows.append(tuning_row(model_name, config_id, hyperparameters, seed, "failed", run_dir, error=repr(exc)))

        if rows:
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(model_dir / "tuning_results.csv", index=False)
            best = select_best_row(rows)
            (model_dir / "best_hyperparameters.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return planned


def main():
    parser = argparse.ArgumentParser(description="Validation-only tuning for static_72h_pycox models.")
    parser.add_argument("--config", default="configs/static_72h_tuning.yaml")
    parser.add_argument("--models", nargs="*", help="Subset of models to tune.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    planned = tune_models(args.config, requested_models=args.models, dry_run=args.dry_run, max_runs=args.max_runs)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
