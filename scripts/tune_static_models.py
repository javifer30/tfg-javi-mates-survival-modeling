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

from scripts.train_static_model import TRAINERS, _resolve_config_paths
from src.models.static_common import ensure_output_dirs, save_json
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


FIXED_EVALUATION_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
VALIDATION_SPLITS = ["train", "validation"]


def timestamp_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def expand_grid(grid):
    if not grid:
        return [{}]
    keys = list(grid)
    values = []
    for key in keys:
        raw = grid[key]
        values.append(raw if isinstance(raw, list) else [raw])
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def apply_hyperparameters(config, hyperparameters):
    run_config = copy.deepcopy(config)
    run_config.setdefault("model", {}).update(hyperparameters)
    return run_config


def _output_paths(run_dir):
    return {
        "models_dir": str(run_dir / "models"),
        "checkpoints_dir": str(run_dir / "checkpoints"),
        "metrics_dir": str(run_dir / "metrics"),
        "predictions_dir": str(run_dir / "predictions"),
        "figures_dir": str(run_dir / "figures"),
    }


def _disable_heavy_artifacts(model_cfg, save_models=False, save_checkpoints=False):
    model_cfg["save_model"] = bool(save_models)
    if not save_checkpoints:
        model_cfg["save_best_checkpoint"] = False
        model_cfg["save_last_checkpoint"] = False
        model_cfg["save_every_n_epochs"] = None
    return model_cfg


def prepare_run_config(
    base_config,
    model_name,
    config_id,
    hyperparameters,
    seed,
    output_root,
    phase,
    include_test,
    save_predictions,
    save_models=False,
    save_checkpoints=False,
):
    run_config = apply_hyperparameters(base_config, hyperparameters)
    run_config["seed"] = int(seed)
    run_config.setdefault("model", {})["name"] = model_name
    run_config["model"]["evaluation_time_grid"] = list(FIXED_EVALUATION_GRID)
    run_config["model"]["horizon_times"] = list(FIXED_EVALUATION_GRID)
    _disable_heavy_artifacts(run_config["model"], save_models=save_models, save_checkpoints=save_checkpoints)

    if phase == "tuning":
        run_dir = Path(output_root) / model_name / config_id / f"seed_{seed}"
    elif phase == "final_static":
        run_dir = Path(output_root) / model_name / f"seed_{seed}"
    else:
        raise ValueError(f"Unsupported static run phase: {phase}")

    run_config.setdefault("paths", {}).update(_output_paths(run_dir))
    eval_cfg = run_config.setdefault("evaluation", {})
    eval_cfg["evaluation_time_grid"] = list(FIXED_EVALUATION_GRID)
    eval_cfg["horizon_times"] = list(FIXED_EVALUATION_GRID)
    eval_cfg["splits"] = ["train", "validation", "test"] if include_test else list(VALIDATION_SPLITS)
    eval_cfg["allow_test_metrics"] = bool(include_test)
    eval_cfg["save_predictions"] = bool(save_predictions)
    eval_cfg["save_test_survival_curves"] = bool(save_predictions and include_test)
    eval_cfg.pop("weighted_c_index_path", None)
    eval_cfg.pop("antolini_ctd_path", None)
    run_config["run"] = {
        "phase": phase,
        "model_name": model_name,
        "config_id": config_id,
        "hyperparameters": copy.deepcopy(hyperparameters),
        "seed": int(seed),
        "timestamp": timestamp_utc(),
    }
    return run_config, run_dir


def metric_value(metrics, split_name, metric_name):
    value = metrics.get("splits", {}).get(split_name, {}).get(metric_name)
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def tuning_summary_row(model_name, config_id, hyperparameters, seed, metrics, run_dir):
    validation = metrics.get("splits", {}).get("validation", {})
    return {
        "timestamp": timestamp_utc(),
        "model": model_name,
        "config_id": config_id,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "seed": int(seed),
        "validation_ctd_antolini": metric_value(metrics, "validation", "ctd_antolini"),
        "validation_ibll": metric_value(metrics, "validation", "ibll"),
        "validation_nbll": metric_value(metrics, "validation", "nbll"),
        "validation_ibs": metric_value(metrics, "validation", "ibs"),
        "validation_mean_horizon_c_index": metric_value(metrics, "validation", "mean_horizon_c_index"),
        "test_metrics_recorded": "test" in metrics.get("splits", {}),
        "metrics_path": str(Path(run_dir) / "metrics" / model_name / f"{model_name}_metrics.json"),
        "config_snapshot_path": str(Path(run_dir) / "config_snapshot.yaml"),
        "output_dir": str(run_dir),
        "selection_split": "validation",
        "selection_primary_metric": "validation_ctd_antolini",
        "selection_tiebreaker_metric": "validation_ibll",
        "validation_metrics": json.dumps(validation, sort_keys=True),
    }


def _score_for_selection(row):
    ctd = row.get("validation_ctd_antolini", math.nan)
    ibll = row.get("validation_ibll", math.nan)
    if math.isnan(ctd):
        ctd = -math.inf
    if math.isnan(ibll):
        ibll = math.inf
    return (-ctd, ibll, row["config_id"], row["seed"])


def select_best_row(rows):
    if not rows:
        raise ValueError("No tuning rows available for selection")
    return sorted(rows, key=_score_for_selection)[0]


def save_config_snapshot(run_config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config_snapshot.yaml"
    path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return path


def run_training(run_config, logger):
    resolved_config = _resolve_config_paths(copy.deepcopy(run_config))
    set_seed(resolved_config.get("seed", 42))
    ensure_output_dirs(resolved_config["paths"])
    model_name = resolved_config["model"]["name"]
    if model_name not in TRAINERS:
        raise ValueError(f"Unsupported static model: {model_name}")
    return TRAINERS[model_name](resolved_config, logger)


def _requested_models(tuning_config, requested):
    available = tuning_config.get("models", {})
    models = requested or list(available)
    unknown = [model for model in models if model not in available]
    if unknown:
        raise ValueError(f"Unknown tuning model(s): {unknown}")
    return models


def _write_model_summary(model_output_dir, rows):
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(model_output_dir / "tuning_results.csv", index=False)
    best = select_best_row(rows)
    save_json(best, model_output_dir / "best_hyperparameters.json")
    return best


def tune_models(config_path, requested_models=None, dry_run=False, max_runs=None):
    logger = get_logger("tune_static_models")
    tuning_config = load_yaml(config_path)
    seed = int(tuning_config.get("tuning", {}).get("seed", 42))
    output_root = tuning_config.get("tuning", {}).get("output_dir", "outputs/tuning")
    save_predictions = bool(tuning_config.get("tuning", {}).get("save_predictions", False))
    save_models = bool(tuning_config.get("tuning", {}).get("save_models", False))
    save_checkpoints = bool(tuning_config.get("tuning", {}).get("save_checkpoints", False))
    planned = []

    for model_name in _requested_models(tuning_config, requested_models):
        model_cfg = tuning_config["models"][model_name]
        base_config = load_yaml(model_cfg["base_config"])
        rows = []
        grid = expand_grid(model_cfg.get("grid", {}))
        for index, hyperparameters in enumerate(grid, start=1):
            if max_runs is not None and len(planned) >= int(max_runs):
                break
            config_id = f"{model_name}_cfg_{index:03d}"
            run_config, run_dir = prepare_run_config(
                base_config,
                model_name,
                config_id,
                hyperparameters,
                seed,
                output_root,
                phase="tuning",
                include_test=False,
                save_predictions=save_predictions,
                save_models=save_models,
                save_checkpoints=save_checkpoints,
            )
            planned.append({"model": model_name, "config_id": config_id, "seed": seed, "output_dir": str(run_dir)})
            if dry_run:
                continue
            save_config_snapshot(run_config, run_dir)
            metrics = run_training(run_config, logger)
            rows.append(tuning_summary_row(model_name, config_id, hyperparameters, seed, metrics, run_dir))

        if rows:
            best = _write_model_summary(Path(output_root) / model_name, rows)
            logger.info(
                "Best %s tuning config by validation Ctd/IBLL: %s",
                model_name,
                best["config_id"],
            )
    return planned


def main():
    parser = argparse.ArgumentParser(description="Run validation-only hyperparameter tuning for static survival models.")
    parser.add_argument("--config", default="configs/static_tuning.yaml")
    parser.add_argument("--models", nargs="*", help="Subset of models to tune.")
    parser.add_argument("--dry-run", action="store_true", help="Print/validate planned runs without training.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for smoke testing.")
    args = parser.parse_args()
    planned = tune_models(args.config, requested_models=args.models, dry_run=args.dry_run, max_runs=args.max_runs)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
