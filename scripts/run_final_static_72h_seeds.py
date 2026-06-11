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

from scripts.tune_static_72h_models import _model_list, _resolve_config_paths, _run_config, metric, save_config_snapshot, timestamp_utc
from src.models.static_72h_pycox import train_static_72h_model
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


FINAL_SEEDS = [42, 123, 2026]


def _validate_seeds(seeds):
    if [int(seed) for seed in seeds] != FINAL_SEEDS:
        raise ValueError(f"Final static_72h seeds must be exactly {FINAL_SEEDS}")


def _load_best(tuning_root, model_name):
    path = Path(tuning_root) / model_name / "best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing best hyperparameters for {model_name}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["config_id"], json.loads(data["hyperparameters"])


def final_row(model_name, selected_config_id, hyperparameters, seed, output_dir, metrics):
    return {
        "timestamp": timestamp_utc(),
        "model": model_name,
        "selected_config_id": selected_config_id,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "seed": int(seed),
        "validation_ctd_antolini": metric(metrics, "validation", "ctd_antolini"),
        "validation_ibs": metric(metrics, "validation", "ibs"),
        "validation_ibll": metric(metrics, "validation", "ibll"),
        "validation_nbll": metric(metrics, "validation", "nbll"),
        "validation_mean_horizon_c_index": metric(metrics, "validation", "mean_horizon_c_index"),
        "test_ctd_antolini": metric(metrics, "test", "ctd_antolini"),
        "test_ibs": metric(metrics, "test", "ibs"),
        "test_ibll": metric(metrics, "test", "ibll"),
        "test_nbll": metric(metrics, "test", "nbll"),
        "test_mean_horizon_c_index": metric(metrics, "test", "mean_horizon_c_index"),
        "output_dir": str(output_dir),
        "metrics_path": str(Path(output_dir) / "metrics" / model_name / f"{model_name}_metrics.json"),
    }


def _mean_std(values):
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return math.nan, math.nan
    return float(pd.Series(clean).mean()), float(pd.Series(clean).std(ddof=0))


def _write_model_summary(model_dir, rows, selected_config_id, hyperparameters):
    df = pd.DataFrame(rows)
    df.to_csv(model_dir / "final_seed_results.csv", index=False)
    summary = {
        "model": rows[0]["model"],
        "selected_config_id": selected_config_id,
        "selected_hyperparameters": hyperparameters,
        "seeds": FINAL_SEEDS,
        "metrics": {},
    }
    for metric_name in ["test_ctd_antolini", "test_ibs", "test_ibll", "test_nbll", "test_mean_horizon_c_index"]:
        mean, std = _mean_std(df[metric_name].tolist())
        summary["metrics"][f"{metric_name}_mean"] = mean
        summary["metrics"][f"{metric_name}_std"] = std
    (model_dir / "final_seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_comparison(final_root):
    rows = []
    for path in Path(final_root).glob("*/final_seed_summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        rows.append(
            {
                "model": data["model"],
                "test_ctd_antolini_mean": metrics.get("test_ctd_antolini_mean"),
                "test_ctd_antolini_std": metrics.get("test_ctd_antolini_std"),
                "test_ibs_mean": metrics.get("test_ibs_mean"),
                "test_ibs_std": metrics.get("test_ibs_std"),
                "test_ibll_mean": metrics.get("test_ibll_mean"),
                "test_ibll_std": metrics.get("test_ibll_std"),
                "test_mean_horizon_c_index_mean": metrics.get("test_mean_horizon_c_index_mean"),
                "test_mean_horizon_c_index_std": metrics.get("test_mean_horizon_c_index_std"),
                "selected_hyperparameters": json.dumps(data["selected_hyperparameters"], sort_keys=True),
            }
        )
    if rows:
        pd.DataFrame(rows).sort_values("model").to_csv(Path(final_root) / "static_72h_model_comparison.csv", index=False)


def run_final_models(config_path, requested_models=None, dry_run=False):
    logger = get_logger("run_final_static_72h_seeds")
    tuning_config = load_yaml(config_path)
    seeds = tuning_config["final"].get("seeds", FINAL_SEEDS)
    _validate_seeds(seeds)
    output_root = Path(tuning_config["paths"].get("outputs_dir", "outputs/static_72h"))
    tuning_root = output_root / "tuning"
    final_root = output_root / "final"
    planned = []

    for model_name in _model_list(tuning_config, requested_models):
        selected_config_id, hyperparameters = _load_best(tuning_root, model_name)
        rows = []
        model_dir = final_root / model_name
        for seed in seeds:
            run_dir = model_dir / f"seed_{int(seed)}"
            run_config = _run_config(tuning_config, model_name, selected_config_id, hyperparameters, run_dir, include_test=True, seed=int(seed))
            planned.append({"model": model_name, "seed": int(seed), "output_dir": str(run_dir)})
            if dry_run:
                continue
            save_config_snapshot(run_config, run_dir)
            resolved = _resolve_config_paths(copy.deepcopy(run_config))
            set_seed(int(seed))
            metrics = train_static_72h_model(resolved, logger)
            rows.append(final_row(model_name, selected_config_id, hyperparameters, int(seed), run_dir, metrics))
        if rows:
            model_dir.mkdir(parents=True, exist_ok=True)
            _write_model_summary(model_dir, rows, selected_config_id, hyperparameters)
    if not dry_run:
        _write_comparison(final_root)
    return planned


def main():
    parser = argparse.ArgumentParser(description="Run final static_72h_pycox models with exactly three seeds.")
    parser.add_argument("--config", default="configs/static_72h_tuning.yaml")
    parser.add_argument("--models", nargs="*", help="Subset of models to run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    planned = run_final_models(args.config, requested_models=args.models, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
