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

from scripts.tune_dynamic_72h_models import _model_list, _run_config
from src.models.dynamic_72h.common import load_yaml, metric, save_json, timestamp_utc
from src.models.dynamic_72h.train import train_dynamic_72h_model
from src.utils.logger import get_logger


FINAL_SEEDS = [42, 123, 2026]


def _validate_seeds(seeds):
    if [int(seed) for seed in seeds] != FINAL_SEEDS:
        raise ValueError(f"Final dynamic_72h seeds must be exactly {FINAL_SEEDS}")


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
    }


def _mean_std(values):
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return math.nan, math.nan
    return float(pd.Series(clean).mean()), float(pd.Series(clean).std(ddof=0))


def _write_model_summary(model_dir, rows, selected_config_id, hyperparameters):
    df = pd.DataFrame(rows)
    df.to_csv(model_dir / "final_seed_results.csv", index=False)
    summary = {"model": rows[0]["model"], "selected_config_id": selected_config_id, "selected_hyperparameters": hyperparameters, "seeds": FINAL_SEEDS, "metrics": {}}
    for metric_name in ["test_ctd_antolini", "test_ibs", "test_ibll", "test_nbll", "test_mean_horizon_c_index"]:
        mean, std = _mean_std(df[metric_name].tolist())
        summary["metrics"][f"{metric_name}_mean"] = mean
        summary["metrics"][f"{metric_name}_std"] = std
    save_json(model_dir / "final_seed_summary.json", summary)


def run_final_models(config_path, requested_models=None, dry_run=False, sample_size=None, device="auto"):
    logger = get_logger("run_final_dynamic_72h_seeds")
    cfg = load_yaml(config_path)
    if "base_config" in cfg:
        base = load_yaml(cfg["base_config"])
        base["final"] = cfg.get("final", base.get("final", {}))
        cfg = base
    seeds = cfg["final"].get("seeds", FINAL_SEEDS)
    _validate_seeds(seeds)
    output_root = Path(cfg["paths"].get("outputs_dir", "outputs/dynamic_72h"))
    tuning_root = output_root / "tuning"
    final_root = output_root / "final"
    planned = []
    for model_name in _model_list(cfg, requested_models):
        selected_config_id, hyperparameters = _load_best(tuning_root, model_name)
        rows = []
        model_dir = final_root / model_name
        for seed in seeds:
            run_dir = model_dir / f"seed_{int(seed)}"
            run_config = _run_config(cfg, model_name, selected_config_id, hyperparameters, run_dir, True, int(seed), sample_size, device)
            planned.append({"model": model_name, "seed": int(seed), "output_dir": str(run_dir)})
            if dry_run:
                continue
            metrics = train_dynamic_72h_model(run_config, logger)
            rows.append(final_row(model_name, selected_config_id, hyperparameters, int(seed), run_dir, metrics))
        if rows:
            model_dir.mkdir(parents=True, exist_ok=True)
            _write_model_summary(model_dir, rows, selected_config_id, hyperparameters)
    return planned


def main():
    parser = argparse.ArgumentParser(description="Run final dynamic_72h models with exactly three seeds.")
    parser.add_argument("--config", default="configs/dynamic_72h_final.yaml")
    parser.add_argument("--model", "--models", dest="models", nargs="*", choices=["dysurv", "dynamic_deephit"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    planned = run_final_models(args.config, args.models, args.dry_run, args.sample_size, args.device)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()

