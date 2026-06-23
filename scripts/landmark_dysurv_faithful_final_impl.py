import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_dysurv_faithful_tuning_impl import build_run_config
from src.models.landmark_dynamic.common import load_yaml, save_json, timestamp_utc
from src.models.landmark_dynamic.train_dysurv_faithful import train_dysurv_faithful
from src.utils.logger import get_logger


FINAL_SEEDS = [42, 123, 2026]


def _metric(metrics, split, name):
    return float(metrics["splits"][split].get(name, math.nan))


def run_final(config_path: str, dry_run=False, sample_size=None, device="auto", allow_collapsed=False):
    base = load_yaml(config_path)
    seeds = [int(seed) for seed in base["final"]["seeds"]]
    if seeds != FINAL_SEEDS:
        raise ValueError(f"Final faithful seeds must be exactly {FINAL_SEEDS}")
    output_root = Path(base["paths"]["outputs_dir"])
    selection_path = output_root / "best_hyperparameters.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selected" and not allow_collapsed:
        raise ValueError("No accepted non-collapsed tuning selection. Review best_hyperparameters.json or pass --allow-collapsed explicitly.")
    selected = selection["selected"]
    params = json.loads(selected["hyperparameters"])
    config_id = selected["config_id"]
    logger = get_logger("run_final_dysurv_faithful_72h_seeds")
    planned, rows = [], []
    for seed in seeds:
        run_dir = output_root / "final" / f"seed_{seed}"
        planned.append({"seed": seed, "selected_config_id": config_id, "output_dir": str(run_dir)})
        if dry_run:
            continue
        config = build_run_config(base, config_id, params, run_dir, seed, sample_size, device, include_test=True)
        metrics = train_dysurv_faithful(config, logger)
        rows.append({
            "timestamp": timestamp_utc(),
            "model": "dysurv_faithful_72h",
            "selected_config_id": config_id,
            "seed": seed,
            "collapse_suspected": metrics["collapse"]["collapse_suspected"],
            "validation_ctd_antolini": _metric(metrics, "validation", "ctd_antolini"),
            "validation_ibs": _metric(metrics, "validation", "ibs"),
            "validation_ibll": _metric(metrics, "validation", "ibll"),
            "validation_mean_horizon_c_index": _metric(metrics, "validation", "mean_horizon_c_index"),
            "test_ctd_antolini": _metric(metrics, "test", "ctd_antolini"),
            "test_ibs": _metric(metrics, "test", "ibs"),
            "test_ibll": _metric(metrics, "test", "ibll"),
            "test_nbll": _metric(metrics, "test", "nbll"),
            "test_mean_horizon_c_index": _metric(metrics, "test", "mean_horizon_c_index"),
            "test_std_risk10": metrics["splits"]["test"]["diagnostics"]["std_risk10"],
            "test_range_risk10": metrics["splits"]["test"]["diagnostics"]["range_risk10"],
            "output_dir": str(run_dir),
        })

    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(output_root / "final_seed_results.csv", index=False)
        summary = {
            "status": "completed",
            "model": "dysurv_faithful_72h",
            "selected_config_id": config_id,
            "selected_hyperparameters": params,
            "seeds": seeds,
            "collapsed_seeds": frame.loc[frame["collapse_suspected"], "seed"].astype(int).tolist(),
            "metrics": {},
        }
        for name in ["test_ctd_antolini", "test_ibs", "test_ibll", "test_nbll", "test_mean_horizon_c_index", "test_std_risk10", "test_range_risk10"]:
            summary["metrics"][f"{name}_mean"] = float(frame[name].mean())
            summary["metrics"][f"{name}_std"] = float(frame[name].std(ddof=0))
        save_json(output_root / "final_seed_summary.json", summary)
    return planned


def main():
    parser = argparse.ArgumentParser(description="Final three-seed DySurv faithful 72h evaluation.")
    parser.add_argument("--config", default="configs/landmark_dysurv_faithful.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--allow-collapsed", action="store_true")
    args = parser.parse_args()
    planned = run_final(args.config, args.dry_run, args.sample_size, args.device, args.allow_collapsed)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
