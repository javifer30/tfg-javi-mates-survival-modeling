"""Run validation-only hyperparameter search for static landmark models.

The test split is not used during tuning. Use --dry-run to print the planned
runs, --models to select a subset, and --resume to skip completed candidates.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_static_tuning_impl import tune_models
from src.utils.config import load_yaml
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_static_tuning_config, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Validation-only tuning for static models at a selected landmark.")
    parser.add_argument("--config", default="configs/landmark_static_tuning.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = apply_landmark_static_tuning_config(load_yaml(args.config), args.landmark_hours)
    config_path = Path(config["outputs"]["config_used_path"])
    save_config_used(config, config_path)
    planned = tune_models(
        str(config_path),
        requested_models=args.models,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
        resume=args.resume,
    )
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
