"""Tune the static-only DySurv faithful baseline.

This baseline uses the static part of the faithful dataset and ignores temporal
sequences. It helps separate the effect of DySurv's architecture from the effect
of adding longitudinal information.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_dysurv_static_faithful_tuning_impl import tune
from src.models.landmark_dynamic.common import load_yaml
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_faithful_config, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Tune static-only DySurv faithful for a selected landmark.")
    parser.add_argument("--config", default="configs/landmark_dysurv_static_faithful.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    config = apply_landmark_faithful_config(load_yaml(args.config), args.landmark_hours, "dysurv_static_faithful")
    config_path = Path(config["outputs"]["config_used_path"])
    save_config_used(config, config_path)
    planned = tune(str(config_path), dry_run=args.dry_run, max_runs=args.max_runs, device=args.device, resume=args.resume)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
