import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_static_final_impl import run_final_models
from src.utils.config import load_yaml
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_static_tuning_config, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Run final static seeds for a selected landmark.")
    parser.add_argument("--config", default="configs/landmark_static_tuning.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = apply_landmark_static_tuning_config(load_yaml(args.config), args.landmark_hours)
    config_path = Path(config["outputs"]["config_used_path"])
    save_config_used(config, config_path)
    planned = run_final_models(str(config_path), requested_models=args.models, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
