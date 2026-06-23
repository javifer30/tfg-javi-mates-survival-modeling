import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.landmark_dynamic_dataset import build_landmark_dynamic_dataset
from src.utils.config import load_yaml, resolve_path
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_dynamic_data_config, save_config_used
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


PATH_KEYS = {
    ("paths", "static_landmark_dir"),
    ("paths", "timeseries_path"),
    ("paths", "timeserieslab_path"),
    ("paths", "output_dir"),
    ("paths", "audit_dir"),
    ("paths", "preprocessor_path"),
    ("static", "train_path"),
    ("static", "val_path"),
    ("static", "test_path"),
}


def _resolve_config_paths(config):
    for section, key in PATH_KEYS:
        if section in config and key in config[section] and isinstance(config[section][key], str):
            config[section][key] = str(resolve_path(config[section][key]))
    return config


def main():
    parser = argparse.ArgumentParser(description="Build dynamic data for a CLI-selected landmark.")
    parser.add_argument("--config", default="configs/landmark_dynamic_data.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int)
    args = parser.parse_args()

    logger = get_logger("build_landmark_dynamic_data")
    config = apply_landmark_dynamic_data_config(load_yaml(args.config), args.landmark_hours)
    save_config_used(config, config["outputs"]["config_used_path"])
    set_seed(config.get("seed", 42))
    _, summary = build_landmark_dynamic_dataset(
        _resolve_config_paths(config),
        logger,
        force=args.force,
        dry_run=args.dry_run,
        sample_size=args.sample_size,
    )
    if args.dry_run:
        print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
