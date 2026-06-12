import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dynamic_72h_dataset import build_dynamic_72h_dataset
from src.utils.config import load_yaml, resolve_path
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


PATH_KEYS = {
    ("paths", "static_72h_dir"),
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
    parser = argparse.ArgumentParser(description="Build the dynamic_72h dataset from static_72h splits and temporal MIMIC extraction files.")
    parser.add_argument("--config", default="configs/dynamic_72h_data.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dynamic_72h outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Build in memory and print shapes, but do not save heavy outputs.")
    parser.add_argument("--sample-size", type=int, default=None, help="Use the first N patients per split for a debug build.")
    args = parser.parse_args()

    logger = get_logger("build_dynamic_72h_data")
    config = _resolve_config_paths(load_yaml(args.config))
    set_seed(config.get("seed", 42))
    _, summary = build_dynamic_72h_dataset(
        config,
        logger,
        force=args.force,
        dry_run=args.dry_run,
        sample_size=args.sample_size,
    )
    if args.dry_run:
        logger.info("Dry-run summary: %s", summary["splits"])
        logger.info("Selected temporal features: %s", summary["temporal_features"])


if __name__ == "__main__":
    main()
