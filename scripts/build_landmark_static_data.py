"""Build the static table for one landmark.

This is the first step of the pipeline. It reads the local MIMIC-derived flat
features and labels, keeps only patients still observed at the selected
landmark, and writes train/validation/test parquet files for the static models.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.landmark_static_dataset import build_landmark_static_dataset
from src.utils.config import load_yaml, resolve_path
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_static_data_config, save_config_used
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def _resolve_paths(config):
    for key, value in config.get("paths", {}).items():
        if isinstance(value, str):
            config["paths"][key] = str(resolve_path(value))
    return config


def main():
    parser = argparse.ArgumentParser(description="Build static data for a CLI-selected landmark.")
    parser.add_argument("--config", default="configs/landmark_static_data.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logger = get_logger("build_landmark_static_data")
    config = apply_landmark_static_data_config(load_yaml(args.config), args.landmark_hours)
    save_config_used(config, config["outputs"]["config_used_path"])
    suffix = config["output_file_suffix"]
    output_dir = Path(config["paths"]["processed_dir"])
    expected = [output_dir / f"{name}_{suffix}.parquet" for name in ["train", "val", "test"]]
    if any(path.exists() for path in expected) and not args.force:
        raise FileExistsError(f"Landmark static outputs already exist in {output_dir}; use --force to overwrite")
    set_seed(config.get("seed", 42))
    build_landmark_static_dataset(_resolve_paths(config), logger)


if __name__ == "__main__":
    main()
