import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.static_dataset import build_static_dataset
from src.utils.config import load_yaml, resolve_path
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def _resolve_config_paths(config):
    for section in ["paths"]:
        for key, value in config.get(section, {}).items():
            if isinstance(value, str):
                config[section][key] = str(resolve_path(value))
    return config


def main(config_path):
    logger = get_logger("build_static_data")
    config = _resolve_config_paths(load_yaml(config_path))
    set_seed(config.get("seed", 42))
    build_static_dataset(config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
