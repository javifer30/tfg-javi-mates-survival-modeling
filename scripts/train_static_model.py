import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.coxph_tfg import train_coxph
from src.models.deephit_tfg import train_deephit
from src.models.deepsurv_tfg import train_deepsurv
from src.models.kaplan_meier_tfg import train_kaplan_meier
from src.models.pchazard_tfg import train_pchazard
from src.models.static_common import ensure_output_dirs
from src.utils.config import load_yaml, resolve_path
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


TRAINERS = {
    "kaplan_meier": train_kaplan_meier,
    "coxph": train_coxph,
    "deepsurv": train_deepsurv,
    "pchazard": train_pchazard,
    "deephit": train_deephit,
}


def _resolve_config_paths(config):
    for key, value in config.get("paths", {}).items():
        if isinstance(value, str):
            config["paths"][key] = str(resolve_path(value))
    return config


def main(config_path):
    logger = get_logger("train_static_model")
    config = _resolve_config_paths(load_yaml(config_path))
    set_seed(config.get("seed", 42))
    ensure_output_dirs(config["paths"])
    model_name = config["model"]["name"]
    if model_name not in TRAINERS:
        raise ValueError(f"Unsupported static model: {model_name}")
    logger.info("Training static model: %s", model_name)
    TRAINERS[model_name](config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
