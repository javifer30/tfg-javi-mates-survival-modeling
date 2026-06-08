import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_static_data import main as build_static_data
from scripts.evaluate_static_model import main as evaluate_static_models
from scripts.train_static_model import main as train_static_model
from src.utils.config import load_yaml
from src.utils.logger import get_logger


def main(config_path):
    logger = get_logger("run_static_pipeline")
    config = load_yaml(config_path)
    if config.get("run_build_static_data", True):
        logger.info("Running static dataset build")
        build_static_data(config["static_data_config"])
    for model_config in config["model_configs"]:
        logger.info("Running model config: %s", model_config)
        train_static_model(model_config)
    if config.get("run_evaluation", True):
        logger.info("Running final static evaluation")
        evaluate_static_models(config["evaluation_config"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
