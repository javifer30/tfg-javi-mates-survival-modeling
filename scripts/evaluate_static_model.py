import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml, resolve_path
from src.utils.logger import get_logger


def _read_metrics(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main(config_path):
    logger = get_logger("evaluate_static_model")
    config = load_yaml(config_path)
    rows = []
    for model in config["models"]:
        metrics_path = resolve_path(model["metrics_path"])
        if not metrics_path.exists():
            logger.warning("Metrics file not found for %s: %s", model["name"], metrics_path)
            continue
        metrics = _read_metrics(metrics_path)
        split_metrics = metrics.get("splits", {})
        if split_metrics:
            for split_name, values in split_metrics.items():
                row = {"model": model["name"], "split": split_name}
                row.update(values)
                rows.append(row)
        else:
            row = {"model": model["name"], "split": "descriptive"}
            row.update({k: v for k, v in metrics.items() if k != "model"})
            rows.append(row)

    comparison = pd.DataFrame(rows)
    output_path = resolve_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    logger.info("Static model comparison saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
