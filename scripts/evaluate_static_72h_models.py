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


def consolidate_final_static_72h(config_path):
    logger = get_logger("evaluate_static_72h_models")
    config = load_yaml(config_path)
    final_dir = resolve_path(config["paths"]["final_dir"])
    output_path = resolve_path(config["paths"]["comparison_path"])
    rows = []
    for summary_path in Path(final_dir).glob("*/final_seed_summary.json"):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        rows.append(
            {
                "model": data["model"],
                "test_ctd_antolini_mean": metrics.get("test_ctd_antolini_mean"),
                "test_ctd_antolini_std": metrics.get("test_ctd_antolini_std"),
                "test_ibs_mean": metrics.get("test_ibs_mean"),
                "test_ibs_std": metrics.get("test_ibs_std"),
                "test_ibll_mean": metrics.get("test_ibll_mean"),
                "test_ibll_std": metrics.get("test_ibll_std"),
                "test_mean_horizon_c_index_mean": metrics.get("test_mean_horizon_c_index_mean"),
                "test_mean_horizon_c_index_std": metrics.get("test_mean_horizon_c_index_std"),
                "selected_hyperparameters": json.dumps(data["selected_hyperparameters"], sort_keys=True),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No final_seed_summary.json files found under {final_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("model").to_csv(output_path, index=False)
    logger.info("static_72h comparison saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Consolidate static_72h final model summaries.")
    parser.add_argument("--config", default="configs/static_72h_evaluation.yaml")
    args = parser.parse_args()
    consolidate_final_static_72h(args.config)


if __name__ == "__main__":
    main()
