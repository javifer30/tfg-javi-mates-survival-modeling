import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_static_72h_models import consolidate_final_static_72h
from src.utils.config import load_yaml
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, landmark_tag, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Consolidate final static model summaries for a selected landmark.")
    parser.add_argument("--config", default="configs/static_72h_evaluation.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    args = parser.parse_args()

    tag = landmark_tag(args.landmark_hours)
    cfg = load_yaml(args.config)
    cfg.setdefault("paths", {})
    cfg["paths"]["final_dir"] = f"outputs/{tag}/static/final"
    cfg["paths"]["comparison_path"] = f"outputs/{tag}/static/final/static_{args.landmark_hours}h_model_comparison.csv"
    cfg.setdefault("outputs", {})["config_used_path"] = f"outputs/{tag}/static/evaluation_config_used.yaml"
    config_path = Path(cfg["outputs"]["config_used_path"])
    save_config_used(cfg, config_path)
    consolidate_final_static_72h(str(config_path))


if __name__ == "__main__":
    main()
