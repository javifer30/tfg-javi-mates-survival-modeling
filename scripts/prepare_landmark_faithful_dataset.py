import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dysurv_faithful_72h_dataset import prepare_dataset
from src.models.dynamic_72h.common import load_yaml
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_faithful_config, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Prepare common faithful dataset for a CLI-selected landmark.")
    parser.add_argument("--config", default="configs/dysurv_faithful_72h.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = apply_landmark_faithful_config(load_yaml(args.config), args.landmark_hours, "dysurv_faithful")
    save_config_used(config, config["outputs"]["config_used_path"])
    result = prepare_dataset(config, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
