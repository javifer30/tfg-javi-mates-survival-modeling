"""Prepare the common faithful dataset for DySurv-style models.

The input is the dynamic landmark dataset. The output is a cleaned dataset shared
by temporal DySurv, Dynamic-DeepHit and static-only DySurv, so these models are
compared on exactly the same patients, splits and targets.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.landmark_faithful_dataset import prepare_dataset
from src.models.landmark_dynamic.common import load_yaml
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, apply_landmark_faithful_config, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Prepare common faithful dataset for a CLI-selected landmark.")
    parser.add_argument("--config", default="configs/landmark_dysurv_faithful.yaml")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = apply_landmark_faithful_config(load_yaml(args.config), args.landmark_hours, "dysurv_faithful")
    save_config_used(config, config["outputs"]["config_used_path"])
    result = prepare_dataset(config, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
