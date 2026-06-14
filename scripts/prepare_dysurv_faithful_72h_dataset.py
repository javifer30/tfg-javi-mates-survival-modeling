import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dysurv_faithful_72h_dataset import prepare_dataset
from src.models.dynamic_72h.common import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Prepare the isolated DySurv-faithful 72h dataset.")
    parser.add_argument("--config", default="configs/dysurv_faithful_72h.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare_dataset(load_yaml(args.config), force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
