import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landmark_dysurv_feature_filter_impl import (
    copy_if_exists,
    filter_split,
    load_features,
    selected_feature_indices,
)
from src.utils.landmark import ALLOWED_LANDMARK_HOURS, dynamic_suffix, landmark_tag, save_config_used


def main():
    parser = argparse.ArgumentParser(description="Create DySurv-compatible dynamic feature subset for a landmark.")
    parser.add_argument("--landmark-hours", type=int, choices=ALLOWED_LANDMARK_HOURS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tag = landmark_tag(args.landmark_hours)
    suffix = dynamic_suffix(args.landmark_hours)
    input_dir = Path("data") / "processed" / tag / "dynamic"
    output_dir = Path("data") / "processed" / tag / "dynamic_dysurv_features"
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"{output_dir} already exists; use --force to overwrite files")
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(input_dir)
    indices, matched, missing = selected_feature_indices(features)
    selected_features = [features[idx] for idx in indices]
    if not selected_features:
        raise ValueError("No DySurv-compatible temporal features found")
    split_summaries = [
        filter_split(input_dir, output_dir, split, indices, suffix, suffix)
        for split in ("train", "val", "test")
    ]
    copy_if_exists(input_dir / "static_feature_columns.json", output_dir / "static_feature_columns.json")
    copy_if_exists(input_dir / "preprocessing_metadata.json", output_dir / "source_preprocessing_metadata.json")
    (output_dir / "temporal_feature_columns.json").write_text(
        json.dumps({"temporal_features": selected_features, "source_feature_count": len(features)}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "landmark_hours": int(args.landmark_hours),
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_suffix": suffix,
        "output_suffix": suffix,
        "method": "column_subset_from_existing_landmark_dynamic_npz",
        "n_source_temporal_features": len(features),
        "n_selected_temporal_features": len(selected_features),
        "selected_temporal_features": selected_features,
        "dysurv_matched_variables": matched,
        "dysurv_missing_variables": missing,
        "static_features_unchanged": True,
        "splits": split_summaries,
    }
    (output_dir / f"{suffix}_dysurv_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_config_used(summary, Path("outputs") / tag / "dynamic" / "dysurv_feature_filter_config_used.yaml")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
