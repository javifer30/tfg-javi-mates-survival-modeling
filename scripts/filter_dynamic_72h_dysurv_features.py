"""Create a DySurv-compatible feature subset from the built dynamic_72h arrays.

This script does not rebuild temporal data from MIMIC-derived CSVs. It only
selects columns from X_seq/M_seq in the already-built dynamic_72h NPZ files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DYSURV_FEATURE_MAP = {
    "Eye Response": ["chart::GCS - Eye Opening"],
    "Braden Score": [
        "chart::Braden Activity",
        "chart::Braden Friction/Shear",
        "chart::Braden Mobility",
        "chart::Braden Moisture",
        "chart::Braden Nutrition",
        "chart::Braden Sensory Perception",
    ],
    "Strength L Arm": ["chart::Strength L Arm"],
    "Strength R Arm": ["chart::Strength R Arm"],
    "Strength L Leg": ["chart::Strength L Leg"],
    "Strength R Leg": ["chart::Strength R Leg"],
    "ALT": ["lab::Alanine Aminotransferase (ALT)", "chart::ALT"],
    "Anion Gap": ["chart::Anion gap", "lab::Anion Gap"],
    "Base Excess": ["chart::Arterial Base Excess"],
    "Bilirubin": ["lab::Bilirubin, Total", "chart::Bilirubin"],
    "Total CO2": ["chart::TCO2 (calc) Arterial"],
    "Creatinine": ["chart::Creatinine (serum)", "lab::Creatinine"],
    "Hematocrit": ["chart::Hematocrit (serum)", "lab::Hematocrit"],
    "INR(PT)": ["lab::INR(PT)"],
    "MCH": ["lab::MCH"],
    "MCV": ["lab::MCV"],
    "PT": ["lab::PT"],
    "Phosphate": ["lab::Phosphate"],
    "Potassium": ["chart::Potassium (serum)", "chart::Potassium (whole blood)", "lab::Potassium"],
    "Red Blood Cells": ["lab::Red Blood Cells"],
    "Urea Nitrogen": ["lab::Urea Nitrogen", "chart::BUN"],
    "pCO2": ["chart::Arterial CO2 Pressure"],
    "pO2": ["chart::Arterial O2 pressure"],
    "Dyspnea Assessment": ["chart::Current Dyspnea Assessment"],
    "Glucose": [
        "chart::Glucose (serum)",
        "chart::Glucose (whole blood)",
        "chart::Glucose finger stick (range 70-100)",
        "lab::Glucose",
    ],
    "DBP": ["chart::Arterial Blood Pressure diastolic", "chart::Non Invasive Blood Pressure diastolic"],
    "O2 Flow": ["chart::O2 Flow"],
    "Pain Level": ["chart::Pain Level"],
    "Phosphorous": ["chart::Phosphorous", "lab::Phosphate"],
    "Richmond-RAS Scale": ["chart::Richmond-RAS Scale"],
    "GCS - Eye": ["chart::GCS - Eye Opening"],
    "GCS - Motor": ["chart::GCS - Motor Response"],
    "GCS - Verbal": ["chart::GCS - Verbal Response"],
    "Daily Weight": ["chart::Daily Weight"],
    "AST": ["lab::Aspartate Aminotransferase (AST)", "chart::AST"],
    "HCO3": ["chart::HCO3 (serum)", "lab::Bicarbonate"],
    "Hct": ["chart::Hematocrit (serum)", "lab::Hematocrit"],
    "Alkaline Phosphatase": ["lab::Alkaline Phosphatase", "chart::Alkaline Phosphatase"],
    "Bicarbonate": ["lab::Bicarbonate", "chart::HCO3 (serum)"],
    "Calcium": ["chart::Calcium non-ionized", "chart::Ionized Calcium", "lab::Calcium, Total"],
    "Chloride": ["chart::Chloride (serum)", "lab::Chloride"],
    "Hemoglobin": ["chart::Hemoglobin", "lab::Hemoglobin"],
    "Lactate": ["chart::Lactic Acid"],
    "MCHC": ["lab::MCHC"],
    "Magnesium": ["chart::Magnesium", "lab::Magnesium"],
    "PTT": ["lab::PTT"],
    "Platelet Count": ["chart::Platelet Count", "lab::Platelet Count"],
    "RDW": ["lab::RDW"],
    "Sodium": ["chart::Sodium (serum)", "lab::Sodium"],
    "White Blood Cells": ["lab::White Blood Cells", "chart::WBC"],
    "pH": ["chart::PH (Arterial)"],
    "JH-HLM": ["chart::Activity / Mobility (JH-HLM)"],
    "Heart Rate": ["chart::Heart Rate"],
    "SBP": ["chart::Arterial Blood Pressure systolic", "chart::Non Invasive Blood Pressure systolic"],
    "O2 Sat (%)": ["chart::O2 saturation pulseoxymetry"],
    "Pain Level Response": ["chart::Pain Level Response"],
    "Respiratory Rate": ["chart::Respiratory Rate"],
    "Temperature (F)": ["chart::Temperature Fahrenheit"],
}


def load_features(input_dir: Path) -> list[str]:
    with (input_dir / "temporal_feature_columns.json").open("r", encoding="utf-8") as f:
        return json.load(f)["temporal_features"]


def selected_feature_indices(features: list[str]) -> tuple[list[int], dict[str, list[str]], list[str]]:
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    selected: list[str] = []
    matched: dict[str, list[str]] = {}
    missing: list[str] = []
    for dysurv_name, candidates in DYSURV_FEATURE_MAP.items():
        found = [candidate for candidate in candidates if candidate in feature_to_index]
        if found:
            matched[dysurv_name] = found
            selected.extend(found)
        else:
            missing.append(dysurv_name)
    selected_unique = list(dict.fromkeys(selected))
    return [feature_to_index[feature] for feature in selected_unique], matched, missing


def filter_split(input_dir: Path, output_dir: Path, split: str, indices: list[int], input_suffix: str, output_suffix: str) -> dict[str, object]:
    src = input_dir / f"{split}_{input_suffix}.npz"
    dst = output_dir / f"{split}_{output_suffix}.npz"
    with np.load(src) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["X_seq"] = arrays["X_seq"][:, :, indices].astype("float32", copy=False)
    arrays["M_seq"] = arrays["M_seq"][:, :, indices].astype("float32", copy=False)
    np.savez_compressed(dst, **arrays)
    return {
        "split": split,
        "n_patients": int(arrays["patient_ids"].shape[0]),
        "X_seq_shape": list(arrays["X_seq"].shape),
        "M_seq_shape": list(arrays["M_seq"].shape),
        "X_static_shape": list(arrays["X_static"].shape),
        "event_rate": float(arrays["event_eval"].mean()),
        "observed_temporal_fraction": float(arrays["M_seq"].mean()),
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/dynamic_72h")
    parser.add_argument("--output-dir", default="data/processed/dynamic_72h_dysurv_features")
    parser.add_argument("--input-suffix", default="dynamic_72h")
    parser.add_argument("--output-suffix", default="dynamic_72h")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"{output_dir} already exists; use --force to overwrite files")
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(input_dir)
    indices, matched, missing = selected_feature_indices(features)
    selected_features = [features[idx] for idx in indices]
    if not selected_features:
        raise ValueError("No DySurv-compatible temporal features found")

    split_summaries = [
        filter_split(input_dir, output_dir, split, indices, args.input_suffix, args.output_suffix)
        for split in ("train", "val", "test")
    ]

    copy_if_exists(input_dir / "static_feature_columns.json", output_dir / "static_feature_columns.json")
    copy_if_exists(input_dir / "preprocessing_metadata.json", output_dir / "source_preprocessing_metadata.json")

    with (output_dir / "temporal_feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump({"temporal_features": selected_features, "source_feature_count": len(features)}, f, indent=2)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "method": "column_subset_from_existing_dynamic_landmark_npz",
        "input_suffix": args.input_suffix,
        "output_suffix": args.output_suffix,
        "n_source_temporal_features": len(features),
        "n_selected_temporal_features": len(selected_features),
        "selected_temporal_features": selected_features,
        "dysurv_matched_variables": matched,
        "dysurv_missing_variables": missing,
        "static_features_unchanged": True,
        "splits": split_summaries,
    }
    with (output_dir / "dynamic_72h_dysurv_feature_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
