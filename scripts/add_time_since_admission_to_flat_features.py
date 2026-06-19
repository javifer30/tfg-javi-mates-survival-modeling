import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_time_since_admission(raw_mimic_dir: Path) -> pd.DataFrame:
    icustays_path = raw_mimic_dir / "icu" / "icustays.csv.gz"
    admissions_path = raw_mimic_dir / "hosp" / "admissions.csv.gz"

    icustays = pd.read_csv(
        icustays_path,
        compression="gzip",
        usecols=["stay_id", "hadm_id", "intime"],
    )
    admissions = pd.read_csv(
        admissions_path,
        compression="gzip",
        usecols=["hadm_id", "admittime"],
    )

    merged = icustays.merge(admissions, on="hadm_id", how="left", validate="many_to_one")
    merged["intime"] = pd.to_datetime(merged["intime"], errors="coerce")
    merged["admittime"] = pd.to_datetime(merged["admittime"], errors="coerce")
    merged["time_since_admission_hours"] = (
        (merged["intime"] - merged["admittime"]).dt.total_seconds() / 3600.0
    )

    out = merged[["stay_id", "time_since_admission_hours"]].rename(
        columns={"stay_id": "patientunitstayid"}
    )
    if out["patientunitstayid"].duplicated().any():
        raise ValueError("Duplicate stay_id values found in icustays.csv.gz")
    return out


def add_time_since_admission(flat_features_path: Path, raw_mimic_dir: Path, output_path: Path) -> None:
    flat = pd.read_csv(flat_features_path)
    if "patientunitstayid" not in flat.columns:
        raise ValueError("flat_features file must contain patientunitstayid")
    if "time_since_admission_hours" in flat.columns:
        raise ValueError("flat_features file already contains time_since_admission_hours")

    original_columns = list(flat.columns)
    original_ids = flat["patientunitstayid"].copy()
    time_since_admission = load_time_since_admission(raw_mimic_dir)

    enriched = flat.merge(time_since_admission, on="patientunitstayid", how="left", validate="one_to_one")
    if not original_ids.equals(enriched["patientunitstayid"]):
        raise ValueError("Patient order changed while adding time_since_admission_hours")
    if enriched["time_since_admission_hours"].isna().any():
        missing = int(enriched["time_since_admission_hours"].isna().sum())
        raise ValueError(f"Missing time_since_admission_hours for {missing} flat_features rows")

    expected_columns = original_columns + ["time_since_admission_hours"]
    enriched = enriched[expected_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)

    print(f"Read flat features: {flat_features_path}")
    print(f"Read raw MIMIC-IV tables from: {raw_mimic_dir}")
    print(f"Wrote enriched flat features: {output_path}")
    print(f"Rows: {len(enriched)}")
    negative_count = int((enriched["time_since_admission_hours"] < 0).sum())
    if negative_count:
        print(
            "Warning: preserved "
            f"{negative_count} rows with negative time_since_admission_hours "
            "(ICU intime earlier than hospital admittime in raw timestamps)."
        )
    print(
        "time_since_admission_hours range: "
        f"{enriched['time_since_admission_hours'].min():.3f} to "
        f"{enriched['time_since_admission_hours'].max():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create flat_features_with_time_since_admission.csv from existing flat_features."
    )
    parser.add_argument(
        "--flat-features",
        default="data/processed/mimic_extraction/flat_features.csv",
        help="Existing flat_features.csv path.",
    )
    parser.add_argument(
        "--raw-mimic-dir",
        default="data/raw/mimic-iv-3.1",
        help="Root directory containing hosp/ and icu/ MIMIC-IV folders.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/mimic_extraction/flat_features_with_time_since_admission.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    add_time_since_admission(
        flat_features_path=resolve_path(args.flat_features),
        raw_mimic_dir=resolve_path(args.raw_mimic_dir),
        output_path=resolve_path(args.output),
    )


if __name__ == "__main__":
    main()
