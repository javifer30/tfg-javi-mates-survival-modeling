# Reproducibility

## Purpose

This file documents the current GitHub-facing pipeline: the parametrizable
landmark survival modeling workflow for `landmark_hours` in `{24, 48, 72}`.

Historical pre-landmark commands are preserved in Git history and in the backup
branch `backup/pre-cleanup-landmark`, but they are no longer part of the active
repository interface.

## Environment

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

Use relative paths from the repository root. Do not commit MIMIC-IV data,
derived datasets, outputs, checkpoints or trained model artifacts.

## Required Local Inputs

The active landmark pipeline expects these local MIMIC-derived inputs:

```text
data/processed/mimic_extraction/flat_features_with_time_since_admission.csv
data/processed/mimic_extraction/labels.csv
data/processed/mimic_extraction/timeseries.csv
data/processed/mimic_extraction/timeserieslab.csv
```

If the enriched flat-feature file is missing, create it from the existing local
MIMIC extraction:

```bash
python scripts/add_time_since_admission_to_flat_features.py
```

This follows DEC-021 and keeps `time_since_admission_hours` in the static
covariate set.

## Landmark Data Build

Choose one landmark:

```text
24, 48, 72
```

Build the static landmark dataset:

```bash
python scripts/build_landmark_static_data.py --config configs/landmark_static_data.yaml --landmark-hours 72 --force
```

Build the common dynamic landmark dataset:

```bash
python scripts/build_landmark_dynamic_data.py --config configs/landmark_dynamic_data.yaml --landmark-hours 72 --force
```

Derive the DySurv-compatible temporal feature subset:

```bash
python scripts/filter_landmark_dysurv_features.py --landmark-hours 72 --force
```

Prepare the common faithful dataset used by temporal DySurv, Dynamic-DeepHit
faithful and static-only DySurv:

```bash
python scripts/prepare_landmark_faithful_dataset.py --config configs/landmark_dysurv_faithful.yaml --landmark-hours 72 --force
```

## Validation-Only Tuning

Static pycox/lifelines models:

```bash
python scripts/tune_landmark_static_models.py --config configs/landmark_static_tuning.yaml --landmark-hours 72 --dry-run
python scripts/tune_landmark_static_models.py --config configs/landmark_static_tuning.yaml --landmark-hours 72 --models coxph deepsurv logistic_hazard pchazard deephit_single --resume
```

DySurv faithful temporal:

```bash
python scripts/tune_landmark_dysurv_faithful.py --config configs/landmark_dysurv_faithful.yaml --landmark-hours 72 --dry-run --device cpu
python scripts/tune_landmark_dysurv_faithful.py --config configs/landmark_dysurv_faithful.yaml --landmark-hours 72 --device cuda --resume
```

Dynamic-DeepHit faithful:

```bash
python scripts/tune_landmark_dynamic_deephit_faithful.py --config configs/landmark_dynamic_deephit_faithful.yaml --landmark-hours 72 --dry-run --device cpu
python scripts/tune_landmark_dynamic_deephit_faithful.py --config configs/landmark_dynamic_deephit_faithful.yaml --landmark-hours 72 --device cuda --resume
```

DySurv static-only faithful:

```bash
python scripts/tune_landmark_dysurv_static_faithful.py --config configs/landmark_dysurv_static_faithful.yaml --landmark-hours 72 --dry-run --device cpu
python scripts/tune_landmark_dysurv_static_faithful.py --config configs/landmark_dysurv_static_faithful.yaml --landmark-hours 72 --device cuda --resume
```

Tuning is validation-only. Test metrics are reserved for final-seed scripts.

## Final Seed Runs

Run final seeds only after validation selection has completed.

Static models:

```bash
python scripts/run_final_landmark_static_seeds.py --config configs/landmark_static_tuning.yaml --landmark-hours 72
```

DySurv faithful temporal:

```bash
python scripts/run_final_landmark_dysurv_faithful_seeds.py --config configs/landmark_dysurv_faithful.yaml --landmark-hours 72 --device cuda
```

Dynamic-DeepHit faithful:

```bash
python scripts/run_final_landmark_dynamic_deephit_faithful_seeds.py --config configs/landmark_dynamic_deephit_faithful.yaml --landmark-hours 72 --device cuda
```

DySurv static-only faithful:

```bash
python scripts/run_final_landmark_dysurv_static_faithful_seeds.py --config configs/landmark_dysurv_static_faithful.yaml --landmark-hours 72 --device cuda
```

Final seeds are fixed:

```text
42, 123, 2026
```

## Outputs

Landmark-specific data and outputs are separated:

```text
data/processed/landmark_<s>h/
outputs/landmark_<s>h/
```

Resolved configs are stored at:

```text
outputs/landmark_<s>h/static/config_used.yaml
outputs/landmark_<s>h/dynamic/config_used.yaml
outputs/landmark_<s>h/dysurv_faithful/config_used.yaml
outputs/landmark_<s>h/dynamic_deephit_faithful/config_used.yaml
outputs/landmark_<s>h/dysurv_static_faithful/config_used.yaml
```

## Active Models

Static models:

- Kaplan-Meier descriptive baseline.
- CoxPH.
- DeepSurv-style CoxPH.
- LogisticHazard.
- PCHazard.
- DeepHitSingle.

Faithful dynamic/static-DySurv models:

- DySurv faithful temporal.
- Dynamic-DeepHit faithful.
- DySurv static-only faithful.

All models within a landmark share patient IDs, splits, targets and evaluation
horizon.

## Notes

- The active CLI source of truth is `--landmark-hours`.
- Base YAML configs use 72h defaults only as examples; the CLI overwrites the
  operational landmark paths and metadata.
- Historical code removed from the GitHub-facing tree can be recovered from the
  branch `backup/pre-cleanup-landmark`.
