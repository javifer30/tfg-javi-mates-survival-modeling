# Reproducibility

## Purpose

This file explains how to reproduce the current project pipeline from a local
checkout with local MIMIC-IV access. It should be updated only when commands,
dependencies, configs, data preparation steps or pipeline assumptions change.

Related documentation:

- [PROJECT_HISTORY.md](PROJECT_HISTORY.md) explains how the current pipeline was
  reached.
- [DECISIONS.md](DECISIONS.md) records methodological choices.
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) records executed runs.
- [TODO.md](TODO.md) tracks missing reproducibility work.

## Environment Setup

1. Create and activate the local environment.

```bash
python -m venv env
env\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Keep all paths relative to the repository root.

4. Do not commit MIMIC-IV data, derived datasets, outputs, checkpoints or local
   environment folders.

## Data Preparation

The static pipeline expects the direct MIMIC-IV extraction artifacts:

- `data/processed/mimic_extraction/flat_features.csv`
- `data/processed/mimic_extraction/labels.csv`

The canonical static data configuration is:

- `configs/static_data.yaml`

It defines:

- seed `42`;
- split proportions `60/20/20`;
- event stratification;
- input columns from the extracted flat features and labels;
- output directory `data/processed/static`;
- fitted preprocessor path `outputs/preprocessors/static_preprocessor.pkl`;
- dataset summary path `outputs/metrics/static_dataset_summary.json`.

Build the static dataset with:

```bash
python scripts/build_static_data.py --config configs/static_data.yaml
```

Expected static outputs:

- `data/processed/static/train_static.parquet`
- `data/processed/static/val_static.parquet`
- `data/processed/static/test_static.parquet`
- `data/processed/static/split_assignments.parquet`
- `outputs/preprocessors/static_preprocessor.pkl`
- `outputs/metrics/static_dataset_summary.json`

## Training

The current static model configs are:

- `configs/kaplan_meier.yaml`
- `configs/coxph.yaml`
- `configs/deepsurv.yaml`
- `configs/pchazard.yaml`
- `configs/deephit.yaml`

Train one model with:

```bash
python scripts/train_static_model.py --config configs/coxph.yaml
```

Run the configured static pipeline with:

```bash
python scripts/run_static_pipeline.py --config configs/static_pipeline.yaml
```

`configs/static_pipeline.yaml` currently builds the static dataset, trains
Kaplan-Meier, CoxPH, DeepSurv, PCHazard and DeepHit, then runs the configured
evaluation consolidation.

## Static Hyperparameter Tuning

The static tuning config is:

- `configs/static_tuning.yaml`

It defines small validation-only grids for CoxPH, DeepSurv, PCHazard and
DeepHit. The tuning objective is:

- primary: validation `ctd_antolini` maximized;
- tie-breaker: validation `ibll`/`nbll` minimized;
- additional logged metrics: validation `ibs` and
  `mean_horizon_c_index`.

The CoxPH tuning grid includes ridge penalties
`penalizer: [0.0, 0.001, 0.01, 0.1]` with `l1_ratio: [0.0]`. The unpenalized
and weakly penalized candidates are included for comparison, but convergence
warnings should be treated as evidence against a candidate if validation
metrics degrade.

The fixed evaluation protocol remains:

```yaml
evaluation_time_grid: [1, 2, 3, 4, 5, 6, 7, 8, 9]
horizon_times: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Plan tuning runs without training:

```bash
python scripts/tune_static_models.py --config configs/static_tuning.yaml --dry-run
```

Run validation-only tuning:

```bash
python scripts/tune_static_models.py --config configs/static_tuning.yaml
```

Tune a subset of models:

```bash
python scripts/tune_static_models.py --config configs/static_tuning.yaml --models coxph deephit
```

Tuning outputs are written under:

- `outputs/tuning/{model}/`

Each completed tuning run saves a config snapshot, validation metrics and a
model-level `best_hyperparameters.json`. Tuning configurations evaluate only
train and validation splits and do not load or score the test split.

## Final Static Seed Runs

After tuning has produced `best_hyperparameters.json` for each selected model,
run final static evaluation with exactly seeds `42`, `123` and `2026`:

```bash
python scripts/run_final_static_seeds.py --config configs/static_tuning.yaml
```

Run a subset of final models:

```bash
python scripts/run_final_static_seeds.py --config configs/static_tuning.yaml --models deephit pchazard
```

Final outputs are written under:

- `outputs/final_static/{model}/seed_{seed}/`

Final runs use the selected validation hyperparameters and may record test
metrics. Large model/checkpoint artifacts are disabled by default unless
explicitly enabled in `configs/static_tuning.yaml`.

## Static 72h pycox Pipeline

The new main-methodology static benchmark is isolated from the previous static
pipeline under the name `static_72h_pycox`.

Methodological definition:

- include only stays with observed time `Y_i > 72h`;
- prediction time is 72 hours after ICU admission;
- target duration is relative to that prediction time;
- horizon is 10 days after hour 72;
- stays without event inside the 10-day post-72h horizon are censored at 10
  days rather than removed.

Build the 72-hour static dataset:

```bash
python scripts/build_static_72h_data.py --config configs/static_72h_data.yaml
```

Expected dataset outputs:

- `data/processed/static_72h/train_static_72h.parquet`
- `data/processed/static_72h/val_static_72h.parquet`
- `data/processed/static_72h/test_static_72h.parquet`
- `data/processed/static_72h/split_assignments.parquet`
- `outputs/static_72h/preprocessors/static_72h_preprocessor.pkl`
- `outputs/static_72h/metrics/static_72h_dataset_summary.json`

Plan validation-only tuning without training:

```bash
python scripts/tune_static_72h_models.py --config configs/static_72h_tuning.yaml --dry-run
```

Run validation-only tuning:

```bash
python scripts/tune_static_72h_models.py --config configs/static_72h_tuning.yaml
```

Tune a subset or cap runs for smoke testing:

```bash
python scripts/tune_static_72h_models.py --config configs/static_72h_tuning.yaml --models coxph deephit_single --max-runs 3
```

Tuning outputs:

- `outputs/static_72h/tuning/{model}/tuning_results.csv`
- `outputs/static_72h/tuning/{model}/best_hyperparameters.json`

Run final 3-seed evaluation after tuning:

```bash
python scripts/run_final_static_72h_seeds.py --config configs/static_72h_tuning.yaml
```

Run a subset of final models:

```bash
python scripts/run_final_static_72h_seeds.py --config configs/static_72h_tuning.yaml --models logistic_hazard deephit_single
```

Final outputs:

- `outputs/static_72h/final/{model}/seed_42/`
- `outputs/static_72h/final/{model}/seed_123/`
- `outputs/static_72h/final/{model}/seed_2026/`
- `outputs/static_72h/final/{model}/final_seed_results.csv`
- `outputs/static_72h/final/{model}/final_seed_summary.json`
- `outputs/static_72h/final/static_72h_model_comparison.csv`

Regenerate the final comparison table without retraining:

```bash
python scripts/evaluate_static_72h_models.py --config configs/static_72h_evaluation.yaml
```

The 72-hour static models currently implemented are:

- Kaplan-Meier via lifelines;
- CoxPH via lifelines;
- DeepSurv-style neural CoxPH via pycox `CoxPH`;
- LogisticHazard via pycox;
- PCHazard via pycox;
- DeepHitSingle via pycox.

The main 72-hour metrics are Antolini Ctd, IBS and IBLL/NBLL from pycox
`EvalSurv`, plus the project extension C-index by horizon for days 1 through
10.

Audit outputs for the 72-hour pycox static pipeline are written to:

- `outputs/static_72h/audit/deephit_single_time_grid_audit.json`
- `outputs/static_72h/audit/deephit_single_survival_tail_check.csv`
- `outputs/static_72h/audit/pchazard_audit.json`
- `outputs/static_72h/audit/evaluation_grids.json`
- `outputs/static_72h/audit/discrete_time_cuts_summary.json`
- `outputs/static_72h/audit/survival_curve_sanity_checks.csv`

For `static_72h_pycox`, IBS and IBLL/NBLL use a per-split 100-point integration
grid bounded by the observed evaluation range, survival prediction support and
the 10-day horizon. The daily grid `[1, ..., 10]` is used only for horizon
C-index. PCHazard sets `sub=10` before `predict_surv_df`, matching the DySurv
static notebook convention.

## Dynamic 72h Dataset

The dynamic 72-hour dataset is built on top of the same cohort, splits and
targets as `static_72h_pycox`.

Required inputs:

- `data/processed/static_72h/train_static_72h.parquet`
- `data/processed/static_72h/val_static_72h.parquet`
- `data/processed/static_72h/test_static_72h.parquet`
- `data/processed/mimic_extraction/timeseries.csv`
- `data/processed/mimic_extraction/timeserieslab.csv`

Build the dataset:

```bash
python scripts/build_dynamic_72h_data.py --config configs/dynamic_72h_data.yaml --force
```

Smoke-test without writing outputs:

```bash
python scripts/build_dynamic_72h_data.py --config configs/dynamic_72h_data.yaml --dry-run --sample-size 2
```

Expected dataset outputs:

- `data/processed/dynamic_72h/train_dynamic_72h.npz`
- `data/processed/dynamic_72h/val_dynamic_72h.npz`
- `data/processed/dynamic_72h/test_dynamic_72h.npz`
- `data/processed/dynamic_72h/dynamic_72h_dataset_summary.json`
- `data/processed/dynamic_72h/temporal_feature_columns.json`
- `data/processed/dynamic_72h/static_feature_columns.json`
- `data/processed/dynamic_72h/preprocessing_metadata.json`
- `data/processed/dynamic_72h/preprocessor.joblib`

Expected audit outputs:

- `outputs/dynamic_72h/audit/dynamic_72h_data_audit.json`
- `outputs/dynamic_72h/audit/missingness_summary.csv`
- `outputs/dynamic_72h/audit/temporal_coverage_summary.csv`
- `outputs/dynamic_72h/audit/feature_coverage_by_split.csv`
- `outputs/dynamic_72h/audit/hourly_missingness_summary.csv`

The saved arrays contain:

- `patient_ids`
- `X_seq` with shape `[N, 72, F]`
- `M_seq` with shape `[N, 72, F]`
- `X_static` with shape `[N, P]`
- `duration_eval_days`
- `duration_rel_days`
- `event_eval`

Current build summary from `EXP-010`: `F=146` temporal features, `P=28` static
features, train shape `(18706, 72, 146)`, validation/test shape
`(6236, 72, 146)`. Temporal features are selected using train coverage only;
imputation and p05/p95 scaling are fitted on train only. Offsets are restricted
to `0 <= offset_minutes < 4320`, then binned into hours `0..71`.

### DySurv-compatible dynamic feature subset

For a faster first training pass closer to the DySurv reference variable table,
derive a reduced dataset from the already-built `dynamic_72h` arrays:

```bash
python scripts/filter_dynamic_72h_dysurv_features.py --force
```

This does not rescan MIMIC-derived temporal CSV files and does not refit
imputation or scaling. It only slices `X_seq` and `M_seq` columns and keeps
`patient_ids`, `X_static`, durations and events unchanged.

Expected outputs:

- `data/processed/dynamic_72h_dysurv_features/train_dynamic_72h.npz`
- `data/processed/dynamic_72h_dysurv_features/val_dynamic_72h.npz`
- `data/processed/dynamic_72h_dysurv_features/test_dynamic_72h.npz`
- `data/processed/dynamic_72h_dysurv_features/temporal_feature_columns.json`
- `data/processed/dynamic_72h_dysurv_features/dynamic_72h_dysurv_feature_summary.json`

Current subset summary after `EXP-012`: 61 temporal features, train shape
`(18706, 72, 61)`, validation/test shape `(6236, 72, 61)`. The folder
`data/processed/dynamic_72h_dysurv_features/` was overwritten after removing
15 additional chart-derived columns from the initial 76-feature subset. The
DySurv table variables still missing from the temporal set are `ALT`,
`Bilirubin`, `AST` and `Alkaline Phosphatase`.

## Dynamic 72h Models

The dynamic 72h model layer is isolated from the static pipelines.

Implemented models:

- `dysurv`
- `dynamic_deephit`

Main config:

- `configs/dynamic_72h_tuning.yaml`

Final wrapper config:

- `configs/dynamic_72h_final.yaml`

Default input mode:

```yaml
input_mode: values_plus_mask_plus_static
```

This creates model inputs by concatenating:

- `X_seq`
- `M_seq`
- `X_static` repeated across all 72 timesteps

For the current 61-feature subset, this gives model input shape
`[N, 72, 150]`.

Plan validation-only tuning:

```bash
python scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv dynamic_deephit --dry-run
```

Run a small smoke test:

```bash
python scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv dynamic_deephit --max-runs 2 --sample-size 128 --device cpu --force
```

Run validation-only tuning:

```bash
python scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv dynamic_deephit
```

Plan final three-seed evaluation after tuning:

```bash
python scripts/run_final_dynamic_72h_seeds.py --config configs/dynamic_72h_final.yaml --model dysurv dynamic_deephit --dry-run
```

Run final three-seed evaluation after tuning:

```bash
python scripts/run_final_dynamic_72h_seeds.py --config configs/dynamic_72h_final.yaml --model dysurv dynamic_deephit
```

Consolidate final dynamic results:

```bash
python scripts/evaluate_dynamic_72h_models.py --outputs-dir outputs/dynamic_72h
```

Dynamic tuning outputs:

- `outputs/dynamic_72h/tuning/{model}/tuning_results.csv`
- `outputs/dynamic_72h/tuning/{model}/best_hyperparameters.json`
- `outputs/dynamic_72h/tuning/{model}/{config_id}/seed_{seed}/metrics/metrics.json`
- `outputs/dynamic_72h/tuning/{model}/{config_id}/seed_{seed}/train_log.csv`

Dynamic audit outputs:

- `outputs/dynamic_72h/audit/{model}/{config_id}/seed_{seed}/{model}_input_audit.json`
- `outputs/dynamic_72h/audit/{model}/{config_id}/seed_{seed}/{model}_target_audit.json`
- `outputs/dynamic_72h/audit/{model}/{config_id}/seed_{seed}/{model}_prediction_sanity.csv`
- `outputs/dynamic_72h/audit/{model}/{config_id}/seed_{seed}/target_discretization_summary.csv`
- `outputs/dynamic_72h/audit/dynamic_deephit/{config_id}/seed_{seed}/dynamic_deephit_probability_audit.csv`

The shared target discretization uses daily cuts `[0, 1, ..., 10]` and maps
durations to interval indices `0..9` for intervals `(0,1]`, ..., `(9,10]`.
Tuning evaluates train and validation only; test metrics are produced only by
the final-seed script.

## Evaluation

The static evaluation config is:

- `configs/static_evaluation.yaml`

It consolidates model metrics into:

- `outputs/metrics/static_model_comparison.csv`

Run evaluation consolidation with:

```bash
python scripts/evaluate_static_model.py --config configs/static_evaluation.yaml
```

The current static metric protocol uses the same day grid across eligible
models:

```yaml
evaluation_time_grid: [1, 2, 3, 4, 5, 6, 7, 8, 9]
horizon_times: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

`evaluation_time_grid` is used for IBS and IBLL/NBLL. `horizon_times` is used
for horizon C-index. IBS and IBLL/NBLL are computed on validation and test by
default, not on full train, to avoid memory-heavy evaluation over all unique
observed times.

Additional time-dependent evaluation scripts exist for the curve-producing
static neural models:

```bash
python scripts/evaluate_pchazard_time_dependent.py --config configs/pchazard.yaml
python scripts/evaluate_deephit_time_dependent.py --config configs/deephit.yaml
```

Expected time-dependent outputs include:

- `outputs/metrics/pchazard/pchazard_weighted_c_index_by_horizon.csv`
- `outputs/metrics/pchazard/pchazard_antolini_ctd.csv`
- `outputs/metrics/deephit/deephit_weighted_c_index_by_horizon.csv`
- `outputs/metrics/deephit/deephit_antolini_ctd.csv`

The main static training pipeline now also writes time-dependent metric files
for CoxPH and DeepSurv:

- `outputs/metrics/coxph/coxph_weighted_c_index_by_horizon.csv`
- `outputs/metrics/coxph/coxph_antolini_ctd.csv`
- `outputs/metrics/deepsurv/deepsurv_weighted_c_index_by_horizon.csv`
- `outputs/metrics/deepsurv/deepsurv_antolini_ctd.csv`

Model-specific metric artifacts are stored under one folder per model inside
`outputs/metrics/`, for example:

- `outputs/metrics/coxph/coxph_metrics.json`
- `outputs/metrics/deepsurv/deepsurv_metrics.json`
- `outputs/metrics/pchazard/pchazard_metrics.json`
- `outputs/metrics/deephit/deephit_metrics.json`

Metrics JSON files distinguish:

- `harrell_c_index` for CoxPH and DeepSurv natural scalar risks;
- `harrell_c_index_final_risk` for DeepHit and PCHazard final cumulative risk;
- `ctd_antolini`, `horizon_c_index`, `mean_horizon_c_index`;
- `ibs`, `ibll` and `nbll`;
- `evaluation_time_grid` and `horizon_times`.

## Assumptions

- MIMIC-IV v3.1 data is available locally and is not versioned.
- The static dataset uses adult ICU stays with valid positive observed time.
- Train, validation and test splits are created before fitting preprocessing
  statistics.
- Preprocessing statistics are fitted on train only.
- Static results are the current consolidated baseline; full dynamic landmark
  training is still pending.
- `src/models_references/` contains local methodological references and should
  not be treated as project-owned model code.

## Reproducibility Notes

- Record every executed experiment in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
- Record methodological changes in [DECISIONS.md](DECISIONS.md).
- Update this file when command names, config names, dependencies or pipeline
  steps change.
- If `outputs/metrics/static_model_comparison.csv` is missing after a run, rerun
  evaluation consolidation and record the result in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
- The root `README.md` may lag behind the final static pipeline. Treat this file
  and `configs/static_pipeline.yaml` as the reproducibility authority until the
  root README is refreshed.
