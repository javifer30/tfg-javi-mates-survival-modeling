# Experiment Log

## Purpose

This file records experiments that were actually executed or explicitly queued
for execution. It is the run-level companion to [PROJECT_HISTORY.md](PROJECT_HISTORY.md):
history summarizes phases, while this file preserves concrete commands, configs,
outputs and outcomes.

Use this file together with:

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for setup and canonical commands;
- [DECISIONS.md](DECISIONS.md) for methodological choices that affect a run;
- [TODO.md](TODO.md) for failed, blocked or follow-up experiment work.

## Usage Instructions

- Append one entry per experiment or evaluation run.
- Log only experiments that were actually run, are currently running, failed
  during execution, or are explicitly scheduled.
- Include the exact command, config path, seed and output paths.
- Do not record speculative results.
- If metrics are consolidated elsewhere, link the artifact instead of copying
  large tables repeatedly.
- If an experiment changes interpretation of the project, add or reference a
  decision in [DECISIONS.md](DECISIONS.md).

## Experiment Template

````md
## EXP-000 — Experiment title

Date: YYYY-MM-DD
Status: planned | running | completed | failed
Model:
Dataset:
Config:
Seed:
Run directory:
Related decision:

### Goal
- What question does this run answer?

### Command
```bash
python path/to/script.py --config path/to/config.yaml
```

### Inputs
- Dataset path:
- Preprocessor:
- Checkpoint or warm start:

### Outputs
- Metrics:
- Predictions:
- Figures:
- Checkpoints:

### Results
- Primary metric:
- Secondary metrics:
- Notes:

### Interpretation
- What does this result mean for the project?

### Follow-up
- [ ] Optional task to add to TODO.md
````

## Example

````md
## EXP-003 — CoxPH baseline on static dataset

Date: 2026-06-07
Status: completed
Model: CoxPH
Dataset: static_v1
Config: configs/coxph.yaml
Seed: 42
Run directory: outputs/
Related decision: DEC-001

### Goal
- Train the classical CoxPH baseline on the final static dataset.

### Command
```bash
python scripts/train_static_model.py --config configs/coxph.yaml
```

### Inputs
- Dataset path: data/processed/static/
- Preprocessor: outputs/preprocessors/static_preprocessor.pkl
- Checkpoint or warm start: none

### Outputs
- Metrics: outputs/metrics/coxph/coxph_metrics.json
- Predictions: outputs/predictions/
- Figures: outputs/figures/
- Checkpoints: not applicable

### Results
- Primary metric: test C-index 0.7404
- Secondary metrics: test IBS 0.1246; test NBLL 0.3973
- Notes: CoxPH is stable and competitive as a classical baseline.

### Interpretation
- CoxPH remains an important reference point for judging neural static and dynamic models.

### Follow-up
- [ ] Keep CoxPH results in the final static-vs-dynamic comparison table.
````

## EXP-004 — DeepHit after tail-support and ranking-loss fixes

Date: 2026-06-08 12:58:11
Status: completed
Model: DeepHit
Dataset: static MIMIC-IV adult ICU static dataset
Config: `configs/deephit.yaml`
Seed: 42
Run directory: `outputs/`
Related decision: DEC-005; DEC-006

### Goal
- Verify the effect of the approved DeepHit implementation fixes:
  internal tail category / beyond-horizon support, censored-at-horizon
  likelihood repair, survival reconstruction with nonzero final survival,
  event-time-conditioned pairwise ranking loss and likelihood broadcasting
  correction.

### Command
```bash
python scripts/train_static_model.py --config configs/deephit.yaml
```

- Exact shell history was not available in the metric artifacts; this is the
  canonical command that produces the recorded DeepHit outputs.

### Inputs
- Dataset path: `data/processed/static/`
- Train split: `data/processed/static/train_static.parquet`
- Validation split: `data/processed/static/val_static.parquet`
- Test split: `data/processed/static/test_static.parquet`
- Preprocessor: `outputs/preprocessors/static_preprocessor.pkl`
- Checkpoint or warm start: not identified from the metric artifacts

### Outputs
- Metrics JSON: `outputs/metrics/deephit/deephit_metrics.json`
- Training log: `outputs/metrics/deephit/deephit_train_log.csv`
- Antolini Ctd: `outputs/metrics/deephit/deephit_antolini_ctd.csv`
- Horizon C-index: `outputs/metrics/deephit/deephit_weighted_c_index_by_horizon.csv`
- Predictions: `outputs/predictions/deephit_predictions.parquet`
- Test survival curves: `outputs/predictions/deephit_test_survival_curves.csv`
- Checkpoints: `outputs/checkpoints/deephit_best_model.pt`,
  `outputs/checkpoints/deephit_last_model.pt`

### Results
- Best validation loss: 0.447695.
- Validation final-risk C-index: 0.7591.
- Validation Ctd Antolini: 0.7735.
- Validation mean horizon C-index: 0.7469.
- Validation IBS: 0.1108.
- Validation IBLL/NBLL: 0.3545.
- Test final-risk C-index: 0.7529.
- Test Ctd Antolini: 0.7695.
- Test mean horizon C-index: 0.7505.
- Test IBS: 0.1107.
- Test IBLL/NBLL: 0.3531.

Test C-index by horizon:

| Horizon days | C-index@h |
| ---: | ---: |
| 1 | 0.7925 |
| 2 | 0.7765 |
| 3 | 0.7603 |
| 4 | 0.7499 |
| 5 | 0.7475 |
| 6 | 0.7378 |
| 7 | 0.7315 |
| 8 | 0.7306 |
| 9 | 0.7278 |

PMF/tail diagnostics from `outputs/predictions/deephit_predictions.parquet`:

- Mean probability mass in evaluated bins 1-10: 0.2636.
- Mean tail probability beyond 10 days: 0.7364.
- Tail probability range: 0.2540 to 0.9996.
- Mean `S(10)`: 0.7364.
- `S(10)` range: 0.2540 to 0.9996.
- Share with `S(10) > 0`: 1.0000.
- `S(10)` equals `tail_probability_beyond_horizon` in the generated
  prediction artifact.

### Comparison With Previous DeepHit Audit Metrics
- Previous test final-risk C-index: approximately 0.4879; new value: 0.7529.
- Previous test Ctd Antolini: approximately 0.7509; new value: 0.7695.
- Previous test mean horizon C-index: approximately 0.73; new value: 0.7505.
- Previous test IBS: approximately 0.4044; new value: 0.1107.
- Previous test IBLL/NBLL: approximately 1.0431; new value: 0.3531.
- Previous `S(10)` was forced to 0 by construction; new `S(10)` is nonzero for
  all rows and equals the learned tail probability.

### Interpretation
- The tail-support, censored-horizon likelihood and ranking-loss fixes corrected
  a major structural probability-quality issue in DeepHit.
- Calibration-style curve metrics improved substantially and now sit in the
  same broad region as the strongest static curve baseline previously recorded
  for PCHazard.
- Discrimination did not collapse; Ctd and horizon C-index remain strong, and
  final-risk C-index is no longer pathological. Final-risk C-index should still
  remain secondary for curve-producing models.
- DeepHit can now move to targeted diagnostics and cautious hyperparameter
  tuning, but calibration plots, survival-curve comparison against PCHazard and
  a small synthetic overfit test remain open.

### Follow-up
- [ ] Generate calibration plots for the corrected DeepHit model.
- [ ] Compare corrected DeepHit survival curves against PCHazard survival
      curves.
- [ ] Run a small synthetic overfit test for DeepHit.
- [ ] Tune `alpha`, `beta`, `gamma`, `ranking_sigma`, learning rate, batch size
      and early stopping only after diagnostics.

## EXP-005 — CoxPH old-configuration diagnostic through final-static pipeline

Date: 2026-06-08
Status: completed
Model: CoxPH
Dataset: static MIMIC-IV adult ICU static dataset
Config: generated from `configs/coxph.yaml`
Seed: 42
Run directory: `outputs/diagnostics/coxph/seed_42`
Related decision: DEC-008

### Goal
- Check whether the new tuning/final-seed pipeline can reproduce the previous
  CoxPH benchmark when using the old stable CoxPH configuration.

### Command
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -c "import copy; from pathlib import Path; from src.utils.config import load_yaml; from scripts.tune_static_models import prepare_run_config, run_training, save_config_snapshot; from src.utils.logger import get_logger; cfg=load_yaml('configs/coxph.yaml'); rc, rd=prepare_run_config(copy.deepcopy(cfg),'coxph','diagnostic_old_fixed',{'penalizer':0.1,'l1_ratio':0.0},42,'outputs/diagnostics',phase='final_static',include_test=True,save_predictions=False,save_models=False,save_checkpoints=False); save_config_snapshot(rc, rd); metrics=run_training(rc, get_logger('coxph_old_fixed_diagnostic')); print('run_dir=', rd); print('val_ctd=', metrics['splits']['validation']['ctd_antolini']); print('test_ctd=', metrics['splits']['test']['ctd_antolini']); print('test_ibs=', metrics['splits']['test']['ibs']); print('test_ibll=', metrics['splits']['test']['ibll'])"
```

### Inputs
- Dataset path: `data/processed/static/`
- Train split: `data/processed/static/train_static.parquet`
- Validation split: `data/processed/static/val_static.parquet`
- Test split: `data/processed/static/test_static.parquet`
- Preprocessor: `outputs/preprocessors/static_preprocessor.pkl`
- Hyperparameters: `penalizer=0.1`, `l1_ratio=0.0`

### Outputs
- Metrics: `outputs/diagnostics/coxph/seed_42/metrics/coxph/coxph_metrics.json`
- Config snapshot: `outputs/diagnostics/coxph/seed_42/config_snapshot.yaml`
- Predictions: disabled for this diagnostic
- Model artifact: disabled for this diagnostic

### Results
- Validation Ctd/Harrell: 0.7415.
- Test Ctd/Harrell: 0.7411.
- Test IBS: 0.1147.
- Test IBLL/NBLL: 0.3693.

### Interpretation
- The new final-static pipeline reproduces the previous CoxPH benchmark when
  the old stable penalization is used.
- The CoxPH smoke-test regression is therefore attributed to an unstable
  `penalizer=0.01` candidate and incomplete smoke tuning, not to changed split,
  feature set, duration/event columns, preprocessing or evaluation grid.

### Follow-up
- [ ] Re-run CoxPH validation-only tuning with
      `penalizer: [0.0, 0.001, 0.01, 0.1]` before Lightning AI final runs.

## EXP-006 — Final static seed runs for DeepHit, DeepSurv and PCHazard

Date: 2026-06-09
Status: completed
Model: DeepHit, DeepSurv, PCHazard
Dataset: static MIMIC-IV adult ICU static dataset
Config: `configs/static_tuning.yaml`
Seed: 42, 123, 2026
Run directory: `outputs/final_static/`
Related decision: DEC-007

### Goal
- Summarize the completed three-seed final static evaluation for the three
  neural/static curve-producing models with validation-selected
  hyperparameters.

### Command
```bash
python scripts/run_final_static_seeds.py --config configs/static_tuning.yaml --models deephit deepsurv pchazard
```

### Inputs
- Dataset path: `data/processed/static/`
- Train split: `data/processed/static/train_static.parquet`
- Validation split: `data/processed/static/val_static.parquet`
- Test split: `data/processed/static/test_static.parquet`
- Tuning selections:
  `outputs/tuning/deephit/best_hyperparameters.json`,
  `outputs/tuning/deepsurv/best_hyperparameters.json`,
  `outputs/tuning/pchazard/best_hyperparameters.json`

### Outputs
- DeepHit summary: `outputs/final_static/deephit/final_seed_results.csv`
- DeepSurv summary: `outputs/final_static/deepsurv/final_seed_results.csv`
- PCHazard summary: `outputs/final_static/pchazard/final_seed_results.csv`
- Per-seed metrics:
  `outputs/final_static/{model}/seed_{seed}/metrics/{model}/{model}_metrics.json`

### Selected Hyperparameters
- DeepHit: `shared_layers=[128, 64]`, `cause_layers=[64]`,
  `dropout=0.1`, `learning_rate=0.0005`, `alpha=1.0`, `beta=0.5`,
  `gamma=0.0`, `ranking_sigma=0.1`, `include_tail_category=true`.
- DeepSurv: `hidden_layers=[128, 64]`, `dropout=0.1`,
  `learning_rate=0.0001`, `weight_decay=0.001`.
- PCHazard: `hidden_layers=[128, 64]`, `dropout=0.3`,
  `learning_rate=0.0005`.

### Results
Mean test metrics across seeds:

| Model | Test Ctd/Harrell | Test mean C-index@h | Test IBS | Test IBLL/NBLL |
| --- | ---: | ---: | ---: | ---: |
| DeepHit | 0.7690 | 0.7490 | 0.1104 | 0.3526 |
| DeepSurv | 0.7615 | 0.7463 | 0.1110 | 0.3560 |
| PCHazard | 0.7688 | 0.7491 | 0.1095 | 0.3507 |

Standard deviations across seeds:

| Model | Ctd sd | Mean C-index@h sd | IBS sd | IBLL sd |
| --- | ---: | ---: | ---: | ---: |
| DeepHit | 0.0017 | 0.0013 | 0.0002 | 0.0011 |
| DeepSurv | 0.0008 | 0.0004 | 0.0005 | 0.0013 |
| PCHazard | 0.0014 | 0.0005 | 0.0002 | 0.0008 |

### Interpretation
- DeepHit has the highest mean test Ctd by a very small margin over PCHazard.
- PCHazard has the best calibration/error metrics, with the lowest mean test
  IBS and IBLL/NBLL.
- DeepSurv remains stable but is below the two curve-discrete models on mean
  test Ctd and calibration metrics.
- Differences between DeepHit and PCHazard are small, so final claims should be
  cautious unless supported by uncertainty intervals or additional diagnostics.

### Follow-up
- [ ] Complete CoxPH tuning/final-seed evaluation before calling this the final
      complete static benchmark.
- [ ] Consolidate final static metrics into one comparison artifact/table.

## EXP-007 — Complete final static seed comparison including CoxPH

Date: 2026-06-09
Status: completed
Model: CoxPH, DeepHit, DeepSurv, PCHazard
Dataset: static MIMIC-IV adult ICU static dataset
Config: `configs/static_tuning.yaml`
Seed: 42, 123, 2026
Run directory: `outputs/final_static/`
Related decisions: DEC-007, DEC-008

### Goal
- Add CoxPH final-seed results to the static benchmark and compare all four
  tuned static models under the common fixed grid protocol.

### Command
```bash
python scripts/run_final_static_seeds.py --config configs/static_tuning.yaml --models coxph deephit deepsurv pchazard
```

### Inputs
- Dataset path: `data/processed/static/`
- Train split: `data/processed/static/train_static.parquet`
- Validation split: `data/processed/static/val_static.parquet`
- Test split: `data/processed/static/test_static.parquet`
- Tuning selections:
  `outputs/tuning/coxph/best_hyperparameters.json`,
  `outputs/tuning/deephit/best_hyperparameters.json`,
  `outputs/tuning/deepsurv/best_hyperparameters.json`,
  `outputs/tuning/pchazard/best_hyperparameters.json`

### Outputs
- CoxPH summary: `outputs/final_static/coxph/final_seed_results.csv`
- DeepHit summary: `outputs/final_static/deephit/final_seed_results.csv`
- DeepSurv summary: `outputs/final_static/deepsurv/final_seed_results.csv`
- PCHazard summary: `outputs/final_static/pchazard/final_seed_results.csv`
- Per-seed metrics:
  `outputs/final_static/{model}/seed_{seed}/metrics/{model}/{model}_metrics.json`

### Selected Hyperparameters
- CoxPH: `penalizer=0.1`, `l1_ratio=0.0`.
- DeepHit: `shared_layers=[128, 64]`, `cause_layers=[64]`,
  `dropout=0.1`, `learning_rate=0.0005`, `alpha=1.0`, `beta=0.5`,
  `gamma=0.0`, `ranking_sigma=0.1`, `include_tail_category=true`.
- DeepSurv: `hidden_layers=[128, 64]`, `dropout=0.1`,
  `learning_rate=0.0001`, `weight_decay=0.001`.
- PCHazard: `hidden_layers=[128, 64]`, `dropout=0.3`,
  `learning_rate=0.0005`.

### Results
Mean test metrics across seeds:

| Model | Test Harrell/Ctd | Test mean C-index@h | Test IBS | Test IBLL/NBLL |
| --- | ---: | ---: | ---: | ---: |
| CoxPH | 0.7411 | 0.7266 | 0.1147 | 0.3693 |
| DeepHit | 0.7690 | 0.7490 | 0.1104 | 0.3526 |
| DeepSurv | 0.7615 | 0.7463 | 0.1110 | 0.3560 |
| PCHazard | 0.7688 | 0.7491 | 0.1095 | 0.3507 |

Standard deviations across seeds:

| Model | Harrell/Ctd sd | Mean C-index@h sd | IBS sd | IBLL sd |
| --- | ---: | ---: | ---: | ---: |
| CoxPH | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DeepHit | 0.0017 | 0.0013 | 0.0002 | 0.0011 |
| DeepSurv | 0.0008 | 0.0004 | 0.0005 | 0.0013 |
| PCHazard | 0.0014 | 0.0005 | 0.0002 | 0.0008 |

### Interpretation
- CoxPH now matches the previously stable benchmark after validation tuning
  selected the stronger penalization `penalizer=0.1`.
- DeepSurv improves over CoxPH in all reported test metrics, supporting the
  value of a nonlinear proportional-risk representation over the linear Cox
  baseline.
- DeepHit and PCHazard form the strongest static pair. DeepHit has the highest
  mean test Ctd by a negligible margin, while PCHazard has the best mean
  horizon C-index and the lowest IBS and IBLL/NBLL.
- The model ranking should be reported cautiously because DeepHit and PCHazard
  are separated by very small absolute differences.
- CoxPH has zero seed variation because this fitted Cox model is deterministic
  once the selected hyperparameters and data split are fixed.

### Follow-up
- [ ] Consolidate final static metrics into one comparison artifact/table.
- [ ] Compare the completed static benchmark with dynamic survival models once
      the dynamic pipeline scope is finalized.

## EXP-008 — Thesis-ready final static tables and figures

Date: 2026-06-10
Status: completed
Model: CoxPH, DeepSurv, PCHazard, DeepHit
Dataset: static MIMIC-IV adult ICU static dataset
Source run directory: `outputs/final_static/`
Output tables directory: `outputs/thesis_tables/static/`
Output figures directory: `outputs/figures/static/`
Related decisions: DEC-007, DEC-008

### Goal
- Generate clean thesis-ready static result tables, a regenerated final static
  comparison CSV and compact figures from final static seed outputs only.

### Command
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe - <inline metrics/figures generation script>
```

### Inputs
- Final per-model summaries:
  `outputs/final_static/{coxph,deepsurv,pchazard,deephit}/final_seed_results.csv`
- Final per-model seed summaries:
  `outputs/final_static/{coxph,deepsurv,pchazard,deephit}/final_seed_summary.json`
- Final per-seed metric JSONs:
  `outputs/final_static/{model}/seed_{seed}/metrics/{model}/{model}_metrics.json`
- Validation-selected hyperparameters:
  `outputs/tuning/{model}/best_hyperparameters.json`

### Outputs
- Consolidated final metrics:
  `outputs/metrics/final_static_model_comparison.csv`
- Thesis tables:
  `outputs/thesis_tables/static/static_final_test_comparison.{csv,tex}`,
  `outputs/thesis_tables/static/static_horizon_c_index.{csv,tex}`,
  `outputs/thesis_tables/static/static_probabilistic_metrics.{csv,tex}`,
  `outputs/thesis_tables/static/static_selected_hyperparameters.{csv,tex}`,
  `outputs/thesis_tables/static/static_per_seed_results.{csv,tex}`
- Thesis figures:
  `outputs/figures/static/static_ctd_antolini_comparison.png`,
  `outputs/figures/static/static_ibs_comparison.png`,
  `outputs/figures/static/static_ibll_nbll_comparison.png`,
  `outputs/figures/static/static_horizon_c_index.png`,
  `outputs/figures/static/static_discrimination_vs_calibration_summary.png`

### Validation
- Confirmed DeepHit selected hyperparameters include
  `include_tail_category=true`.
- Confirmed final DeepHit IBS is in the corrected range near 0.1104, not the
  old pre-tail audit value near 0.40.
- Confirmed CoxPH test standard deviation is zero across final seeds, matching
  deterministic behavior.
- Confirmed non-applicable scalar Harrell metrics are left empty/NA for
  curve-based final comparison rows.
- Did not use stale `outputs/metrics/static_model_comparison.csv` as source.

### Notes
- Kaplan-Meier remains descriptive only and is not included in predictive final
  comparison tables.
- Individual survival-curve examples were not generated because the available
  curve CSVs are matrix-style prediction artifacts without clinically
  interpretable patient identifiers; such a figure would be arbitrary for the
  main memory.

## EXP-009 — static_72h_pycox audit validation after grid/PCHazard fix

Date: 2026-06-12
Status: completed with caveat
Model: LogisticHazard, PCHazard, DeepHitSingle; partial Kaplan-Meier/CoxPH checks
Dataset: processed `static_72h_pycox` train/validation splits
Config: `configs/static_72h_tuning.yaml`
Related decisions: DEC-010

### Goal
- Audit pycox discrete-time cuts, DeepHitSingle tail behavior, PCHazard
  interpolation and the separation between integrated-metric and horizon grids
  before final 3-seed evaluation.

### Commands
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile src/evaluation/static_72h_metrics.py src/models/static_72h_pycox.py scripts/tune_static_72h_models.py scripts/run_final_static_72h_seeds.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_static_72h_pipeline.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe - <inline static_72h audit validation script>
```

### Outputs
- `outputs/static_72h/audit/deephit_single_time_grid_audit.json`
- `outputs/static_72h/audit/deephit_single_survival_tail_check.csv`
- `outputs/static_72h/audit/pchazard_audit.json`
- `outputs/static_72h/audit/evaluation_grids.json`
- `outputs/static_72h/audit/discrete_time_cuts_summary.json`
- `outputs/static_72h/audit/survival_curve_sanity_checks.csv`
- `outputs/static_72h/audit/discrete_audit_validation_summary.json`
- Temporary audit run directories under `outputs/static_72h/audit_runs/`

### Results
- DeepHitSingle `labtrans.cuts` spans 0 to 10 days with 10 approximately
  equally spaced cuts. Validation survival at 10 days remained positive and
  heterogeneous: min 0.00949, mean 0.66919, max 0.96999, share below `1e-6` 0.
- PCHazard with `sub=10` improved validation Antolini Ctd from the stale
  pre-fix value 0.40349 to 0.64926. Mean horizon C-index was 0.69561.
- PCHazard validation survival curves were monotone, finite and inside
  `[0, 1]`; validation survival at 10 days had min 0.01092, mean 0.70618,
  max 0.99575.
- LogisticHazard and DeepHitSingle validation Ctd after the grid change were
  0.68528 and 0.68852 respectively.
- IBS/IBLL now use a 100-point per-split integration grid; daily horizons
  `[1, ..., 10]` remain reserved for horizon C-index.

### Validation
- `pytest tests/test_static_72h_pipeline.py`: 4 passed.
- `py_compile`: passed.

### Caveat
- A full audit validation attempt including DeepSurv stopped when pycox CoxPH
  tried to allocate approximately 2.20 GiB while predicting survival over a
  dense train+validation Cox time index. This does not affect the DeepHitSingle
  or PCHazard audit, but should be considered before running all final models
  with the current DeepSurv survival prediction path.

## EXP-010 — dynamic_72h dataset build

Date: 2026-06-12
Status: completed
Model: not applicable
Dataset: `dynamic_72h`
Config: `configs/dynamic_72h_data.yaml`
Seed: 42
Run directory: `data/processed/dynamic_72h/`; audit under `outputs/dynamic_72h/audit/`
Related decision: DEC-011

### Goal
- Build the dynamic 72-hour input dataset for future DySurv/Dynamic-DeepHit
  experiments while preserving the `static_72h_pycox` cohort, splits and
  targets.

### Command
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/build_dynamic_72h_data.py --config configs/dynamic_72h_data.yaml --force
```

### Inputs
- Static train split: `data/processed/static_72h/train_static_72h.parquet`
- Static validation split: `data/processed/static_72h/val_static_72h.parquet`
- Static test split: `data/processed/static_72h/test_static_72h.parquet`
- Temporal chart source: `data/processed/mimic_extraction/timeseries.csv`
- Temporal lab source: `data/processed/mimic_extraction/timeserieslab.csv`

### Outputs
- Train arrays: `data/processed/dynamic_72h/train_dynamic_72h.npz`
- Validation arrays: `data/processed/dynamic_72h/val_dynamic_72h.npz`
- Test arrays: `data/processed/dynamic_72h/test_dynamic_72h.npz`
- Dataset summary: `data/processed/dynamic_72h/dynamic_72h_dataset_summary.json`
- Temporal columns: `data/processed/dynamic_72h/temporal_feature_columns.json`
- Static columns: `data/processed/dynamic_72h/static_feature_columns.json`
- Preprocessing metadata: `data/processed/dynamic_72h/preprocessing_metadata.json`
- Preprocessor: `data/processed/dynamic_72h/preprocessor.joblib`
- Audit files:
  `outputs/dynamic_72h/audit/dynamic_72h_data_audit.json`,
  `outputs/dynamic_72h/audit/missingness_summary.csv`,
  `outputs/dynamic_72h/audit/temporal_coverage_summary.csv`,
  `outputs/dynamic_72h/audit/feature_coverage_by_split.csv`,
  `outputs/dynamic_72h/audit/hourly_missingness_summary.csv`

### Results
- Selected 146 temporal features and 28 static features.
- Train: `X_seq=(18706, 72, 146)`, `M_seq=(18706, 72, 146)`,
  `X_static=(18706, 28)`, event rate 0.1369.
- Validation: `X_seq=(6236, 72, 146)`, `M_seq=(6236, 72, 146)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Test: `X_seq=(6236, 72, 146)`, `M_seq=(6236, 72, 146)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Used offset range was strictly inside the first 72 hours: minimum 0 minutes,
  maximum 4319 minutes.
- Checks passed: exact static ID ordering, no split overlap, positive
  `duration_rel_days`, no `offset_minutes >= 4320`, train-only feature
  selection, train-only imputation and train-only scaling.
- Raw temporal observed fraction before imputation was 0.1135 for train,
  0.1135 for validation and 0.1127 for test.
- Every train and validation patient had at least one selected temporal
  measurement; 6234/6236 test patients did.

### Validation
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_dynamic_72h_dataset.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile src/data/dynamic_72h_dataset.py scripts/build_dynamic_72h_data.py
```

- Unit tests: 2 passed.
- `py_compile`: passed.

### Interpretation
- The dynamic 72h dataset is now available as a reproducible preprocessing
  artifact for dynamic model adaptation.
- No model training or dynamic-vs-static evaluation was performed in this run.
- The large missingness fractions are expected for hourly ICU trajectories and
  are explicitly represented through `M_seq`.

### Follow-up
- [ ] Adapt and smoke-test DySurv on `dynamic_72h`.
- [ ] Adapt and smoke-test Dynamic-DeepHit on `dynamic_72h`.
- [ ] Decide whether `delta_seq` is needed before dynamic model training.

## EXP-011 — DySurv-compatible dynamic_72h feature subset

Date: 2026-06-12
Status: completed
Model: not applicable
Dataset: `dynamic_72h_dysurv_features`
Config: not applicable; derived from `data/processed/dynamic_72h/`
Seed: not applicable
Run directory: `data/processed/dynamic_72h_dysurv_features/`
Related decision: DEC-012

### Goal
- Create a smaller dynamic 72-hour dataset for a first DySurv-style training
  pass by removing temporal variables not represented in the DySurv reference
  table.

### Command
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/filter_dynamic_72h_dysurv_features.py --force
```

### Inputs
- Source arrays: `data/processed/dynamic_72h/{train,val,test}_dynamic_72h.npz`
- Source temporal columns:
  `data/processed/dynamic_72h/temporal_feature_columns.json`

### Outputs
- Train arrays:
  `data/processed/dynamic_72h_dysurv_features/train_dynamic_72h.npz`
- Validation arrays:
  `data/processed/dynamic_72h_dysurv_features/val_dynamic_72h.npz`
- Test arrays:
  `data/processed/dynamic_72h_dysurv_features/test_dynamic_72h.npz`
- Feature list:
  `data/processed/dynamic_72h_dysurv_features/temporal_feature_columns.json`
- Summary:
  `data/processed/dynamic_72h_dysurv_features/dynamic_72h_dysurv_feature_summary.json`

### Results
- Source temporal features: 146.
- Selected DySurv-compatible temporal features: 76.
- Removed temporal features: 70.
- Missing DySurv table variables in the generated temporal feature set:
  `ALT`, `Bilirubin`, `AST`, `Alkaline Phosphatase`.
- Train: `X_seq=(18706, 72, 76)`, `M_seq=(18706, 72, 76)`,
  `X_static=(18706, 28)`, event rate 0.1369.
- Validation: `X_seq=(6236, 72, 76)`, `M_seq=(6236, 72, 76)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Test: `X_seq=(6236, 72, 76)`, `M_seq=(6236, 72, 76)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Observed temporal mask fraction: train 0.1307, validation 0.1309,
  test 0.1299.

### Validation
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile scripts/filter_dynamic_72h_dysurv_features.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -c "<shape/NaN/mask validation>"
```

- `py_compile`: passed.
- Verified no NaNs in `X_seq`.
- Verified `M_seq` remains binary.
- Verified static covariates and event rates are unchanged.

### Interpretation
- This subset is ready for a quick first dynamic-model training pass.
- It should not replace the full `dynamic_72h` dataset for final comparisons
  unless that methodological choice is explicitly adopted.

## EXP-012 — Additional in-place reduction of dynamic_72h_dysurv_features

Date: 2026-06-12
Status: completed
Model: not applicable
Dataset: `dynamic_72h_dysurv_features`
Config: not applicable; derived in place from the existing 76-feature subset
Seed: not applicable
Run directory: `data/processed/dynamic_72h_dysurv_features/`
Related decision: DEC-013

### Goal
- Overwrite the existing DySurv-compatible subset by removing 15 additional
  chart-derived temporal variables before the first dynamic-model training
  pass.

### Command
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe - <inline in-place NPZ feature filtering script>
```

### Removed Features
- `chart::Anion gap`
- `chart::Creatinine (serum)`
- `chart::Hematocrit (serum)`
- `chart::Potassium (serum)`
- `chart::Potassium (whole blood)`
- `chart::Glucose (serum)`
- `chart::Glucose (whole blood)`
- `chart::Glucose finger stick (range 70-100)`
- `chart::Calcium non-ionized`
- `chart::Ionized Calcium`
- `chart::Chloride (serum)`
- `chart::Hemoglobin`
- `chart::Magnesium`
- `chart::Platelet Count`
- `chart::Sodium (serum)`

### Outputs
- Overwritten train arrays:
  `data/processed/dynamic_72h_dysurv_features/train_dynamic_72h.npz`
- Overwritten validation arrays:
  `data/processed/dynamic_72h_dysurv_features/val_dynamic_72h.npz`
- Overwritten test arrays:
  `data/processed/dynamic_72h_dysurv_features/test_dynamic_72h.npz`
- Updated feature list:
  `data/processed/dynamic_72h_dysurv_features/temporal_feature_columns.json`
- Updated summary:
  `data/processed/dynamic_72h_dysurv_features/dynamic_72h_dysurv_feature_summary.json`

### Results
- Previous subset features: 76.
- Removed features: 15.
- Current subset features: 61.
- Train: `X_seq=(18706, 72, 61)`, `M_seq=(18706, 72, 61)`,
  `X_static=(18706, 28)`, event rate 0.1369.
- Validation: `X_seq=(6236, 72, 61)`, `M_seq=(6236, 72, 61)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Test: `X_seq=(6236, 72, 61)`, `M_seq=(6236, 72, 61)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Observed temporal mask fraction: train 0.1448, validation 0.1450,
  test 0.1439.

### Validation
- Verified all 15 requested features are absent from
  `temporal_feature_columns.json`.
- Verified no NaNs in `X_seq`.
- Verified `M_seq` remains binary.
- Verified static covariates and event rates are unchanged.

### Interpretation
- `data/processed/dynamic_72h_dysurv_features/` now refers to the 61-feature
  reduced subset, not the previous 76-feature subset.

## EXP-013 — dynamic_72h DySurv/Dynamic-DeepHit smoke tuning

Date: 2026-06-12
Status: completed
Model: DySurv, Dynamic-DeepHit
Dataset: `dynamic_72h_dysurv_features`
Config: `configs/dynamic_72h_tuning.yaml`
Seed: 42
Run directory: `outputs/dynamic_72h/tuning/`
Related decision: DEC-014

### Goal
- Verify that the new dynamic_72h model layer can load the 61-feature dynamic
  dataset, train DySurv and Dynamic-DeepHit on train only, evaluate validation
  metrics, write audits and keep test locked during tuning.

### Command
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv dynamic_deephit --max-runs 2 --sample-size 128 --device cpu --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts\tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dynamic_deephit --max-runs 1 --sample-size 128 --device cpu --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts\run_final_dynamic_72h_seeds.py --config configs/dynamic_72h_final.yaml --model dysurv dynamic_deephit --dry-run --sample-size 32 --device cpu
```

The second command regenerated the Dynamic-DeepHit smoke run after adding the
probability/CIF audit.

### Inputs
- Dynamic dataset: `data/processed/dynamic_72h_dysurv_features/`
- Input mode: `values_plus_mask_plus_static`
- Model input shape in smoke runs: `[N, 72, 150]`, from 61 values, 61 masks
  and 28 repeated static features.
- Tuning splits: train and validation only.

### Outputs
- DySurv run:
  `outputs/dynamic_72h/tuning/dysurv/dysurv_cfg_001/seed_42/`
- Dynamic-DeepHit run:
  `outputs/dynamic_72h/tuning/dynamic_deephit/dynamic_deephit_cfg_001/seed_42/`
- Audits:
  `outputs/dynamic_72h/audit/dysurv/dysurv_cfg_001/seed_42/`
  and
  `outputs/dynamic_72h/audit/dynamic_deephit/dynamic_deephit_cfg_001/seed_42/`

### Results
- DySurv validation Ctd Antolini: 0.6367.
- DySurv validation IBS: 0.6475.
- DySurv validation IBLL/NBLL: 2.5545.
- DySurv validation mean horizon C-index: 0.6407.
- Dynamic-DeepHit validation Ctd Antolini: 0.7543.
- Dynamic-DeepHit validation IBS: 0.1385.
- Dynamic-DeepHit validation IBLL/NBLL: 0.4416.
- Dynamic-DeepHit validation mean horizon C-index: 0.7741.
- No test metrics were recorded during tuning.

### Audits
- Split overlap checks passed.
- Discrete target indices were inside `[0, 9]`.
- Input metadata for both models: 61 temporal value features, 61 mask features,
  28 static features and 150 model input features.
- Survival curves had no NaNs, stayed in `[0, 1]` and were monotone
  non-increasing.
- Dynamic-DeepHit PMF sums were approximately 1, CIF was non-decreasing,
  survival was non-increasing and `S(10)` was not forced to zero.
- DySurv recorded survival, reconstruction and KL losses separately in
  `train_log.csv`.

### Interpretation
- The dynamic model pipeline is executable end to end on a small smoke subset.
- The smoke metrics are not final evidence because only 128 patients per split
  and one small config per model were used.
- Full validation-only tuning should be run before any final 3-seed dynamic
  evaluation.

### Follow-up
- [ ] Run full validation-only dynamic_72h tuning.
- [ ] Inspect DySurv calibration/loss behavior before final seeds.
- [ ] Launch final 3-seed dynamic evaluation only after tuning selection is
      stable.

## EXP-014 — DySurv posterior-collapse audit and tiny-overfit controls

Date: 2026-06-14
Status: completed
Model: DySurv
Dataset: `dynamic_72h_dysurv_features`
Config: diagnostic use of selected `dysurv_cfg_032` architecture
Seed: 31415 for controlled checks
Run directory: no model run directory; report at
`outputs/dynamic_72h/dysurv_audit_report.md`

### Goal
- Determine whether almost identical final survival curves were caused by
  repeated inputs, broadcasting, an incorrect survival loss or training
  collapse.

### Checks run
- Read-only array/dataloader variability audit for train, validation and test.
- Numerical equivalence check between the current LogisticHazard NLL and the
  cumulative BCE formula used by the reference notebook.
- Analysis of final seeds 42, 123 and 2026 and 85 tuning train logs.
- Two 64-patient, 200-epoch CPU controls: current loss weights and
  survival-only.
- Focused unit tests: `tests/test_dynamic_72h_models.py`.

### Results
- No repeated-input, batch-loss or prediction-broadcasting bug was found.
- Final seed risk10 ranges were `0.003110` (42), `0` (123) and `0.004192`
  (2026), close to the test Kaplan-Meier marginal risk `0.300532`.
- Final KL losses were approximately zero, consistent with posterior collapse.
- A train-mean reconstruction baseline achieved validation MSE `0.512927`,
  slightly better than the selected DySurv decoder (`0.5147--0.5150`).
- The unscaled repeated static `hour` feature accounts for about 59.75% of the
  combined mean-reconstruction MSE.
- Tiny controls showed that different/perturbed inputs can change predictions;
  survival-only training produced substantially larger individual variation.
- NLL equivalence difference: `0.0`; focused tests: 2 passed.

### Interpretation
- The current runs collapse toward a marginal survival curve because of the
  adapted VAE/reconstruction objective, not because evaluation copies one
  prediction across patients.
- Current DySurv final results should not be used for individual dynamic-model
  interpretation until the adaptation is corrected and retrained.

## EXP-015 — DySurv-faithful dataset, tiny-overfit and smoke validation

Date: 2026-06-14
Status: completed smoke; full tuning pending
Model: DySurv faithful 72h
Dataset: `dysurv_faithful_72h`
Config: `configs/dysurv_faithful_72h.yaml`
Seed: 42
Run directory: `outputs/dysurv_faithful_72h/`
Related decision: DEC-016

### Goal
- Verify the isolated faithful data/model pipeline, demonstrate that the model
  can produce individualized risk, and test collapse-aware epoch selection
  without using test data.

### Commands
```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/prepare_dysurv_faithful_72h_dataset.py --config configs/dysurv_faithful_72h.yaml --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/audit_dysurv_faithful_72h.py --config configs/dysurv_faithful_72h.yaml --run-tiny-overfit --device cpu
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dysurv_faithful_72h.py --config configs/dysurv_faithful_72h.yaml --max-runs 1 --sample-size 128 --device cpu --force
```

### Dataset results
- Train: `18706 x 72 x 61`; event rate `0.136908`.
- Validation: `6236 x 72 x 61`; event rate `0.136947`.
- Test: `6236 x 72 x 61`; event rate `0.136947`.
- Static shape: 28 variables per patient.
- Split-overlap, finite-value and binary-mask checks passed.
- Imputation residuals and static scaling were fitted on train only.

### Tiny-overfit results
- Patients: 64 per train/validation split; survival-only diagnostic.
- Epochs: 100.
- Train survival loss: `3.275400 -> 0.219514`.
- Final train risk10 standard deviation: `0.363450`.
- Final validation risk10 standard deviation: `0.388562`.
- Final collapse flag: false.

### Weighted smoke results
- Patients: 128 per train/validation split.
- Configuration: first 16-grid candidate, with survival/reconstruction/KL
  weights `0.70/0.20/0.10` and 20-epoch KL warm-up.
- No test data or test metrics were loaded.
- Pure metric-best epoch: 14, validation Ctd `0.575526`, but collapsed with
  risk10 standard deviation `0.000523` and range `0.002368`.
- Selected non-collapsed epoch: 8, validation Ctd `0.566922`, IBS `0.281065`,
  IBLL/NBLL `0.768921`, mean horizon C-index `0.554385`.
- Selected risk10 standard deviation: `0.029566`; range `0.134672`.

### Interpretation
- The faithful architecture can overfit a small sample and produce strongly
  individualized curves, so it is not structurally forced to a marginal curve.
- The weighted smoke still demonstrates collapse pressure after early epochs;
  collapse-aware checkpoint selection is necessary.
- These metrics are smoke diagnostics only and must not be used as final model
  evidence.

## EXP-016 — Full DySurv-faithful tuning and final three-seed evaluation

Date: 2026-06-14
Status: completed
Model: DySurv faithful 72h
Dataset: `dysurv_faithful_72h`
Config: `configs/dysurv_faithful_72h.yaml`
Seeds: tuning `42`; final `42`, `123`, `2026`
Run directory: `outputs/dysurv_faithful_72h/`
Related decision: DEC-016

### Selection

- Completed all 16 validation-only candidates without loading test data.
- Selected `dysurv_faithful_cfg_002` by validation Ctd Antolini, with IBLL as
  tiebreaker and preference for non-collapsed candidates.
- Hyperparameters: learning rate `0.001`, dropout `0.1`, weight decay `0.0001`,
  batch size `128`, KL warm-up `50`, and loss weights survival/reconstruction/KL
  `0.70/0.20/0.10`.
- Selected validation metrics: Ctd `0.787732`, mean horizon C-index `0.791629`,
  IBS `0.298985`, IBLL/NBLL `0.825261`.
- None of the 16 candidates was flagged as collapsed.

### Final results

- Mean test Ctd Antolini: `0.777839` (stored std `0.002461`).
- Mean test horizon C-index: `0.776494` (stored std `0.008187`).
- Mean test IBS: `0.251928` (stored std `0.032800`).
- Mean test IBLL/NBLL: `0.706178` (stored std `0.083272`).
- No final seed was flagged as collapsed. Mean test risk10 standard deviation
  was `0.125594`, with mean range `0.604031`.
- Horizon discrimination decreased from mean C-index `0.811680` at day 1 to
  `0.750409` at day 10.

### Interpretation

- The faithful adaptation resolves the previous near-constant prediction and
  posterior-collapse failure and provides individualized, monotone curves.
- Discrimination is stable across seeds, but probabilistic quality is weaker
  and more seed-dependent.
- Mean predicted 10-day risk was approximately `0.813`, `0.717` and `0.696`
  for seeds 42, 123 and 2026, respectively, while the observed event indicator
  rate was `0.137`. Together with IBS/IBLL, this indicates substantial absolute
  risk miscalibration/overprediction despite useful ranking.
- Final interpretation should therefore separate discrimination from
  calibration and should not describe this model as globally superior.
## EXP-017 — Dynamic-DeepHit-faithful smoke and tiny-overfit validation

**Date:** 2026-06-15

**Purpose:** Verify that the isolated Dynamic-DeepHit adaptation can train on
the faithful 72h dataset, save auditable artifacts, produce individualized
curves and overfit a small sample before full validation tuning.

**Config:** `configs/dynamic_deephit_faithful_72h.yaml`

**Commands:**

```bash
python scripts/tune_dynamic_deephit_faithful_72h.py --config configs/dynamic_deephit_faithful_72h.yaml --max-runs 1 --sample-size 128 --device cpu --force
python scripts/audit_dynamic_deephit_faithful_72h.py --config configs/dynamic_deephit_faithful_72h.yaml --run-tiny-overfit --device cpu
python scripts/audit_dynamic_deephit_faithful_72h.py --config configs/dynamic_deephit_faithful_72h.yaml
```

**Results:** The smoke candidate completed with validation Ctd `0.836520`,
IBS `0.126745`, IBLL/NBLL `0.406526`, mean horizon C-index `0.863040`,
`risk10_std=0.150010`, 128 unique rounded risks and no collapse flag. Test was
not evaluated. The 64-patient tiny-overfit reduced train total loss from
`1.095767` to `0.021100` and train PMF NLL from `1.189985` to `0.002844`; final
train `risk10_std=0.374864`, confirming nonconstant individualized output.

The first audit command completed training but exited during report generation
because a boolean parsing helper was defined after the script entry point. The
helper order was corrected and the report was regenerated successfully without
retraining. These reduced-sample metrics are diagnostics, not final model
results.

**Outputs:** `outputs/dynamic_deephit_faithful_72h/smoke/`,
`outputs/dynamic_deephit_faithful_72h/tiny_overfit/` and
`outputs/dynamic_deephit_faithful_72h/dynamic_deephit_faithful_audit_report.md`.

**Decision:** Proceed to full validation-only tuning; do not run final seeds
until the selected validation candidate and its probability diagnostics have
been reviewed. See DEC-018.

## EXP-018 — DySurv static faithful smoke and tiny-overfit validation

**Date:** 2026-06-15

**Purpose:** Validate the isolated static-only MLP-VAE DySurv implementation on
the exact faithful 72h cohort before full tuning.

**Config:** `configs/dysurv_static_faithful_72h.yaml`

**Commands:**

```bash
python scripts/tune_dysurv_static_faithful_72h.py --config configs/dysurv_static_faithful_72h.yaml --dry-run --device cpu
python scripts/tune_dysurv_static_faithful_72h.py --config configs/dysurv_static_faithful_72h.yaml --max-runs 1 --sample-size 128 --device cpu --force
python scripts/audit_dysurv_static_faithful_72h.py --config configs/dysurv_static_faithful_72h.yaml --run-tiny-overfit --device cpu
```

**Dataset:** The existing `data/processed/dysurv_faithful_72h/` split files,
with train/validation/test patient counts `18706/6236/6236` and 28
train-standardized static covariates. Ordered ID and target hashes are stored
in `outputs/dysurv_static_faithful_72h/audit/dataset_identity.json`.

**Results:** The 128-patient smoke candidate completed without loading test.
Its selected validation metrics were Ctd `0.690249`, mean horizon C-index
`0.690247`, IBS `0.392552`, IBLL/NBLL `1.092452`, `risk10_std=0.089190` and
127 unique rounded risks; no collapse flag was triggered. The 64-patient
tiny-overfit reduced train survival NLL from `3.248966` to `0.362316` and
static reconstruction MSE from `1.026930` to `0.982480`; final train
`risk10_std=0.278752` and the selected validation epoch was non-collapsed.

These reduced-sample results are implementation diagnostics, not final model
performance. Full 16-candidate tuning and final seeds were not run.

**Outputs:** `outputs/dysurv_static_faithful_72h/smoke/`,
`outputs/dysurv_static_faithful_72h/audit/tiny_overfit/` and
`outputs/dysurv_static_faithful_72h/dysurv_static_faithful_audit_report.md`.

**Decision:** Proceed to full validation-only tuning, inspect curves and
collapse diagnostics, then run final seeds only after accepting the selected
candidate. See DEC-019.

## EXP-019 — Final faithful model prediction export and result inspection

**Date:** 2026-06-15

**Purpose:** Verify and standardize complete validation/test survival-curve
exports for the final faithful Dynamic-DeepHit, temporal DySurv and static
DySurv runs, and document final metrics already present in the output
artifacts.

**Models and selected configs:**

- Dynamic-DeepHit faithful: `dynamic_deephit_faithful_cfg_002`; hyperparameters
  `learning_rate=0.001`, `dropout=0.1`, `weight_decay=0.0001`,
  `batch_size=128`, `alpha_ranking=0.1`, `beta_nll=0.5`, `sigma=0.2`.
- DySurv faithful temporal: `dysurv_faithful_cfg_002`; hyperparameters
  `learning_rate=0.001`, `dropout=0.1`, `weight_decay=0.0001`,
  `batch_size=128`, `kl_warmup_epochs=50`, loss weights `0.70/0.20/0.10`.
- DySurv static faithful: `dysurv_static_faithful_cfg_007`; hyperparameters
  `learning_rate=0.001`, `dropout=0.1`, `weight_decay=0.0001`,
  `batch_size=256`, `kl_warmup_epochs=20`, loss weights `0.80/0.15/0.05`.

**Seeds:** `42`, `123`, `2026`.

**Final aggregate test metrics:**

- Dynamic-DeepHit faithful: Ctd `0.780743 +/- 0.003989`, mean horizon C-index
  `0.787014 +/- 0.005717`, IBS `0.121338 +/- 0.001455`, IBLL/NBLL
  `0.385481 +/- 0.004907`; no collapsed seeds.
- DySurv faithful temporal: Ctd `0.777839 +/- 0.002461`, mean horizon C-index
  `0.776494 +/- 0.008187`, IBS `0.251928 +/- 0.032800`, IBLL/NBLL
  `0.706178 +/- 0.083272`; no collapsed seeds.
- DySurv static faithful: Ctd `0.683475 +/- 0.000998`, mean horizon C-index
  `0.682671 +/- 0.000665`, IBS `0.165585 +/- 0.020709`, IBLL/NBLL
  `0.499855 +/- 0.047991`; no collapsed seeds.

**Per-seed test metrics:**

| Model | Seed | Ctd | Mean horizon C-index | IBS | IBLL/NBLL | Collapse |
|---|---:|---:|---:|---:|---:|---|
| Dynamic-DeepHit faithful | 42 | 0.775156 | 0.779323 | 0.123386 | 0.392381 | false |
| Dynamic-DeepHit faithful | 123 | 0.782861 | 0.788699 | 0.120151 | 0.382679 | false |
| Dynamic-DeepHit faithful | 2026 | 0.784212 | 0.793021 | 0.120477 | 0.381384 | false |
| DySurv faithful temporal | 42 | 0.775757 | 0.777275 | 0.297286 | 0.821111 | false |
| DySurv faithful temporal | 123 | 0.781294 | 0.786108 | 0.237660 | 0.670936 | false |
| DySurv faithful temporal | 2026 | 0.776464 | 0.766100 | 0.220838 | 0.626485 | false |
| DySurv static faithful | 42 | 0.683393 | 0.682379 | 0.189751 | 0.556476 | false |
| DySurv static faithful | 123 | 0.684737 | 0.683591 | 0.167831 | 0.503953 | false |
| DySurv static faithful | 2026 | 0.682296 | 0.682043 | 0.139174 | 0.439136 | false |

**Prediction exports:** Existing complete prediction parquet files were found
for validation and test in all three models and all final seeds. Standardized
derivative files were created under each
`outputs/<pipeline>/final/seed_<seed>/predictions/` directory:
`validation_survival_curves.parquet`, `test_survival_curves.parquet`,
`validation_patient_predictions.csv` and `test_patient_predictions.csv`.
The export audit is stored in `outputs/final_faithful_curve_export_audit.json`.

**Orientation and completeness audit:** All checked files contained 10 survival
columns, `patient_id`, observed relative duration, event indicator and `risk10`.
For every checked split/seed/model, `risk10` matched `1 - S(10)` within
tolerance, patient IDs were unique, and patient order matched across the three
pipelines for the same split and seed. No missing final prediction outputs were
detected.

**Calibration and interpretation:** Dynamic-DeepHit faithful is the best global
model among these final faithful runs because it combines the highest aggregate
test discrimination with clearly superior IBS/IBLL. Temporal DySurv faithful has
good discrimination, close to Dynamic-DeepHit, but poor probability calibration
and substantially worse IBS/IBLL. Static DySurv faithful is an inferior baseline
in discrimination, although its calibration metrics are better than temporal
DySurv and worse than Dynamic-DeepHit.

**Finality warning:** These results should be treated as final only conditional
on the completed audit assumptions: correct risk orientation, validation-only
tuning without test selection, and identical faithful split/patient ordering
across the compared models.
