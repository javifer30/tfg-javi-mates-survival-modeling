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
