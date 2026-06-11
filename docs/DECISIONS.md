# Decisions

## Purpose

This file records important technical, methodological and documentation
governance decisions that affect the interpretation, reproducibility or future
maintenance of the project.

Use this file together with:

- [PROJECT_HISTORY.md](PROJECT_HISTORY.md) for consolidated chronology;
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for executed runs;
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands and environment;
- [TODO.md](TODO.md) for open follow-up work.

## Usage Instructions

- Append new decisions; do not rewrite or delete previous decisions.
- Use one numbered entry per decision.
- Record decisions when a change affects data splits, model assumptions,
  evaluation, reproducibility, repository organization or thesis interpretation.
- If a decision changes or deprecates an older one, create a new entry with
  `Status: changed` or `Status: deprecated` and reference the earlier decision.
- Link related configs, scripts, notebooks, outputs or docs wherever possible.
- The Project Manager/Historian may reorganize this file for consistency, but
  should preserve the original decision content.

## Decision Template

```md
## DEC-000 — Decision title

Date: YYYY-MM-DD
Status: proposed | accepted | changed | deprecated
Scope: data | model | evaluation | infrastructure | writing | documentation
Owner: agent or role responsible
Related history: PROJECT_HISTORY.md section or "not yet consolidated"

### Context
- What problem, constraint or ambiguity required a decision?

### Decision
- What was decided?

### Reason
- Why is this the preferred choice?

### Consequences
- What future work, limitations or risks follow from this decision?

### Related files
- path/to/file

### Follow-up
- [ ] Optional task to add to TODO.md
```

## Example

```md
## DEC-001 — Patient-level split before landmark generation

Date: 2026-06-07
Status: accepted
Scope: data
Owner: technical agent
Related history: PROJECT_HISTORY.md sections 4 and 14

### Context
- The dynamic dataset creates multiple landmark examples per patient.

### Decision
- Train, validation and test splits must be created at patient or stay level before generating landmarks.

### Reason
- This avoids leakage between train and test when the same patient contributes several temporal examples.

### Consequences
- All dynamic preprocessing scripts must receive precomputed splits or create them before landmark expansion.

### Related files
- src/data/dynamic_dataset.py
- configs/data_preprocessing.yaml

### Follow-up
- [ ] Implement and validate the dynamic landmark dataset builder.
```

## DEC-001 — Common fixed-grid evaluation for static survival models

Date: 2026-06-07
Status: accepted
Scope: evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- DeepHit and PCHazard produce full survival curves, but their previous scalar
  Harrell C-index used final cumulative risk, `1 - S(t_final | x)`.
- This final-risk summary can be misleading for curve-based models because risk
  curves may saturate, cross or behave differently at short and medium horizons.
- IBS and IBLL/NBLL previously relied on the full set of observed times through
  `pycox`, which can create very large intermediate matrices and memory errors.

### Decision
- Use a common grid for static survival evaluation:
  `[1, 2, 3, 4, 5, 6, 7, 8, 9]` days.
- Use this grid as both `evaluation_time_grid` for IBS/IBLL/NBLL and
  `horizon_times` for horizon C-index.
- Keep natural Harrell C-index as primary scalar discrimination only for
  proportional-risk models:
  CoxPH partial hazard and DeepSurv neural log-risk.
- For DeepHit and PCHazard, rename final cumulative-risk C-index to
  `harrell_c_index_final_risk` and treat it as secondary.
- For curve-based models, primary discrimination should be based on
  `ctd_antolini`, `horizon_c_index` and `mean_horizon_c_index`.

### Reason
- Horizon-specific risk, `1 - S(h | x)`, better matches the temporal output of
  DeepHit and PCHazard than a single final-risk summary.
- A fixed grid makes CoxPH, DeepSurv, PCHazard and DeepHit comparable on the
  same days, time unit and censoring/IPCW convention.
- Restricting IBS/IBLL/NBLL to a small fixed grid avoids memory-heavy evaluation
  over all unique observed event times.

### Consequences
- Metrics JSONs distinguish `harrell_c_index`,
  `harrell_c_index_final_risk`, `ctd_antolini`, `horizon_c_index`,
  `mean_horizon_c_index`, `ibs`, `ibll`, `nbll`, `evaluation_time_grid` and
  `horizon_times`.
- IBS/IBLL/NBLL are computed on validation and test by default, while train
  curve metrics are left as NaN to avoid unnecessary memory use.
- Static model metrics should be regenerated after this decision before final
  thesis tables are interpreted.

### Related files
- configs/coxph.yaml
- configs/deepsurv.yaml
- configs/pchazard.yaml
- configs/deephit.yaml
- src/evaluation/metrics.py
- src/evaluation/time_dependent_survival.py
- src/models/coxph_tfg.py
- src/models/deepsurv_tfg.py
- src/models/pchazard_tfg.py
- src/models/deephit_tfg.py

### Follow-up
- [ ] Regenerate `outputs/metrics/` with the fixed-grid protocol.

## DEC-002 — Store model metrics under model-specific subfolders

Date: 2026-06-07
Status: accepted
Scope: evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- Static model runs write several JSON and CSV metric artifacts.
- Writing every model artifact directly under `outputs/metrics/` makes it easy
  to mix old and new outputs while iterating on the evaluation protocol.

### Decision
- Keep `outputs/metrics/` as the root metrics directory, but write each model's
  metric artifacts under one subfolder per model.
- Keep existing filenames unchanged inside each model folder.

### Reason
- This keeps the artifact names stable while making it clearer which files
  belong to each model.

### Consequences
- Example paths now include `outputs/metrics/coxph/coxph_metrics.json` and
  `outputs/metrics/deephit/deephit_metrics.json`.
- Consolidation config must read metrics from model-specific subfolders.
- Existing root-level metric files from older runs are not deleted
  automatically; they should be ignored or manually cleaned after confirming
  the new run.

### Related files
- configs/static_evaluation.yaml
- configs/pchazard.yaml
- configs/deephit.yaml
- src/models/static_common.py
- src/models/kaplan_meier_tfg.py
- src/models/coxph_tfg.py
- src/models/deepsurv_tfg.py
- src/models/pchazard_tfg.py
- src/models/deephit_tfg.py
- docs/REPRODUCIBILITY.md

### Follow-up
- [ ] After the next pipeline run, verify metric artifacts exist under each
      model subfolder.

## DEC-003 — Permanent Project Manager/Historian ownership

Date: 2026-06-07
Status: accepted
Scope: documentation
Owner: Project Manager/Historian
Related history: PROJECT_HISTORY.md section 13

### Context
- The repository now has several documentation channels: session notes,
  project history, decisions, experiment log, reproducibility instructions and
  TODO tracking.
- Technical sessions may append working records, but stable chronology and
  cross-document consistency need one owner.

### Decision
- Treat the Project Manager/Historian role as the permanent owner of
  `docs/PROJECT_HISTORY.md` consolidation and periodic documentation cleanup.
- The role should not modify model code, training scripts, evaluation scripts,
  preprocessing code, data, outputs or checkpoints unless explicitly instructed.

### Reason
- A single documentation owner reduces duplicated history, stale TODO items and
  missing links between decisions, experiments and thesis claims.

### Consequences
- Future important sessions should end with session notes that can be
  consolidated by the Project Manager/Historian.
- Technical agents should continue to update append-only operational documents,
  while the Project Manager/Historian maintains the clean historical narrative.

### Related files
- AGENTS.md
- docs/README.md
- docs/PROJECT_HISTORY.md
- docs/TODO.md
- SESSION_NOTES.md

### Follow-up
- [ ] Periodically consolidate important `SESSION_NOTES.md` entries into
      `docs/PROJECT_HISTORY.md`.

## DEC-004 — DeepHit review after first static benchmark

Date: 2026-06-08
Status: accepted
Scope: model
Owner: Project Manager/Historian
Related history: PROJECT_HISTORY.md sections 11, 12 and 13

### Context
- The first static benchmark identifies DeepSurv and PCHazard as the strongest
  current static references.
- DeepHit shows final-risk C-index approximately 0.49, Antolini Ctd
  approximately 0.75, mean horizon C-index approximately 0.73, IBS
  approximately 0.40 and IBLL approximately 1.02.
- The strongest static comparators, especially PCHazard and the current
  DeepSurv/PCHazard benchmark context, are around IBS approximately 0.11 and
  IBLL approximately 0.35 where curve calibration metrics are available.
- This creates a split interpretation: DeepHit's time-dependent ranking metrics
  are informative, but its curve-quality metrics are much worse than the
  strongest static baselines.

### Findings
- DeepHit appears to rank patients reasonably well under time-dependent
  discrimination metrics.
- DeepHit appears poorly calibrated.
- PMF sums correctly.
- Survival and CIF monotonicity checks passed.
- Mean PMF allocates excessive mass to early and final bins.
- `S(10) = 0` for all patients under the current 10-day support.
- The current design forces all event probability inside the 10-day horizon.
- The current ranking loss differs from the original DeepHit paper's
  event-time-specific pairwise comparison.
- Calibration loss is disabled with `gamma = 0`.

### Decision
- Do not start DeepHit hyperparameter tuning yet.
- Treat the current DeepHit result as an implementation and calibration review
  target before using it for final tuning or thesis-level claims.

### Reason
- Tuning before resolving support, censoring, ranking-loss and calibration
  questions could optimize around a flawed objective rather than improving the
  intended DeepHit replication.
- The valid PMF and monotone curves suggest the model is numerically coherent,
  so the next step should be targeted review rather than broad parameter search.

### Consequences
- DeepHit should not be presented as a competitive final static model until it
  is re-evaluated after the review tasks.
- Current static conclusions should emphasize DeepSurv and PCHazard as the
  strongest baselines while marking DeepHit as under audit.

### Related files
- docs/PROJECT_HISTORY.md
- docs/TODO.md
- SESSION_NOTES.md
- configs/deephit.yaml
- src/models/deephit_tfg.py
- src/evaluation/deephit_time_dependent.py

### Planned actions
- [ ] Investigate explicit beyond-horizon / tail category.
- [ ] Review event and censoring encoding.
- [ ] Review ranking loss implementation against the original paper.
- [ ] Run calibration diagnostics.
- [ ] Re-evaluate DeepHit before tuning.

## DEC-005 — DeepHit support and ranking-loss correction before tuning

Date: 2026-06-08
Status: accepted
Scope: model
Owner: technical agent
Related history: not yet consolidated

### Context
- The current DeepHit configuration uses `num_Category: 10` and
  `max_horizon_days: 10`.
- The adapted PyTorch network applies a softmax over
  `num_Event * num_Category`; with one event, all probability mass is allocated
  inside the 10 configured bins.
- Survival is computed as `1 - cumulative_event_probability`, so survival at
  the final configured bin is forced to zero.
- Censored stays capped at the 10-day horizon have no bin after their censoring
  time under the current mask logic, so their censoring likelihood selects an
  empty tail.
- The current ranking loss compares each subject's cumulative risk at its own
  discretized time, while the original DeepHit ranking loss compares pairwise
  risks at the event subject's time.

### Decision
- Do not tune DeepHit until the support and ranking-loss issues are corrected
  and tested.
- Prefer introducing an explicit beyond-horizon tail/output category, or an
  equivalent extended support, before retraining.
- Correct the ranking loss to the original event-time-conditioned pairwise
  formulation before interpreting DeepHit calibration or curve metrics.
- Implement the correction as an internal tail duration category, not as a
  competing event or as part of the fixed-grid horizon metrics.

### Reason
- The current support makes `P(T <= horizon) = 1` by construction for the single
  event setting, which is inconsistent with censored patients who survive past
  the prediction horizon.
- An empty censoring mask at the final bin creates a training signal that cannot
  be fixed by hyperparameter tuning.
- The ranking objective should preserve the original DeepHit comparison logic
  before performance differences are attributed to the model.

### Consequences
- DeepHit metrics generated before these corrections should remain marked as
  audit results, not final model results.
- The next DeepHit implementation pass should include focused tests for tail
  mass, censoring masks, survival reconstruction and ranking-loss pair logic.
- Static benchmark metrics must be regenerated after the correction.
- `num_Category` remains the number of evaluated event-time bins, while
  `include_tail_category: true` adds one internal output bin beyond the
  10-day horizon.
- The fixed-grid evaluation protocol remains unchanged.

### Related files
- configs/deephit.yaml
- src/models/deephit_tfg.py
- src/models_references/DeepHit/class_DeepHit.py
- src/models_references/DeepHit/import_data.py
- tests/test_static_pipeline.py

### Follow-up
- [x] Add an explicit tail/support correction for DeepHit.
- [x] Add DeepHit mask tests covering censored observations at the horizon.
- [x] Add a small ranking-loss test matching the reference pairwise logic.
- [ ] Re-run DeepHit and static metric consolidation after correction.

## DEC-006 — Corrected DeepHit metrics after support and ranking fixes

Date: 2026-06-08
Status: accepted
Scope: model
Owner: technical agent
Related history: not yet consolidated

### Context
- `EXP-004` reran DeepHit after implementing the approved internal tail
  category, censored-at-horizon likelihood repair, nonzero final survival
  reconstruction, event-time-conditioned ranking loss and likelihood
  broadcasting fix.
- Previous DeepHit audit metrics showed test final-risk C-index approximately
  0.4879, Ctd approximately 0.7509, mean horizon C-index approximately 0.73,
  IBS approximately 0.4044 and IBLL/NBLL approximately 1.0431.

### Decision
- Treat the corrected DeepHit implementation as having resolved a major
  structural calibration/probability-support defect.
- DeepHit is no longer blocked by the original tail-support and ranking-loss
  implementation issues.
- Keep final-risk C-index secondary; continue reporting Ctd, horizon C-index,
  IBS and IBLL/NBLL as the main curve-model view.
- Allow the next phase to move to diagnostics and cautious hyperparameter
  tuning, but only after calibration plots, survival-curve comparison and a
  small synthetic overfit test are completed.

### Reason
- The corrected run achieved test IBS 0.1107 and IBLL/NBLL 0.3531, improving
  sharply from the audit values near 0.4044 and 1.0431.
- Test Ctd remained strong at 0.7695, and mean horizon C-index was 0.7505.
- `S(10)` is now nonzero for all generated predictions and equals the learned
  beyond-horizon tail probability.

### Consequences
- The old DeepHit run should be cited only as an audit/failure mode showing the
  effect of missing tail support.
- Corrected DeepHit can be considered a viable static curve-producing baseline
  candidate, pending diagnostic plots and comparison against PCHazard.
- Full static comparison artifacts should be regenerated or consolidated before
  final thesis tables are interpreted.

### Related files
- docs/EXPERIMENT_LOG.md
- docs/TODO.md
- SESSION_NOTES.md
- configs/deephit.yaml
- outputs/metrics/deephit/deephit_metrics.json
- outputs/metrics/deephit/deephit_weighted_c_index_by_horizon.csv
- outputs/metrics/deephit/deephit_antolini_ctd.csv
- outputs/predictions/deephit_predictions.parquet

### Follow-up
- [ ] Generate corrected DeepHit calibration plots.
- [ ] Compare corrected DeepHit survival curves against PCHazard survival
      curves.
- [ ] Run a small synthetic DeepHit overfit test.
- [ ] Tune DeepHit hyperparameters after diagnostics.

## DEC-007 — Validation-only static tuning and three-seed final runs

Date: 2026-06-08
Status: accepted
Scope: evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- The corrected DeepHit implementation is now competitive enough to enter the
  static model tuning phase together with CoxPH, DeepSurv and PCHazard.
- Hyperparameters must be selected without using test metrics.
- Final static estimates should be less seed-dependent for neural models.

### Decision
- Add a dedicated static tuning config and scripts for validation-only
  hyperparameter selection.
- Select tuning configurations by validation `ctd_antolini`, using validation
  `ibll`/`nbll` as the tie-breaker and logging validation IBS plus mean horizon
  C-index.
- Keep the fixed evaluation protocol unchanged:
  horizon/evaluation grids `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- During tuning, evaluate only train and validation splits and do not load or
  score the test split.
- Run final static models only after tuning, using the selected validation
  hyperparameters with exactly seeds `42`, `123` and `2026`.

### Reason
- This separates model-selection evidence from final test reporting and avoids
  accidental test leakage.
- Three final seeds provide a lightweight reproducibility check without turning
  tuning into a large compute campaign.
- Keeping output folders separate preserves previous fixed-baseline metrics.

### Consequences
- Tuning outputs go under `outputs/tuning/{model}/`.
- Final static seed outputs go under
  `outputs/final_static/{model}/seed_{seed}/`.
- Config snapshots are saved with real tuning/final runs.
- Large model/checkpoint artifacts are disabled by default unless explicitly
  configured.

### Related files
- configs/static_tuning.yaml
- scripts/tune_static_models.py
- scripts/run_final_static_seeds.py
- src/models/static_common.py
- docs/REPRODUCIBILITY.md

### Follow-up
- [ ] Run validation-only static tuning.
- [ ] Run final static three-seed evaluation after tuning selection.

## DEC-008 — Expand CoxPH ridge grid after smoke-test regression

Date: 2026-06-08
Status: accepted
Scope: evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- A CoxPH final-seed smoke test selected only `penalizer=0.01` and produced
  much worse metrics than the previous static CoxPH benchmark.
- The smoke run also produced a lifelines convergence warning and unstable
  coefficients, especially for `height` and `nullheight`.
- A diagnostic run through the new final-static pipeline using the old fixed
  CoxPH setting `penalizer=0.1` reproduced the previous benchmark metrics.

### Decision
- Keep the CoxPH training and evaluation logic unchanged.
- Expand the CoxPH tuning grid to
  `penalizer: [0.0, 0.001, 0.01, 0.1]` and `l1_ratio: [0.0]`.
- Do not treat the partial smoke result with only `penalizer=0.01` as a valid
  selected CoxPH configuration.

### Reason
- The regression is explained by an unstable weak-penalty CoxPH candidate and
  incomplete smoke coverage, not by a split, preprocessing or evaluation-grid
  mismatch.
- Including the old benchmark setting in the tuning grid lets validation
  selection recover the stable CoxPH baseline before Lightning AI runs.

### Consequences
- CoxPH validation-only tuning must be rerun before final static seed runs.
- Existing `outputs/tuning/coxph/best_hyperparameters.json` from the partial
  smoke run should not be reused for Lightning AI final evaluation.

### Related files
- configs/static_tuning.yaml
- configs/coxph.yaml
- outputs/diagnostics/coxph/seed_42/metrics/coxph/coxph_metrics.json

### Follow-up
- [ ] Re-run CoxPH validation-only tuning with the expanded grid.

## DEC-009 — Separate 72-hour static pycox benchmark pipeline

Date: 2026-06-11
Status: accepted
Scope: data | model | evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- The main TFG methodology has been reformulated around a fixed prediction
  time at 72 hours after ICU admission.
- The previous static pipeline uses the full static cohort and targets measured
  from admission, so its results are not directly comparable with a dynamic
  72-hour experiment.
- The DySurv static MIMIC-IV notebook uses library models and pycox evaluation,
  but it also drops long survivors and uses a weak/non-reproducible split.

### Decision
- Preserve the existing static pipeline unchanged.
- Add a new isolated experimental layer named `static_72h_pycox`.
- Build a new static cohort with inclusion rule `Y_i > 72h`.
- Define targets relative to the 72-hour prediction time:
  `duration_rel_days = Y_i - 72h`, capped at 10 days for evaluation.
- Treat patients without event within 10 days after hour 72 as censored at the
  10-day horizon; do not remove long survivors.
- Use library implementations where possible: lifelines Kaplan-Meier/CoxPH and
  pycox CoxPH, LogisticHazard, PCHazard and DeepHitSingle.
- Select hyperparameters only on validation metrics, then run final evaluation
  with exactly seeds `42`, `123` and `2026`.

### Reason
- This matches the new prospective question: among patients still observable at
  72 hours, predict risk over the next 10 days.
- Keeping the new code and outputs isolated prevents overwriting the previous
  audited static benchmark.
- Using pycox/lifelines improves reproducibility and reduces risk from custom
  survival-model implementations.

### Consequences
- New configs use the `static_72h_` prefix.
- New outputs are placed under `outputs/static_72h/`.
- Final static-vs-dynamic claims should use this 72-hour cohort once dynamic
  models are implemented on the same split and target definition.
- Previous static results remain useful only as historical/comparison material,
  not as the primary comparator for the new 72-hour dynamic experiment.

### Related files
- TFG/Nueva_version_experimento.md
- configs/static_72h_data.yaml
- configs/static_72h_tuning.yaml
- configs/static_72h_evaluation.yaml
- scripts/build_static_72h_data.py
- scripts/tune_static_72h_models.py
- scripts/run_final_static_72h_seeds.py
- scripts/evaluate_static_72h_models.py
- src/data/static_72h_dataset.py
- src/models/static_72h_pycox.py
- src/evaluation/static_72h_metrics.py

### Follow-up
- [ ] Run the full `static_72h_pycox` data build on real MIMIC-derived inputs.
- [ ] Run validation-only tuning for the 72-hour static models.
- [ ] Run final 3-seed evaluation for the selected 72-hour static models.
- [ ] Implement dynamic models on the same 72-hour cohort and split.
