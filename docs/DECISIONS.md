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

## DEC-010 — Audit grids and PCHazard interpolation for static_72h_pycox

Date: 2026-06-12
Status: accepted
Scope: evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- The first `static_72h_pycox` validation tuning run showed PCHazard with
  validation Antolini Ctd near 0.40 while its mean horizon C-index was near
  0.69.
- The DySurv static MIMIC-IV reference notebook sets `model.sub = 10` before
  calling `PCHazard.predict_surv_df`.
- The previous 72-hour config used a daily grid for both integrated IBS/IBLL and
  horizon C-index.

### Decision
- Keep `pycox.DeepHitSingle` without a manual tail category.
- Set PCHazard `sub=10` before survival prediction and expose it in
  `configs/static_72h_tuning.yaml`.
- Use a per-split 100-point integration grid for IBS/IBLL, bounded by observed
  durations, prediction support and the 10-day horizon.
- Keep daily `horizon_times: [1, ..., 10]` only for horizon C-index.
- Add audit artifacts under `outputs/static_72h/audit/` for discretization cuts,
  evaluation grids, survival-curve sanity checks and DeepHit/PCHazard tail
  diagnostics.

### Reason
- PCHazard is a continuous-time piecewise-constant hazard model in pycox; the
  `sub` interpolation setting affects the survival support used by Antolini Ctd.
- DySurv's notebook uses the same `model.sub = 10` convention, so this is a
  fidelity fix rather than a model redesign.
- Separating integration and horizon grids avoids conflating probabilistic
  integrated metrics with the project's daily discrimination summary.

### Consequences
- Existing PCHazard tuning metrics generated before this fix should be treated
  as stale.
- Existing IBS/IBLL values for `static_72h_pycox` generated on the old daily
  grid should not be mixed with new 100-point-grid values.
- Full validation-only tuning should be rerun before final 3-seed evaluation.

### Related files
- configs/static_72h_tuning.yaml
- scripts/tune_static_72h_models.py
- src/models/static_72h_pycox.py
- src/evaluation/static_72h_metrics.py
- outputs/static_72h/audit/

### Follow-up
- [ ] Rerun full validation-only tuning for all `static_72h_pycox` models after
      this audit fix.
- [ ] Run final 3-seed evaluation only after the new selected hyperparameters
      are written.

## DEC-011 — Dynamic 72h dataset anchored to static_72h_pycox cohort

Date: 2026-06-12
Status: accepted
Scope: data
Owner: technical agent
Related history: not yet consolidated

### Context
- The new dynamic experiment must compare DySurv/Dynamic-DeepHit against the
  `static_72h_pycox` benchmark without changing cohort membership, splits or
  survival targets.
- Available temporal sources are the existing processed MIMIC extraction files
  `timeseries.csv` and `timeserieslab.csv`.

### Decision
- Build `dynamic_72h` directly from
  `data/processed/static_72h/{train,val,test}_static_72h.parquet`.
- Preserve patient ordering, split assignment, `duration_eval_days`,
  `duration_rel_days` and `event_eval` from `static_72h_pycox`.
- Use only temporal measurements with `0 <= offset_minutes < 4320`, binned into
  hours `0..71`.
- For repeated patient-feature-hour measurements, keep the last measurement in
  that hour.
- Select temporal variables using train-patient coverage only, with the current
  threshold `5%`.
- Fit forward-fill fallback medians and robust p05/p95 scaling on train only,
  then apply the fitted preprocessor unchanged to validation and test.
- Save explicit observation masks before imputation; do not backward-fill.

### Reason
- Reusing the static 72h cohort and targets prevents static-vs-dynamic leakage
  or cohort drift.
- Train-only feature selection, imputation and scaling preserve validation/test
  independence.
- A strict first-72h temporal window matches the landmark prediction design and
  avoids using post-prediction information.

### Consequences
- Dynamic model adapters should consume the saved arrays under
  `data/processed/dynamic_72h/`.
- `delta_seq` is not included yet; models requiring time-since-last-observed
  inputs need a separate documented extension.
- The current dataset is a clean input artifact only; DySurv and
  Dynamic-DeepHit training remain pending.

### Related files
- configs/dynamic_72h_data.yaml
- scripts/build_dynamic_72h_data.py
- src/data/dynamic_72h_dataset.py
- tests/test_dynamic_72h_dataset.py
- data/processed/dynamic_72h/
- outputs/dynamic_72h/audit/

### Follow-up
- [ ] Adapt DySurv to `dynamic_72h`.
- [ ] Adapt Dynamic-DeepHit to `dynamic_72h`.
- [ ] Decide whether to add `delta_seq` before dynamic model training.

## DEC-012 — DySurv-compatible dynamic feature subset for first training pass

Date: 2026-06-12
Status: accepted
Scope: data
Owner: technical agent
Related history: not yet consolidated

### Context
- The full `dynamic_72h` build selected 146 temporal variables by train
  coverage.
- The DySurv reference table uses a smaller curated clinical variable set.
- Rebuilding `dynamic_72h` from the large temporal CSVs is slow, while the
  existing arrays already contain imputed/scaled values and masks.

### Decision
- Create a derived dataset under
  `data/processed/dynamic_72h_dysurv_features/` by slicing `X_seq` and `M_seq`
  columns from the already-built `dynamic_72h` arrays.
- Preserve `patient_ids`, `X_static`, durations, events, split membership and
  ordering unchanged.
- Do not refit imputation or scaling for this subset.
- Keep the full `dynamic_72h` dataset unchanged.

### Reason
- This provides a fast first-pass training input closer to the DySurv reference
  variable set without reprocessing the large MIMIC-derived time-series CSV.
- Since only columns are removed after preprocessing, the operation does not
  introduce validation/test leakage.

### Consequences
- The reduced dataset has 76 temporal columns rather than 146.
- Four DySurv table variables are still unavailable in the generated temporal
  feature set: `ALT`, `Bilirubin`, `AST` and `Alkaline Phosphatase`.
- Results from this reduced dataset should be labelled as
  DySurv-compatible/curated-feature first pass, not as the full automatic
  dynamic feature set.

### Related files
- scripts/filter_dynamic_72h_dysurv_features.py
- data/processed/dynamic_72h/
- data/processed/dynamic_72h_dysurv_features/

### Follow-up
- [ ] Use `dynamic_72h_dysurv_features` for the first dynamic model smoke
      training pass.

## DEC-013 — Remove selected chart duplicates from DySurv-compatible subset

Date: 2026-06-12
Status: accepted
Scope: data
Owner: technical agent
Related history: not yet consolidated

### Context
- The first `dynamic_72h_dysurv_features` subset retained 76 temporal columns.
- Several retained columns were chart-derived versions of variables that also
  have lab-derived equivalents or were explicitly requested for removal before
  the first dynamic-model training pass.

### Decision
- Overwrite `data/processed/dynamic_72h_dysurv_features/` in place by removing
  15 additional chart-derived temporal columns from `X_seq` and `M_seq`.
- Preserve patient IDs, static features, durations, events and split ordering
  unchanged.
- Do not refit imputation or scaling.
- Keep the full `data/processed/dynamic_72h/` dataset unchanged.

### Reason
- This further reduces redundancy and dimensionality for a first quick dynamic
  training pass.
- The operation is a pure column subset of already preprocessed arrays, so it
  does not introduce validation/test leakage.

### Consequences
- The current `dynamic_72h_dysurv_features` dataset now has 61 temporal columns.
- The earlier 76-feature version has been overwritten in that folder.
- Any run using `dynamic_72h_dysurv_features` after this decision should be
  labelled as the 61-feature reduced subset.

### Related files
- data/processed/dynamic_72h_dysurv_features/
- docs/EXPERIMENT_LOG.md
- SESSION_NOTES.md

### Follow-up
- [ ] Use the 61-feature `dynamic_72h_dysurv_features` subset for the first
      dynamic model smoke training pass.

## DEC-014 — Isolated dynamic_72h model layer

Date: 2026-06-12
Status: accepted
Scope: model | evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- The 72-hour methodology needs dynamic survival models comparable to
  `static_72h_pycox` on the same cohort, splits, targets and horizon.
- `dynamic_72h_dysurv_features` provides sequence tensors for the first
  training pass.

### Decision
- Add a separate `dynamic_72h` experimental layer instead of modifying static
  pipelines.
- Use the saved `dynamic_72h_dysurv_features` `.npz` splits directly; no new
  split creation and no target recomputation.
- Use `values_plus_mask_plus_static` as the default dynamic input mode,
  concatenating `X_seq`, `M_seq` and repeated `X_static` at every timestep.
- Use daily cuts `[0, 1, ..., 10]` and discrete indices `0..9` for the
  10-day post-72h horizon.
- Implement DySurv as a TFG adaptation of the reference notebook: LSTM encoder,
  latent `mu/logvar`, reparameterization, survival head, decoder reconstruction,
  KL loss and reconstruction loss.
- Implement Dynamic-DeepHit as a TFG adaptation of the reference `ddh` PyTorch
  implementation: recurrent embedding, longitudinal prediction network,
  temporal attention, cause-specific network and PMF loss/ranking loss.
- Keep test metrics disabled during tuning; final 3-seed evaluation remains a
  separate script.

### Reason
- This keeps static and dynamic experiments isolated and reproducible.
- The input construction uses only first-72h information and train-fitted
  preprocessing from the saved dataset.
- A common evaluation layer preserves comparability with static 72h models.

### Consequences
- Tuning outputs are written under `outputs/dynamic_72h/tuning/{model}/`.
- Final seed outputs are planned under `outputs/dynamic_72h/final/{model}/`.
- The Dynamic-DeepHit adaptation uses an internal tail/support category by
  default so survival at 10 days is not forced to zero.
- The current smoke results are implementation checks only, not final model
  evidence.

### Related files
- configs/dynamic_72h_tuning.yaml
- configs/dynamic_72h_final.yaml
- src/models/dynamic_72h/
- src/evaluation/dynamic_72h_metrics.py
- scripts/tune_dynamic_72h_models.py
- scripts/run_final_dynamic_72h_seeds.py
- scripts/evaluate_dynamic_72h_models.py
- tests/test_dynamic_72h_models.py

### Follow-up
- [ ] Run full validation-only dynamic_72h tuning.
- [ ] Review dynamic smoke losses/curves before launching final 3-seed runs.
- [ ] Run final dynamic_72h three-seed evaluation only after tuning is complete.

## DEC-015 — Expanded dynamic_72h tuning grid

Date: 2026-06-12
Status: accepted
Scope: config | tuning
Owner: technical agent
Related history: not yet consolidated

### Context
- The first dynamic_72h smoke config was intentionally small.
- The next validation-only tuning pass needs a broader but still explicit grid
  for DySurv and Dynamic-DeepHit.

### Decision
- Replace the small smoke grid in `configs/dynamic_72h_tuning.yaml` with the
  approved dynamic tuning combinations for DySurv and Dynamic-DeepHit.
- Keep tuning validation-only and keep final test evaluation separated.
- Support DySurv `loss_weights` as a list of dictionaries in YAML, normalizing
  each candidate to the implementation keys `w_surv`, `w_recon` and `w_kl`.
- Make Dynamic-DeepHit `num_durations` configurable in the training code while
  keeping the current approved value at 10.
- Keep Dynamic-DeepHit `include_tail_category: true` in the grid.

### Reason
- The YAML notation now matches the approved hyperparameter plan more closely.
- Normalizing `loss_weights` avoids fragile duplicated keys in the grid.
- Propagating `num_durations` removes an unnecessary hardcoded assumption
  without changing the current 10-day protocol.

### Consequences
- The expanded grid currently contains 384 DySurv candidates and 512
  Dynamic-DeepHit candidates.
- Full dynamic tuning is now a substantial GPU run and should be launched
  deliberately, ideally with staged dry-runs or `--max-runs` checks first.
- No test metrics are produced by the tuning script.

### Related files
- configs/dynamic_72h_tuning.yaml
- scripts/tune_dynamic_72h_models.py
- src/models/dynamic_72h/train.py

### Follow-up
- [ ] Run full validation-only dynamic_72h tuning with the expanded grid.
- [ ] Consider staged execution if runtime is too high.

## DEC-016 — Isolated DySurv-faithful 72h pipeline

Date: 2026-06-14
Status: accepted
Scope: data | model | training | evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- The previous DySurv adaptation preserved the batch dimension and used a
  correct LogisticHazard NLL, but final predictions were nearly marginal and
  showed posterior/latent collapse.
- Its decoder reconstructed 61 clinical values, 61 observation masks and 28
  repeated static variables, while the architecture was substantially smaller
  than the reference notebook.

### Decision
- Create `dysurv_faithful_72h` as a new pipeline without modifying or
  overwriting the previous dynamic pipeline.
- Derive a new dataset using within-patient forward fill, backward fill and a
  residual median fitted only on train.
- Standardize static variables with train-only statistics.
- Use temporal clinical variables plus repeated standardized static variables
  as the primary encoder input, but never concatenate `M_seq` as input.
- Reconstruct only temporal clinical variables with a recurrent decoder.
- Preserve the reference model's 72-step LSTM, latent dimension 20 and
  `[294, 490, 294]` encoder/survival MLP capacity.
- Do not condition the decoder on observed duration, because that would expose
  outcome information unavailable at prediction time.
- Add KL warm-up, checkpoints, full patient predictions and collapse
  diagnostics as mandatory run artifacts.
- During epoch and hyperparameter selection, prefer the best non-collapsed
  validation candidate when one exists; record the pure metric maximum
  separately.

### Reason
- This isolates the effect of fidelity, imputation and reconstruction target
  from the earlier implementation problems.
- It preserves the 72-hour landmark, target, horizon, splits and evaluation
  protocol while preventing target and post-landmark leakage.
- Collapse-aware selection prevents a slightly higher Ctd from silently
  selecting nearly constant survival curves.

### Consequences
- New data are stored under `data/processed/dysurv_faithful_72h/`.
- New results are stored under `outputs/dysurv_faithful_72h/`.
- Smoke runs using `--sample-size` are isolated under `smoke/` and cannot be
  consumed by the final-seed script.
- The initial tuning grid contains 16 training/loss configurations and no
  architecture search.
- Full tuning and final three-seed evaluation remain pending.

### Related files
- configs/dysurv_faithful_72h.yaml
- src/data/dysurv_faithful_72h_dataset.py
- src/models/dynamic_72h/dysurv_faithful.py
- src/models/dynamic_72h/train_dysurv_faithful.py
- scripts/prepare_dysurv_faithful_72h_dataset.py
- scripts/tune_dysurv_faithful_72h.py
- scripts/run_final_dysurv_faithful_72h_seeds.py
- scripts/audit_dysurv_faithful_72h.py
- tests/test_dysurv_faithful_72h.py

### Follow-up
- [ ] Run all 16 validation-only faithful tuning candidates on GPU.
- [ ] Review selected curves and collapse diagnostics before final seeds.
- [ ] Run final seeds 42, 123 and 2026 only after a non-collapsed validation
      selection is accepted.

## DEC-017 — Extended per-epoch DySurv latent and calibration diagnostics

Date: 2026-06-14
Status: accepted
Scope: training diagnostics | evaluation
Owner: technical agent

### Decision

- Keep model architecture, losses and selection behavior unchanged while
  extending `metrics/epoch_metrics.csv` with per-epoch calibration and latent
  diagnostics.
- Record mean, standard deviation, minimum, maximum and range of 10-day risk.
- Record total KL and mean/minimum/maximum KL per latent dimension, plus one
  column `kl_dim_XX` for every latent dimension.
- Define an active latent unit as a dimension satisfying
  `Var_x(mu_j) > 0.01` across patients and report the active-unit count.
- Continue recording Ctd, IBS, IBLL/NBLL, reconstruction loss, survival loss
  and existing collapse diagnostics.

### Reason

- Ctd can remain high when risk differences are numerically tiny, while
  calibration and individualization deteriorate.
- Per-dimension KL and active units make posterior contraction visible before
  relying on a binary collapse flag.

### Consequences

- The new metrics are diagnostic only and do not alter optimization,
  checkpoint selection or collapse thresholds.
- Existing output files are not backfilled; the new columns appear in runs
  produced after this change.
## DEC-018 — Isolated Dynamic-DeepHit-faithful 72h pipeline

**Date:** 2026-06-15

Dynamic-DeepHit will be evaluated through a new isolated pipeline under
`outputs/dynamic_deephit_faithful_72h/`, reusing the exact prepared
train/validation/test arrays and patient order from `dysurv_faithful_72h`.
The previous dynamic pipeline and its outputs remain unchanged.

The adaptation preserves the reference model's recurrent embedding,
next-step longitudinal objective, temporal attention, cause-specific PMF NLL
and pairwise ranking loss. It uses the same clean input convention as faithful
DySurv: temporal clinical values plus repeated standardized static covariates,
without observation masks as input channels. The longitudinal auxiliary target
contains temporal clinical variables only, so repeated static covariates do not
dominate its MSE. An explicit tail category represents survival beyond day 10.

Tuning is validation-only and selects by validation Antolini Ctd with IBLL as
tiebreaker, while reporting collapse diagnostics and a non-collapsed
alternative when needed. Test predictions are permitted only in the final
three-seed stage using seeds 42, 123 and 2026.

Related files: `configs/dynamic_deephit_faithful_72h.yaml`,
`src/models/dynamic_72h/dynamic_deephit_faithful.py`,
`scripts/tune_dynamic_deephit_faithful_72h.py` and EXP-017.

## DEC-019 — Isolated static-only DySurv faithful 72h pipeline

**Date:** 2026-06-15

DySurv static-only will be evaluated through an isolated MLP-VAE pipeline
under `outputs/dysurv_static_faithful_72h/`. It reads `X_static`, patient IDs
and unchanged survival targets directly from the exact split files prepared
for `dysurv_faithful_72h`; it creates no cohort or split and never uses
`X_seq` or `M_seq` as inputs.

The model preserves the common structure found in the final `DySurv` sections
of the GBSG, METABRIC, SUPPORT, NWTCO, SAC3 and SAC_ADMIN notebooks:
`F -> 3F -> 5F -> 3F`, latent dimension 20, static reconstruction and a
LogisticHazard head. The decoder defaults to no intermediate activation,
matching the notebook code. Evaluation uses deterministic `mu`, KL is averaged
per patient, and explicit loss weights with KL warm-up replace the notebooks'
internally inconsistent loss call and batch-dependent KL scaling.

Tuning is validation-only with collapse-aware selection. Test is evaluated
only after a full validation selection, using seeds 42, 123 and 2026. See
EXP-018 and `configs/dysurv_static_faithful_72h.yaml`.

## DEC-021 — Add time since hospital admission as a static covariate

Date: 2026-06-19
Status: accepted
Scope: data
Owner: technical agent
Related history: not yet consolidated

### Context
- The static 72h pipeline previously included ICU admission hour of day but did
  not include elapsed time from hospital admission to ICU admission.
- MIMIC-IV provides both hospital `admittime` and ICU `intime`, so this covariate
  can be added without regenerating heavy time-series artifacts.

### Decision
- Create `flat_features_with_time_since_admission.csv` by appending
  `time_since_admission_hours` to the existing flat features.
- Point static dataset configs to the enriched flat-feature file and preprocess
  the new variable as a numeric static feature.
- Preserve the raw signed value when ICU `intime` is earlier than hospital
  `admittime`; do not clamp negative values.

### Reason
- The covariate is available at the 72h landmark and captures pre-ICU hospital
  time without using outcomes or post-landmark measurements.
- A separate enrichment script avoids rerunning expensive time-series
  extraction and preserves the existing flat-feature row order.
- Preserving signed values keeps the generated file faithful to raw MIMIC-IV
  timestamps and avoids silently inventing corrected admission times.

### Consequences
- Static datasets must be rebuilt before training if the new covariate should be
  included.
- Previous static outputs are no longer directly comparable unless the older
  flat-feature config is restored.
- Some rows can contain negative values because of timestamp ordering in the
  raw tables; these should be inspected but are not treated as missing data.

### Related files
- scripts/add_time_since_admission_to_flat_features.py
- configs/static_72h_data.yaml
- configs/static_data.yaml
- docs/REPRODUCIBILITY.md

### Follow-up
- [ ] Rebuild static datasets before rerunning static model tuning with the new
      covariate.

## DEC-022 — CLI-driven parametrizable landmark pipeline

Date: 2026-06-19
Status: accepted
Scope: data | training | evaluation | infrastructure
Owner: technical agent
Related history: not yet consolidated

### Context
- The project needs to run the current 72h static/dynamic faithful design at
  24h, 48h and 72h without duplicating three independent pipelines.
- The recent static covariate addition in DEC-021 must remain the active data
  definition.

### Decision
- Add a CLI-driven landmark layer with `--landmark-hours {24,48,72}`.
- Keep existing 72h scripts/configs available and compatible.
- Resolve landmark-specific data under `data/processed/landmark_<s>h/` and
  outputs under `outputs/landmark_<s>h/`.
- Derive DySurv faithful, Dynamic-DeepHit faithful and static-only DySurv
  faithful inputs from one common prepared faithful dataset per landmark.
- Store resolved configs as `config_used.yaml` at the landmark/family output
  root and at per-candidate run level where tuning creates runs.

### Reason
- The CLI is the operational source of truth for the landmark while preserving
  existing base configs as reusable defaults.
- A single dynamic dataset per landmark prevents patient/split/target drift
  between dynamic models.
- Keeping 72h scripts intact reduces risk and allows equivalence checks before
  running 24h and 48h.

### Consequences
- Real 72h equivalence must still be demonstrated by rebuilding the new
  `landmark_72h` datasets and comparing IDs, targets, columns, feature names and
  shapes against the current 72h artifacts before launching 24h/48h runs.
- New landmark results will not be directly comparable with older outputs
  generated before DEC-021 unless the previous flat-feature config is restored.

### Related files
- src/utils/landmark.py
- scripts/build_landmark_static_data.py
- scripts/build_landmark_dynamic_data.py
- scripts/filter_landmark_dysurv_features.py
- scripts/prepare_landmark_faithful_dataset.py
- scripts/tune_landmark_static_models.py
- scripts/tune_landmark_dysurv_faithful.py
- scripts/tune_landmark_dynamic_deephit_faithful.py
- scripts/tune_landmark_dysurv_static_faithful.py
- tests/test_landmark_pipeline.py

### Follow-up
- [ ] Rebuild and compare `landmark_72h` against the existing 72h pipeline
      before running 24h/48h.

## DEC-023 — Common DySurv loss-weight grid for temporal and static faithful models

Date: 2026-06-19
Status: accepted
Scope: training | evaluation
Owner: technical agent
Related history: not yet consolidated

### Context
- The faithful temporal DySurv and static-only DySurv tuning configs are being
  reduced to compact 16-combination grids for reuse at 24h, 48h and 72h.
- Previous tuning showed that `w_surv=0.70, w_recon=0.20, w_kl=0.10` and
  `w_surv=0.80, w_recon=0.15, w_kl=0.05` are the most defensible shared
  settings.
- Equal-weight DySurv settings produced suspicious or collapsed dynamic
  candidates and should not be retained in the compact grid.

### Decision
- Use the same four `loss_weights` settings for temporal DySurv faithful and
  static DySurv faithful:
  - `w_surv=0.70`, `w_recon=0.20`, `w_kl=0.10`
  - `w_surv=0.75`, `w_recon=0.20`, `w_kl=0.05`
  - `w_surv=0.80`, `w_recon=0.15`, `w_kl=0.05`
  - `w_surv=0.85`, `w_recon=0.10`, `w_kl=0.05`
- Keep `kl_warmup_epochs=[50]` in both configs.
- Keep each grid at exactly 16 combinations.

### Reason
- A common loss-weight grid makes the static-vs-temporal DySurv comparison
  cleaner: differences are less likely to come from different loss balances.
- The selected grid keeps the two empirically strongest shared regions and
  adds two nearby high-survival, low-KL variants.
- The grid avoids the equal-weight setting associated with low risk dispersion
  and suspected collapse in temporal DySurv tuning.

### Consequences
- Temporal DySurv no longer tests the exploratory reconstruction-heavy weights
  `0.50/0.30/0.20` and `0.55/0.40/0.05` in the compact landmark grid.
- Static and temporal DySurv now differ in inputs and architecture, but not in
  the candidate loss-weight balances.

### Related files
- configs/dysurv_faithful_72h.yaml
- configs/dysurv_static_faithful_72h.yaml

### Follow-up
- [ ] Use the common loss-weight grid when running the 24h/48h landmark tuning
      experiments.
