## 2026-06-07 — Documentation Governance Initialization

### Purpose

Initialize a coherent documentation governance system for the repository without
modifying model code, configs, data or outputs. The system separates chronology,
decisions, experiments, reproducibility instructions, active tasks and session
handoff notes.

### Files Introduced Or Formalized

- `docs/PROJECT_HISTORY.md`: consolidated project history, maintained only by
  the Project Manager/Historian.
- `docs/DECISIONS.md`: append-only record of technical and methodological
  decisions.
- `docs/EXPERIMENT_LOG.md`: append-only record of executed, running, failed or
  explicitly planned experiments.
- `docs/REPRODUCIBILITY.md`: canonical setup, data preparation, training and
  evaluation instructions.
- `docs/TODO.md`: prioritized active project tasks.
- `docs/README.md`: documentation ownership and cross-linking protocol.
- `SESSION_NOTES.md`: session-level handoff notes and governance health checks.

### Ownership Model

- Technical agents append decisions, experiment entries and TODO updates related
  to their work.
- Technical agents update reproducibility documentation only when commands,
  dependencies, configs or pipeline steps change.
- The Project Manager/Historian owns `docs/PROJECT_HISTORY.md` consolidation and
  periodic documentation cleanup.
- Existing documentation should be preserved; unresolved tasks should not be
  deleted without explanation.

### Current Documentation Status

- `docs/PROJECT_HISTORY.md` contains a strong consolidated narrative of the
  static pipeline, final model results and remaining dynamic-model gap.
- `docs/DECISIONS.md` and `docs/EXPERIMENT_LOG.md` needed complete templates and
  fixed examples.
- `docs/REPRODUCIBILITY.md` was empty and needed canonical static pipeline
  commands.
- `docs/TODO.md` contained placeholders and needed project-specific tasks.
- Root `README.md` appears to lag behind the current script/config names and
  should be refreshed separately.

### Audit Summary

- Missing sections: reproducibility instructions, TODO governance rules,
  project-history ownership preface, decision and experiment usage instructions.
- Broken templates: `docs/DECISIONS.md` and `docs/EXPERIMENT_LOG.md` had
  malformed Markdown code fences.
- Duplicated information: static model results are summarized in project
  history, but experiment-level entries are not yet backfilled.
- Inconsistencies: root `README.md` refers to older commands such as
  `scripts/train_static_pipeline.py`, `scripts/evaluate.py`,
  `scripts/run_mimic_pipeline.py` and `configs/train.yaml`; current files use
  `scripts/run_static_pipeline.py`, `scripts/train_static_model.py`,
  `scripts/evaluate_static_model.py` and `configs/static_pipeline.yaml`.
- Governance gaps: no initialized session note, no explicit cross-linking rules,
  and no clear rule that only the Project Manager/Historian consolidates project
  history.

### Next Documentation Actions

- Backfill final static experiments into `docs/EXPERIMENT_LOG.md`.
- Add formal decision entries for final split policy, train-only preprocessing,
  model set and evaluation conventions.
- Refresh root `README.md` to match current commands and configs.
- Confirm whether dynamic landmark modeling remains required for the final
  thesis scope.

# Documentation Health Report

## Missing Information

- Final static experiments need run-level entries in `docs/EXPERIMENT_LOG.md`.
- Several methodological choices summarized in `docs/PROJECT_HISTORY.md` still
  need formal decision records in `docs/DECISIONS.md`.
- The dynamic pipeline scope, especially DySurv training with landmarks, remains
  unresolved.
- The existence and freshness of `outputs/metrics/static_model_comparison.csv`
  should be verified after the next clean pipeline run.

## Files That Need Future Maintenance

- `README.md`: update stale usage and config references.
- `docs/EXPERIMENT_LOG.md`: backfill final static model runs.
- `docs/DECISIONS.md`: add formal records for current accepted methodology.
- `docs/TODO.md`: keep task state current as dynamic scope and thesis writing
  decisions are made.
- `docs/REPRODUCIBILITY.md`: update whenever commands, dependencies or pipeline
  steps change.

## Recommendations For Technical Agents

- Do not edit `docs/PROJECT_HISTORY.md` unless explicitly instructed.
- Append decisions before or during changes that alter methodology,
  reproducibility or interpretation.
- Log every executed experiment with exact command, config, seed and outputs.
- Update `docs/TODO.md` when work creates, resolves or blocks a follow-up.
- Keep code/config changes separate from documentation governance work unless
  the user explicitly asks for both.

## Recommendations For The Project Manager/Historian

- Periodically consolidate stable decisions and experiment outcomes into
  `docs/PROJECT_HISTORY.md`.
- Remove or re-prioritize stale TODO items only with enough explanation to
  preserve traceability.
- Ensure root-facing documentation and docs-facing documentation describe the
  same current pipeline.
- Keep ownership boundaries clear: technical agents produce append-only working
  records; the Project Manager/Historian consolidates history.

## 2026-06-07 — AGENTS Governance Completion

### Purpose

Complete the repository-level governance instructions in `AGENTS.md` after
verifying that the file is not empty.

### Change Summary

- Added a documentation-only governance reading order.
- Added an explicit Documentation Governance Manager protocol.
- Clarified that documentation governance work must not modify model code,
  model configs, data, outputs or checkpoints.
- Added Documentation Governance Manager ownership permissions and limits.

### Validation

- `AGENTS.md` now covers the role used for the documentation audit and matches
  the ownership model in `docs/README.md`.
- No code, model, config, data or output files were modified for this follow-up.

## 2026-06-07 — Static Evaluation Protocol Update

### Purpose

Update static survival evaluation so curve-based models are not primarily judged
by final cumulative-risk Harrell C-index and so IBS/IBLL/NBLL use a fixed,
memory-safe grid.

### Change Summary

- Added fixed common grids:
  `evaluation_time_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9]` and
  `horizon_times = [1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- Kept natural Harrell C-index for CoxPH and DeepSurv.
- Renamed DeepHit/PCHazard final-risk C-index to
  `harrell_c_index_final_risk`.
- Added `ctd_antolini`, `horizon_c_index` and `mean_horizon_c_index` to metrics
  JSONs for eligible curve-producing models.
- Replaced pycox all-time IBS/IBLL evaluation with a fixed-grid implementation
  to avoid large intermediate matrices.
- Left train IBS/IBLL/NBLL as NaN by default; validation and test are computed.

### Files Updated

- `configs/coxph.yaml`
- `configs/deepsurv.yaml`
- `configs/pchazard.yaml`
- `configs/deephit.yaml`
- `src/evaluation/metrics.py`
- `src/evaluation/time_dependent_survival.py`
- `src/models/coxph_tfg.py`
- `src/models/deepsurv_tfg.py`
- `src/models/pchazard_tfg.py`
- `src/models/deephit_tfg.py`
- `scripts/evaluate_pchazard_time_dependent.py`
- `scripts/evaluate_deephit_time_dependent.py`
- `docs/DECISIONS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/TODO.md`

### Validation

- Ran `tfg-survival` Python compilation for modified evaluation/model/script
  files.
- Ran `pytest tests/test_time_dependent_survival_metrics.py
  tests/test_deephit_time_dependent_metrics.py`; result: 5 passed.
- Ran a toy `evaluate_predictions` check confirming fixed-grid IBS/IBLL/NBLL
  returns finite values when at least two valid evaluation times are available.

### Follow-up

- Regenerate `outputs/metrics/` with the updated pipeline before interpreting
  final static model results.

## 2026-06-07 — Metrics Subfolder Organization

### Purpose

Add a minimal organization layer under `outputs/metrics/` so each model writes
its own metric artifacts into a dedicated folder while preserving existing
filenames.

### Change Summary

- Added `model_metrics_dir(paths, model_name)` helper.
- Updated Kaplan-Meier, CoxPH, DeepSurv, PCHazard and DeepHit metric writes to
  use `outputs/metrics/<model>/`.
- Updated static evaluation config to read JSON metrics from model subfolders.
- Updated DeepHit and PCHazard time-dependent metric paths in configs/scripts.
- Updated reproducibility docs and decision log.

### Validation

- Ran `tfg-survival` Python compilation for modified model and script files.
- Ran `pytest tests/test_time_dependent_survival_metrics.py
  tests/test_deephit_time_dependent_metrics.py`; result: 5 passed.

### Follow-up

- Root-level metric files from older runs may still exist locally. New runs
  should write model artifacts into subfolders; old root-level model metric
  files can be ignored or cleaned manually after confirming the new artifacts.

## 2026-06-07 — DeepHit Audit Review

### Purpose

Review the adapted DeepHit implementation against the local original DeepHit
reference, the DeepHit paper and the current static evaluation pipeline without
modifying model code, preprocessing, configs or training outputs.

### Scope

- Inspected `src/models/deephit_tfg.py`, `configs/deephit.yaml`, DeepHit
  evaluation code, common metric code, the local DeepHit reference
  implementation and generated DeepHit metrics/predictions.
- Avoided scanning full `data/`, full `outputs/` and notebooks.
- Ran read-only diagnostics on existing DeepHit predictions.

### Main Findings

- DeepHit outputs a numerically valid PMF and monotone survival/CIF curves.
- The current DeepHit adaptation is suspicious for calibration: average test
  PMF places very large mass in the first bin and forces survival to zero at
  the configured final bin.
- The adapted ranking loss does not match the original event-time-specific
  pairwise comparison exactly.
- The fixed 10-bin softmax support and horizon capping likely contribute to
  poor IBS/IBLL despite reasonable Ctd and horizon C-index.

### Validation

- No retraining was run.
- No code or config files were changed.
- One attempted `python` invocation failed because Windows Python Manager tried
  to access the network; diagnostics were rerun successfully with the local
  `BL-env` Python executable.

## 2026-06-07 — Project Manager / Historian Initialization

### Purpose

Initialize the permanent Project Manager/Historian role for this repository and
assess the current documentation system without modifying model code, training
scripts, evaluation scripts, preprocessing code, data, outputs or checkpoints.

### Documentation Read

- `AGENTS.md`
- `docs/README.md`
- `SESSION_NOTES.md`
- `docs/TODO.md`
- `docs/DECISIONS.md`
- `docs/PROJECT_HISTORY.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/REPRODUCIBILITY.md`

### Current State

- The project is in a consolidated static-baseline stage.
- Static MIMIC-IV extraction, train-only static preprocessing, 60/20/20 split,
  and static model training/evaluation are documented.
- The dynamic DySurv/landmark pipeline remains unresolved and is the main scope
  blocker for final static-vs-dynamic claims.
- Git history contains only the initial commit, while many local files are
  uncommitted or untracked; documentation should therefore treat
  `docs/PROJECT_HISTORY.md` and session notes as the current continuity source.

### Documentation Health

- Strengths: clear ownership rules, strong project history narrative, current
  reproducibility commands, and actionable TODO priorities.
- Gaps: final static experiments are summarized in history but not backfilled
  as formal experiment-log entries; several methodological choices still need
  formal decision records; root `README.md` is stale relative to current
  scripts/configs.

### Updates Made

- Added Project Manager/Historian initialization to `docs/PROJECT_HISTORY.md`.
- Added `DEC-003` for permanent Project Manager/Historian ownership.
- Added a TODO item for periodic consolidation of important session notes.

### Next Actions

- Backfill final static experiment entries in `docs/EXPERIMENT_LOG.md`.
- Add decision records for final split policy, train-only preprocessing and
  model set.
- Decide and document whether full dynamic landmark/DySurv training remains in
  thesis scope.

## 2026-06-08 — DeepHit Audit Documentation

### Purpose

Document the DeepHit audit findings, hypotheses and next actions before any
implementation or tuning changes begin.

### Key Metrics

- DeepHit final-risk C-index: approximately 0.49.
- DeepHit Antolini Ctd: approximately 0.75.
- DeepHit mean horizon C-index: approximately 0.73.
- DeepHit IBS: approximately 0.40.
- DeepHit IBLL: approximately 1.02.
- Strong static comparator region for DeepSurv/PCHazard: IBS approximately
  0.11 and IBLL approximately 0.35 where curve metrics are available.

### Main Findings

- DeepHit appears to rank patients reasonably well in time-dependent metrics.
- DeepHit appears poorly calibrated.
- PMF sums correctly, and survival/CIF monotonicity checks passed.
- Mean PMF allocates excessive mass to early and final bins.
- `S(10) = 0` for all patients under the current 10-day support.
- The current design forces all event probability inside the 10-day horizon.
- The ranking loss differs from the original DeepHit paper.
- Calibration loss is currently disabled with `gamma = 0`.

### Rationale For Postponing Tuning

Hyperparameter tuning is postponed because the current findings point to
implementation, encoding, support and calibration questions. Tuning before those
questions are resolved could mask a methodological problem rather than improve
the intended DeepHit replication.

### Planned Next Steps For Tomorrow

- Investigate an explicit beyond-horizon / tail category.
- Review event and censoring encoding.
- Review the ranking loss against the original DeepHit paper.
- Run calibration diagnostics.
- Re-evaluate DeepHit after fixes before starting hyperparameter tuning.

### Documentation Updates

- Added `DEC-004` to `docs/DECISIONS.md`.
- Added a high-priority DeepHit calibration review section to `docs/TODO.md`.
- Added a short DeepHit benchmark/audit entry to `docs/PROJECT_HISTORY.md`.

## 2026-06-08 — End of Session Project Consolidation

### What was completed today

- Redesigned the static survival metrics protocol so curve-producing models are
  not judged primarily by final-risk Harrell C-index.
- Standardized the common evaluation grid to `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- Added Ctd / horizon-based evaluation as the main discrimination view for
  curve-producing static models.
- Reviewed the DeepHit static benchmark behavior.
- Identified DeepHit calibration issues before starting any tuning cycle.

### Main conclusions

- DeepSurv and PCHazard are currently the strongest static models.
- DeepHit discriminates reasonably under time-dependent metrics, but appears
  poorly calibrated.
- DeepHit hyperparameter tuning is postponed until the implementation review is
  completed.

### Open questions

- Tail mass beyond the prediction horizon.
- Ranking loss implementation.
- Event/censoring encoding.
- Calibration diagnostics.

### Tomorrow's first tasks

1. Review DeepHit time support.
2. Review DeepHit ranking loss.
3. Run diagnostics.
4. Re-evaluate metrics.
5. Decide whether tuning can start.
6. Ensure `docs/TODO.md` priorities remain up to date.

### Short status report

- Current project stage: consolidated static-baseline stage with updated
  evaluation methodology and DeepHit under targeted audit.
- Blockers: DeepHit calibration/support questions, missing final
  `static_model_comparison.csv` consolidation, and unresolved dynamic
  DySurv/landmark scope.
- Next milestone: complete the DeepHit implementation/calibration review and
  re-evaluate metrics before deciding whether tuning is methodologically safe.
- Recommended first task for tomorrow: review DeepHit time support and whether
  an explicit beyond-horizon/tail category is needed.

### Documentation status

- Reviewed `SESSION_NOTES.md`, `docs/TODO.md`, `docs/DECISIONS.md` and
  `docs/PROJECT_HISTORY.md`.
- `docs/TODO.md` priorities are current; no TODO edit was needed during this
  end-of-session consolidation.
- No code, configs, models, training scripts or evaluation scripts were
  modified.

## 2026-06-08 — Start of Session Review

### Purpose

Review current project state before beginning the next work session, with no
code, config, model, data, output, notebook or checkpoint inspection.

### Documents Reviewed

- `SESSION_NOTES.md`
- `docs/TODO.md`
- `docs/DECISIONS.md`
- `docs/PROJECT_HISTORY.md`
- Governance context from `AGENTS.md`, `CODEX_INSTRUCTIONS.md`,
  `docs/README.md`, `docs/EXPERIMENT_LOG.md` and `docs/REPRODUCIBILITY.md`

### Findings

- Yesterday's DeepHit findings are properly documented in `SESSION_NOTES.md`,
  `docs/DECISIONS.md` (`DEC-004`), `docs/TODO.md` and
  `docs/PROJECT_HISTORY.md`.
- `docs/TODO.md` priorities remain correct: DeepHit calibration review is the
  first High Priority block.
- No documentation inconsistency requiring correction was found.

### Current Status

- Current project stage: consolidated static-baseline stage with updated
  evaluation methodology and DeepHit under targeted calibration/implementation
  review.
- Main blockers: DeepHit tail support, ranking loss, event/censoring encoding,
  calibration diagnostics, missing final static metrics consolidation and
  unresolved dynamic DySurv/landmark scope.
- Recommended first task: review DeepHit time support and whether an explicit
  beyond-horizon/tail category is required.

## 2026-06-08 — Technical Implementation Agent Initialization

### Purpose

Initialize the permanent Technical Implementation Agent role for model,
preprocessing, evaluation, debugging, testing and reproducible experimentation
work in this repository.

### Documents Reviewed

- `AGENTS.md`
- `CODEX_INSTRUCTIONS.md`
- `TFG/CODEX_TFG_MATES_JAVI.md`
- `docs/README.md`
- `docs/TODO.md`
- `SESSION_NOTES.md`
- `docs/DECISIONS.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/REPRODUCIBILITY.md`
- `docs/PROJECT_HISTORY.md` for read-only context

### Current Technical Status

- The project is in a consolidated static-baseline stage.
- The static MIMIC-IV pipeline, train-only preprocessing, 60/20/20 split and
  static model set are documented and implemented.
- DeepSurv and PCHazard are the strongest current static baselines.
- DeepHit is under targeted implementation and calibration review before
  tuning or thesis-level interpretation.
- Dynamic DySurv/landmark training remains out of the final executable pipeline
  until scope, split protocol, compute budget and dataset design are confirmed.

### Active Technical Priorities

- Complete the DeepHit audit: tail support, event/censor encoding, ranking loss
  fidelity and calibration diagnostics.
- Regenerate or consolidate final static metrics after the updated fixed-grid
  evaluation protocol.
- Validate metric artifact locations under model-specific subfolders.
- Preserve patient/stay-level split discipline before any dynamic landmark
  generation.
- Keep technical changes small, config-driven and covered by the smallest
  relevant tests.

### Notes

- No model code, configs, data, outputs, notebooks or checkpoints were modified
  during initialization.
- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No new technical decision was made, so `docs/DECISIONS.md` was not updated.
- No priorities changed, so `docs/TODO.md` was not updated.

## 2026-06-08 — DeepHit Implementation Review

### Purpose

Review the current DeepHit implementation for support, censoring/event encoding
and ranking-loss fidelity before making code changes or starting tuning.

### Scope

- Reviewed `configs/deephit.yaml`.
- Reviewed `src/models/deephit_tfg.py`.
- Reviewed shared static target capping in `src/models/static_common.py`.
- Reviewed DeepHit time-dependent and curve metric code in `src/evaluation/`.
- Compared against the local original DeepHit reference in
  `src/models_references/DeepHit/class_DeepHit.py` and
  `src/models_references/DeepHit/import_data.py`.
- Did not inspect full `data/`, full `outputs/`, notebooks or checkpoints.

### Findings

- The current single-event DeepHit softmax allocates all probability mass across
  the 10 configured event bins.
- The resulting survival curve is forced to zero at the final configured bin,
  so the implementation effectively enforces `P(T <= 10 days) = 1`.
- Censored observations capped at the 10-day horizon have an empty
  post-censoring mask under the current `mask1` logic.
- The original DeepHit reference also uses a finite softmax support, but sets
  `num_Category` larger than the maximum observed time to preserve tail room.
- The current ranking loss is not equivalent to the reference ranking loss:
  it compares each patient at its own time rather than comparing pairs at the
  event subject's time.
- Event encoding remains consistent with the intended single-risk setup:
  event `1`, censoring `0`, `num_Event = 1`; the problem is support/capping, not
  the binary event convention itself.

### Recommendation

- Do not tune DeepHit yet.
- First implement an explicit beyond-horizon tail category or equivalent
  extended support.
- Then correct ranking loss to the original event-time-conditioned pairwise
  formulation.
- Add focused unit tests before retraining.

### Documentation Updates

- Added proposed decision `DEC-005` to `docs/DECISIONS.md`.
- No model code, configs, data, outputs, notebooks or checkpoints were modified.
- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- `docs/TODO.md` already contains the relevant high-priority DeepHit tasks, so
  no TODO edit was needed.

## 2026-06-08 — Approved DeepHit Corrections

### Purpose

Implement only the reviewer-approved DeepHit fixes: extended/tail support,
censored-at-horizon mask repair, original-style event-time-conditioned ranking
loss and focused validation tests.

### Changes Made

- Added `include_tail_category: true` to `configs/deephit.yaml` while keeping
  `num_Category: 10` as the evaluated 10-day event-bin count.
- Added an internal DeepHit output category for probability mass beyond the
  10-day horizon.
- Updated DeepHit masks so censored observations at the horizon select the tail
  bin instead of an empty post-censoring slice.
- Reconstructed DeepHit survival curves from only the 10 evaluated event bins,
  with final survival equal to the tail probability when tail support is
  enabled.
- Corrected DeepHit ranking loss to compare pairwise risks at the event
  subject's time, matching the local original DeepHit reference logic.
- Added focused synthetic tests for tail masks, tail survival reconstruction
  and ranking-loss pair logic.
- Fixed a small tensor-shape issue in the DeepHit likelihood calculation so
  per-subject likelihood terms do not broadcast across the batch.

### Validation

- Attempted `python -m pytest tests/test_static_pipeline.py`; failed because
  Windows Python Manager attempted network runtime discovery.
- Attempted `C:\Users\Javi\miniconda3\envs\BL-env\python.exe -m pytest
  tests/test_static_pipeline.py`; failed because `pytest` is not installed in
  `BL-env`.
- Ran `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest
  tests/test_static_pipeline.py`; result: 8 passed.

### Documentation Updates

- Updated `DEC-005` in `docs/DECISIONS.md` from proposed to accepted and
  recorded the implemented tail-support convention.
- Updated `docs/TODO.md` to mark the implemented DeepHit support, ranking-loss
  and censoring tasks complete while leaving calibration diagnostics and
  benchmark regeneration open.
- No experiment or retraining was run, so `docs/EXPERIMENT_LOG.md` was not
  updated.
- `docs/REPRODUCIBILITY.md` was not updated because commands and execution
  steps did not change.

## 2026-06-08 — Corrected DeepHit Run Logging

### Purpose

Log and interpret the new DeepHit run produced after the approved
tail-support, censored-horizon likelihood, survival reconstruction, ranking-loss
and likelihood broadcasting fixes.

### Scope

- Inspected newest model-specific artifacts in `outputs/metrics/deephit/`.
- Ignored old DeepHit files directly under `outputs/metrics/`.
- Inspected `outputs/predictions/deephit_predictions.parquet` only for
  PMF/tail and `S(10)` diagnostics associated with the new run.
- Did not modify model code, retrain models or change configs.

### Command

- Exact shell history was not embedded in the metric artifacts.
- Canonical command recorded for reproducibility:
  `python scripts/train_static_model.py --config configs/deephit.yaml`.

### Run Status

- The run completed and produced new DeepHit metric artifacts dated
  `2026-06-08 12:58:11`.
- Training log stopped after epoch 18 with best validation loss 0.447695.

### Key Metrics

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
- Test C-index@h for h=1,...,9:
  0.7925, 0.7765, 0.7603, 0.7499, 0.7475, 0.7378, 0.7315, 0.7306, 0.7278.

### Tail Diagnostics

- Mean probability mass in evaluated bins 1-10: 0.2636.
- Mean tail probability beyond 10 days: 0.7364.
- Mean `S(10)`: 0.7364.
- `S(10)` was nonzero for all generated predictions.
- `S(10)` equals `tail_probability_beyond_horizon` in the prediction artifact.

### Comparison With Previous DeepHit

- Final-risk C-index improved from approximately 0.4879 to 0.7529.
- Ctd improved from approximately 0.7509 to 0.7695.
- Mean horizon C-index improved from approximately 0.73 to 0.7505.
- IBS improved from approximately 0.4044 to 0.1107.
- IBLL/NBLL improved from approximately 1.0431 to 0.3531.
- The previous run forced `S(10) = 0`; the corrected run now learns nonzero
  beyond-horizon survival.

### Documentation Updates

- Added `EXP-004` to `docs/EXPERIMENT_LOG.md`.
- Added `DEC-006` to `docs/DECISIONS.md` because the results change the
  methodological interpretation of DeepHit.
- Updated `docs/TODO.md` to mark the corrected DeepHit rerun and metric
  inspection complete while keeping calibration plots, PCHazard survival-curve
  comparison, synthetic overfit test and hyperparameter tuning open.

### Next Recommended Action

Generate corrected DeepHit calibration plots and compare DeepHit survival curves
against PCHazard before beginning hyperparameter tuning.

## 2026-06-08 — Git Hygiene / Repo Readiness Audit

### Purpose

Review repository organization and Git safety before pushing to GitHub or
cloning into Lightning AI, without deleting files, untracking files or changing
model logic.

### Scope

- Reviewed repository governance files and documentation protocol.
- Inspected top-level repository structure, `.gitignore`, Git status and tracked
  file names.
- Checked tracked data/output/model artifact paths through the Git index only.
- Avoided deep inspection of `data/`, `outputs/`, local environments,
  checkpoints and model artifact folders.

### Findings

- Git currently tracks only placeholder files under `data/` and `outputs/`, plus
  `data/mimic_extraction_explanation.md`; no raw MIMIC-IV data, processed
  patient tables, checkpoints, trained model weights or prediction artifacts are
  tracked.
- Untracked MIMIC-IV raw and derived files were visible before the hygiene
  update and could have been accidentally staged.
- Local generated artifacts under `outputs/`, `outputs/preprocessors/`,
  checkpoints and external reference/preprocessing repositories should remain
  local unless explicitly reviewed and approved.
- The working tree still contains many code/documentation additions, legacy
  deletions and modified files that need human review before committing.

### Changes Made

- Updated `.gitignore` to ignore MIMIC-IV raw/interim/processed data contents,
  generated preprocessors, root-level checkpoints and local external reference
  repositories while preserving tracked placeholder files and
  `data/mimic_extraction_explanation.md`.

### Validation

- Ran `git status --short --ignored` to confirm sensitive/generated paths are
  ignored.
- Ran `git ls-files` checks to confirm tracked `data/` and `outputs/` entries
  are limited to placeholders and the extraction explanation file.
- Ran `git check-ignore -v` on representative data, output and reference paths.

### Documentation Updates

- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No methodology or execution command changed, so `docs/DECISIONS.md` and
  `docs/REPRODUCIBILITY.md` were not updated.
- Existing `docs/TODO.md` already marks publication/package readiness as blocked
  until data/output exclusion and reference-code policy are reviewed.

## 2026-06-08 — Static Pipeline Git Scope Review

### Purpose

Identify which repository files are actually needed for the current static
pipeline and which files can be kept out of the next Git commit until they are
used.

### Scope

- Reviewed active static scripts and configs:
  `scripts/run_static_pipeline.py`, `scripts/build_static_data.py`,
  `scripts/train_static_model.py`, `scripts/evaluate_static_model.py` and the
  current static YAML configs.
- Traced imports under `scripts/`, `src/`, `tests/` and `configs/`.
- Avoided deep inspection of `data/`, `outputs/`, environments, checkpoints and
  generated artifacts.

### Findings

- The active static pipeline uses the new static configs and adapted TFG model
  files, not the old tracked configs/scripts/model files that currently appear
  as deleted in Git status.
- Notebooks, legacy PWE/RSF modules, old generic configs and old orchestration
  scripts are not needed for the current static training/evaluation pipeline.
- `src/data/mimic_direct_extraction.py`,
  `src/data/mimic_timeseries_sparse.py`, `src/features/build_features.py`,
  `src/features/pwe_transformer.py` and `references/query_original_43kreg.sql`
  are not part of the current static training/evaluation command chain, but are
  documented as upstream extraction/history material.
- Local methodological references under `src/models_references/` and
  `src/preprocessing paper/` remain local-only under `.gitignore`.

### Changes Made

- No code, config or `.gitignore` changes were made for this review.

### Documentation Updates

- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No new methodological decision was made, so `docs/DECISIONS.md` was not
  updated.
- Existing TODO priorities already include publication/package readiness and
  notebook review, so `docs/TODO.md` was not changed.

## 2026-06-08 — Static Pipeline Dependency Closure Audit

### Purpose

Verify that the proposed static-pipeline Git scope is sufficient to run after a
fresh clone, without modifying model code, deleting files or running full
experiments.

### Scope

- Inspected imports for `scripts/run_static_pipeline.py`,
  `scripts/build_static_data.py`, `scripts/train_static_model.py`,
  `scripts/evaluate_static_model.py` and their internal `src/` dependency
  closure.
- Checked static YAML config references.
- Checked test imports.
- Reviewed declared Python dependencies in `requirements.txt` and
  `environment.yml`.

### Findings

- The static runtime dependency closure is covered by the new static scripts,
  configs, adapted model files, shared static/evaluation utilities and
  `src/**/__init__.py` package marker files.
- The previous proposed add list should include `src/__init__.py`,
  `src/data/__init__.py`, `src/evaluation/__init__.py`,
  `src/models/__init__.py` and `src/utils/__init__.py`.
- `configs/static_pipeline.yaml` currently has `run_build_static_data: false`,
  so a fresh clone will not build `data/processed/static/*.parquet` before
  training unless the user either runs `scripts/build_static_data.py` manually
  or changes the config before committing.
- The static data build requires external, non-versioned inputs:
  `data/processed/mimic_extraction/flat_features.csv` and
  `data/processed/mimic_extraction/labels.csv`.
- Tests import only files inside the static closure plus
  `src/evaluation/deephit_time_dependent.py` for DeepHit metric tests.

### Changes Made

- No code, config or `.gitignore` changes were made.
- This session note was appended to satisfy repository documentation protocol.

### Documentation Updates

- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No new technical or methodological decision was made, so
  `docs/DECISIONS.md` was not updated.
- No priorities changed, so `docs/TODO.md` was not updated.
- Execution instructions were not changed, so `docs/REPRODUCIBILITY.md` was not
  updated.

## 2026-06-08 — Static Commit `.gitignore` Refinement

### Purpose

Adapt `.gitignore` so the next Git commit stays focused on the current static
pipeline and does not accidentally include notebooks, legacy code, external
references, generated artifacts or sensitive MIMIC-derived data.

### Changes Made

- Added ignore rules for exploratory and historical folders:
  `notebooks/`, `reports/` and `references/`.
- Added ignore rules for old configs, scripts and legacy model files no longer
  used by the current static pipeline.
- Added ignore rules for upstream extraction/feature-engineering helpers that
  are not imported by `scripts/run_static_pipeline.py` or its dependency
  closure.
- Added ignore rules for optional standalone time-dependent diagnostic scripts.
- Added comments clarifying that some ignored data files are still required at
  runtime but must be supplied manually after cloning.

### Validation

- Ran `git check-ignore --no-index -v` on representative notebook, reference,
  legacy script, legacy model, optional diagnostic and generated data paths.
- Ran `git status --short` to review the remaining commit surface.

### Documentation Updates

- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No methodological or execution decision was made, so `docs/DECISIONS.md` and
  `docs/REPRODUCIBILITY.md` were not updated.
- Existing TODO priorities already cover publication/package readiness and
  notebook review, so `docs/TODO.md` was not changed.

## 2026-06-08 — Static Tuning and Final-Seed Pipeline Preparation

### Purpose

Prepare validation-only hyperparameter tuning and three-seed final evaluation
for CoxPH, DeepSurv, PCHazard and corrected DeepHit without running full
tuning.

### Changes Made

- Added `configs/static_tuning.yaml` with small model-specific grids and fixed
  evaluation/horizon grids `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- Added `scripts/tune_static_models.py` for validation-only tuning output under
  `outputs/tuning/{model}/`.
- Added `scripts/run_final_static_seeds.py` for final selected-hyperparameter
  runs under `outputs/final_static/{model}/seed_{seed}/`.
- Added shared static-trainer support for configurable evaluation splits,
  explicit test-metric guards, optional prediction/model/checkpoint artifacts
  and validation-only runs that do not load the test split.
- Updated CoxPH, DeepSurv, PCHazard and DeepHit trainers to use the shared
  split/artifact controls while preserving baseline defaults.
- Added focused tests for grid expansion, validation-only run config creation,
  validation Ctd/IBLL selection and exact final seed enforcement.

### Safeguards

- Tuning uses train and validation splits only, with
  `evaluation.allow_test_metrics: false`.
- Final-seed runs require exactly seeds `42`, `123` and `2026`.
- Tuning/final runs save config snapshots when actually executed.
- PCHazard/DeepHit baseline-specific time-dependent output paths are stripped
  from generated run configs so runs write into isolated output folders.
- Large model/checkpoint artifacts are disabled by default in tuning and final
  static seed configs unless explicitly enabled.

### Validation

- Ran
  `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_static_tuning.py`;
  result: 5 passed.
- Ran
  `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_static_tuning.py tests/test_static_pipeline.py`;
  result: 13 passed.
- Ran
  `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_static_models.py --config configs/static_tuning.yaml --models coxph --dry-run`;
  result: planned two isolated CoxPH tuning configs under
  `outputs/tuning/coxph/...`.
- Ran
  `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/run_final_static_seeds.py --help`;
  result: command loaded and displayed CLI help.

### Documentation Updates

- Added `DEC-007` to `docs/DECISIONS.md`.
- Updated `docs/REPRODUCIBILITY.md` with tuning and final-seed commands.
- Updated `docs/TODO.md` to mark pipeline preparation complete and keep actual
  tuning/final static runs open.
- No full tuning or final model experiment was run, so
  `docs/EXPERIMENT_LOG.md` was not updated.

### Next Recommended Action

Run validation-only static tuning after the remaining DeepHit diagnostic tasks
or after explicit approval to start tuning.

## 2026-06-08 — Lightning Readiness Audit For Static Tuning

### Purpose

Check whether the static tuning and final-seed pipeline can run from a fresh
clone in Lightning AI without modifying model logic or running full
experiments.

### Scope

- Inspected imports for `scripts/tune_static_models.py`,
  `scripts/run_final_static_seeds.py`, `scripts/train_static_model.py` and
  `scripts/evaluate_static_model.py`.
- Checked `configs/static_tuning.yaml` and static model configs for relative
  paths.
- Checked `.gitignore` coverage for generated tuning/final outputs, data,
  checkpoints and model artifacts.
- Checked declared dependencies in `requirements.txt` and `environment.yml`.

### Findings

- The tuning/final scripts depend on existing static model trainers, shared
  static utilities and standard project utilities; no new Python package was
  found beyond the current requirements.
- `configs/static_tuning.yaml` uses relative paths only.
- `configs/static_pipeline.yaml` currently has `run_build_static_data: true`,
  which is better for fresh clones after external MIMIC-derived inputs are
  supplied.
- Added `.gitignore` coverage for generated `outputs/tuning/` and
  `outputs/final_static/`.
- Legacy tracked files `src/data/mimic_direct_extraction.py` and
  `src/data/mimic_timeseries_sparse.py` still contain hardcoded local Windows
  paths, but they are not needed by the static tuning/final-seed pipeline.

### Changes Made

- Updated `.gitignore` to ignore `outputs/tuning/` and
  `outputs/final_static/`.

### Documentation Updates

- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No methodology changed, so `docs/DECISIONS.md` was not updated.
- No priorities changed, so `docs/TODO.md` was not updated.
- `docs/REPRODUCIBILITY.md` was not updated because this turn only audited the
  existing commands.

## 2026-06-08 — CoxPH Smoke-Test Metric Regression Investigation

### Purpose

Investigate why the new CoxPH final-seed smoke test produced much worse
metrics than the previous static CoxPH benchmark before pushing/running on
Lightning AI.

### Findings

- `configs/static_pipeline.yaml`, `configs/static_data.yaml`,
  `configs/coxph.yaml` and `configs/static_evaluation.yaml` still point to the
  same static train/validation/test parquet files, feature set, duration/event
  columns, preprocessing outputs and fixed evaluation grid.
- The static dataset summary remains at 35 features with 56,101 train, 18,700
  validation and 18,701 test patients.
- The partial CoxPH smoke tuning output only contained `coxph_cfg_001`, which
  used `penalizer=0.01` and `l1_ratio=0.0`.
- That weak-penalty candidate produced unstable coefficients: `height`
  approximately `-1717` and `nullheight` approximately `10.8`, consistent with
  the lifelines convergence warning and the degraded metrics.
- A diagnostic run through the new final-static pipeline with the old stable
  setting `penalizer=0.1`, `l1_ratio=0.0` reproduced the previous benchmark:
  test Ctd/Harrell 0.7411, test IBS 0.1147 and test IBLL/NBLL 0.3693.

### Changes Made

- Expanded the CoxPH tuning grid in `configs/static_tuning.yaml` to
  `penalizer: [0.0, 0.001, 0.01, 0.1]` with `l1_ratio: [0.0]`.
- Updated `docs/REPRODUCIBILITY.md` with the CoxPH grid and convergence-warning
  note.
- Added `DEC-008` documenting the CoxPH grid expansion and warning against
  reusing the partial smoke selection.
- Added `EXP-005` documenting the isolated old-configuration diagnostic run.
- Updated `docs/TODO.md` to require rerunning CoxPH validation-only tuning with
  the full grid before Lightning AI final runs.
- Added `outputs/diagnostics/` to `.gitignore` because the CoxPH diagnostic
  run writes local generated artifacts that should not be committed.

### Validation

- Ran the old fixed CoxPH diagnostic via `tfg-survival`; result reproduced the
  old CoxPH benchmark.
- Ran `tfg-survival` dry-run for CoxPH tuning; result planned four CoxPH
  configurations.
- Ran `tfg-survival` `pytest tests/test_static_tuning.py`; result: 5 passed.

### Recommendation

- The tuning/final pipeline is structurally safe for Lightning AI, but the
  current CoxPH `best_hyperparameters.json` from the partial smoke run is not
  safe to reuse.
- Re-run CoxPH tuning with the expanded grid before any final static seed run.

## 2026-06-09 — Final Static Three-Seed Results Review

### Purpose

Analyze existing tuning and final-seed artifacts for DeepHit, DeepSurv and
PCHazard without running additional training or modifying model logic.

### Scope

- Inspected `outputs/tuning/{deephit,deepsurv,pchazard}/`.
- Inspected `outputs/final_static/{deephit,deepsurv,pchazard}/`.
- Ignored CoxPH for this summary because the CoxPH tuning/final seed sequence
  remains pending after the smoke-test regression investigation.

### Selected Configurations

- DeepHit selected `deephit_cfg_013`: `shared_layers=[128, 64]`,
  `cause_layers=[64]`, `dropout=0.1`, `learning_rate=0.0005`,
  `alpha=1.0`, `beta=0.5`, `gamma=0.0`, `ranking_sigma=0.1`,
  `include_tail_category=true`.
- DeepSurv selected `deepsurv_cfg_018`: `hidden_layers=[128, 64]`,
  `dropout=0.1`, `learning_rate=0.0001`, `weight_decay=0.001`.
- PCHazard selected `pchazard_cfg_011`: `hidden_layers=[128, 64]`,
  `dropout=0.3`, `learning_rate=0.0005`.

### Main Results

- DeepHit mean test Ctd: 0.7690; mean horizon C-index: 0.7490; IBS: 0.1104;
  IBLL/NBLL: 0.3526.
- DeepSurv mean test Ctd/Harrell: 0.7615; mean horizon C-index: 0.7463;
  IBS: 0.1110; IBLL/NBLL: 0.3560.
- PCHazard mean test Ctd: 0.7688; mean horizon C-index: 0.7491; IBS: 0.1095;
  IBLL/NBLL: 0.3507.

### Interpretation

- DeepHit and PCHazard are nearly tied in discrimination, with DeepHit slightly
  higher in mean test Ctd and PCHazard slightly higher in mean horizon C-index.
- PCHazard has the best calibration/error profile by IBS and IBLL/NBLL.
- DeepSurv remains stable and competitive, but trails the two curve-discrete
  models after tuning.
- The three-seed variability is small for all three models, so the ranking is
  not driven by a single anomalous seed.

### Documentation Updates

- Added `EXP-006` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/TODO.md` to record the completed three-model analysis and keep
  CoxPH/final static consolidation open.
- No new methodological decision was made, so `docs/DECISIONS.md` was not
  updated.
- No commands or execution steps changed, so `docs/REPRODUCIBILITY.md` was not
  updated.

## 2026-06-09 — Complete Final Static Results Including CoxPH

### Purpose

Add the newly available CoxPH tuning/final-seed outputs to the static model
comparison and refresh the academic interpretation across CoxPH, DeepHit,
DeepSurv and PCHazard.

### Scope

- Inspected `outputs/tuning/{coxph,deephit,deepsurv,pchazard}/`.
- Inspected `outputs/final_static/{coxph,deephit,deepsurv,pchazard}/`.
- Did not run additional training and did not modify model logic or configs.

### Selected Configurations

- CoxPH selected `coxph_cfg_003`: `penalizer=0.1`, `l1_ratio=0.0`.
- DeepHit selected `deephit_cfg_013`: `shared_layers=[128, 64]`,
  `cause_layers=[64]`, `dropout=0.1`, `learning_rate=0.0005`,
  `alpha=1.0`, `beta=0.5`, `gamma=0.0`, `ranking_sigma=0.1`,
  `include_tail_category=true`.
- DeepSurv selected `deepsurv_cfg_018`: `hidden_layers=[128, 64]`,
  `dropout=0.1`, `learning_rate=0.0001`, `weight_decay=0.001`.
- PCHazard selected `pchazard_cfg_011`: `hidden_layers=[128, 64]`,
  `dropout=0.3`, `learning_rate=0.0005`.

### Main Results

- CoxPH mean test Harrell/Ctd: 0.7411; mean horizon C-index: 0.7266;
  IBS: 0.1147; IBLL/NBLL: 0.3693.
- DeepHit mean test Ctd: 0.7690; mean horizon C-index: 0.7490; IBS: 0.1104;
  IBLL/NBLL: 0.3526.
- DeepSurv mean test Harrell/Ctd: 0.7615; mean horizon C-index: 0.7463;
  IBS: 0.1110; IBLL/NBLL: 0.3560.
- PCHazard mean test Ctd: 0.7688; mean horizon C-index: 0.7491;
  IBS: 0.1095; IBLL/NBLL: 0.3507.

### Interpretation

- CoxPH now reproduces the previous stable benchmark after full-grid tuning
  selected `penalizer=0.1`.
- All neural/static models improve over CoxPH, especially on Ctd and
  calibration-style curve metrics.
- PCHazard has the best IBS and IBLL/NBLL, while DeepHit and PCHazard are
  essentially tied on discrimination.
- DeepSurv provides a stable nonlinear proportional-risk baseline between
  CoxPH and the two curve-discrete models.

### Documentation Updates

- Added `EXP-007` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/TODO.md` to mark CoxPH tuning/final-seed and the four-model
  static comparison as completed.
- No new methodological decision was made, so `docs/DECISIONS.md` was not
  updated.
- No execution commands changed, so `docs/REPRODUCIBILITY.md` was not updated.

## 2026-06-09 — End of Day Session Close

### Purpose

Close the session for the day and preserve handoff context for the next working
session.

### Current State

- Final static three-seed results have been reviewed for CoxPH, DeepHit,
  DeepSurv and PCHazard.
- The four-model static comparison is now documented in
  `docs/EXPERIMENT_LOG.md` and reflected in `docs/TODO.md`.
- DeepHit tail-support/ranking-loss/censoring fixes are already documented as
  completed, with remaining diagnostic tasks still open.
- Dynamic DySurv/landmark scope remains blocked until a thesis-scope decision
  is made.

### Next Recommended Actions

- Consolidate final static results into thesis-ready tables/figures or
  `outputs/metrics/static_model_comparison.csv` if not already generated.
- Complete remaining DeepHit diagnostics: original-paper assumption comparison,
  calibration plots, PCHazard curve comparison and synthetic overfit test.
- Decide whether the thesis still requires a full dynamic landmark/DySurv
  pipeline.

### Documentation Status

- Reviewed `AGENTS.md`, `SESSION_NOTES.md` and `docs/TODO.md`.
- No code, configs, models, data, outputs, notebooks or checkpoints were
  modified during this close-out.
- No experiment was run during this close-out.

## 2026-06-10 — Metrics Agent Static Results Package

### Purpose

Prepare a structured final static results package for handoff to the MEMORY
WRITER agent.

### Scope

- Inspected documentation state in `docs/DECISIONS.md`,
  `docs/EXPERIMENT_LOG.md`, `docs/REPRODUCIBILITY.md`, `docs/TODO.md` and
  `SESSION_NOTES.md`.
- Inspected final static artifacts under
  `outputs/tuning/{coxph,deepsurv,pchazard,deephit}/` and
  `outputs/final_static/{coxph,deepsurv,pchazard,deephit}/`.
- Checked descriptive Kaplan-Meier outputs and dataset summary under
  `outputs/metrics/`.
- Did not run training, tuning or evaluation consolidation.

### Findings

- CoxPH, DeepSurv, PCHazard and DeepHit all have tuning results, selected
  hyperparameters, final seed summaries, final seed CSVs and per-seed metric
  JSONs for seeds 42, 123 and 2026.
- Kaplan-Meier has descriptive cohort metrics and a survival-curve figure, but
  remains descriptive only.
- `outputs/metrics/static_model_comparison.csv` exists but contains older
  static-pipeline results, including old DeepHit metrics before the final
  corrected/tuned DeepHit run; it should not be used as the final static table
  unless regenerated from `outputs/final_static/`.
- The current `configs/static_tuning.yaml` CoxPH grid differs from an older
  documentation note: local tuning results correspond to six CoxPH ridge
  candidates and selected `penalizer=0.1`.

### Documentation Updates

- Appended this session note only.
- No experiment was run, so `docs/EXPERIMENT_LOG.md` was not updated.
- No methodology changed, so `docs/DECISIONS.md` and
  `docs/REPRODUCIBILITY.md` were not updated.
- No priority changed, so `docs/TODO.md` was not updated.

## 2026-06-10 — DeepSurv Audit Review

### Purpose

Audit the current TFG DeepSurv implementation against the original DeepSurv
paper, the local reference implementation, current project methodology and
available tuning/final static artifacts.

### Scope

- Inspected `src/models_references/DeepSurv/`,
  `src/models/deepsurv_tfg.py`, `configs/deepsurv.yaml`,
  `configs/static_tuning.yaml`, shared static/evaluation utilities, relevant
  tests and DeepSurv-specific tuning/final/metrics artifacts.
- Avoided full `data/`, full `outputs/`, checkpoints and raw prediction scans.
- Ran small read-only checks for Cox partial likelihood consistency and final
  DeepSurv survival-curve range/monotonicity.

### Findings

- The adapted DeepSurv model preserves the core Cox neural-risk formulation:
  static covariates to scalar log-risk, trained with Cox partial likelihood.
- Censoring is handled correctly inside the partial likelihood; censored rows
  contribute to risk sets but not event terms.
- The main caveat is minibatch training of the Cox partial likelihood, which is
  an approximation to the full risk-set likelihood used by the original
  reference implementation and should be documented.
- Survival curves are reconstructed using a Breslow baseline cumulative hazard
  estimated on the capped train split; final seed-42 curves were monotone and
  within `[0, 1]`.
- Tuning artifacts show validation-only selection with no test metrics recorded;
  final artifacts use seeds `42`, `123` and `2026`.

### Validation

- Ran a small manual Cox-loss check with `tfg-survival`; corrected manual value
  matched implementation within numerical tolerance.
- Ran a read-only monotonicity/range check on
  `outputs/final_static/deepsurv/seed_42/predictions/deepsurv_test_survival_curves.csv`.
- No code, configs, model artifacts, generated outputs or checkpoints were
  modified.

## 2026-06-10 — Static Thesis Tables And Figures Generation

### Purpose

Generate thesis-ready static result tables, a clean final static comparison
artifact and static model figures from final static outputs only.

### Scope

- Used `outputs/final_static/{coxph,deepsurv,pchazard,deephit}/` as the source
  of truth.
- Used `outputs/tuning/{model}/best_hyperparameters.json` only to populate the
  selected-hyperparameter table.
- Did not retrain models, rerun tuning or modify model code/configs.
- Did not use stale `outputs/metrics/static_model_comparison.csv`.

### Outputs Created

- `outputs/metrics/final_static_model_comparison.csv`.
- `outputs/thesis_tables/static/static_final_test_comparison.{csv,tex}`.
- `outputs/thesis_tables/static/static_horizon_c_index.{csv,tex}`.
- `outputs/thesis_tables/static/static_probabilistic_metrics.{csv,tex}`.
- `outputs/thesis_tables/static/static_selected_hyperparameters.{csv,tex}`.
- `outputs/thesis_tables/static/static_per_seed_results.{csv,tex}`.
- `outputs/figures/static/static_ctd_antolini_comparison.png`.
- `outputs/figures/static/static_ibs_comparison.png`.
- `outputs/figures/static/static_ibll_nbll_comparison.png`.
- `outputs/figures/static/static_horizon_c_index.png`.
- `outputs/figures/static/static_discrimination_vs_calibration_summary.png`.

### Validation

- Confirmed DeepHit final table uses corrected tail-support selection
  `include_tail_category=true`.
- Confirmed final DeepHit IBS is near 0.1104, excluding the old pre-tail audit
  result.
- Confirmed CoxPH deterministic final seeds give zero standard deviation.
- Confirmed non-applicable scalar metrics are empty/NA in the consolidated CSV.

### Documentation Updates

- Added `EXP-008` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/TODO.md` to mark final static table/figure consolidation as
  completed.
- No methodology or execution instructions changed, so `docs/DECISIONS.md` and
  `docs/REPRODUCIBILITY.md` were not updated.

## 2026-06-10 — Static Survival Curve Example Feasibility Check

### Purpose

Check whether final PCHazard and DeepHit prediction artifacts allow rigorous
selection of individual test patients for example survival-curve figures
without rerunning prediction.

### Findings

- `outputs/final_static/pchazard/seed_42/predictions/pchazard_predictions.parquet`
  and
  `outputs/final_static/deephit/seed_42/predictions/deephit_predictions.parquet`
  contain `patientunitstayid`, `time_to_event`, `observed_event`, `split` and
  `risk_score`.
- Filtering `split == "test"` gives 18,701 rows for both models, matching
  `data/processed/static/test_static.parquet`.
- The test prediction row order exactly matches `test_static.parquet`, and the
  survival-curve CSV columns `0` to `18700` correspond to that same test order.
- PCHazard `risk_score` matches `1 - S(10)` from the curve CSV up to numerical
  tolerance; DeepHit `risk_score` exactly matches `1 - survival_at_10.00d`.
- Therefore, example survival curves can be selected reproducibly by risk
  percentile, event/censoring status and observed time without retraining or
  regenerating predictions.

### Documentation Updates

- Appended this session note only.
- No experiment, methodology or priority changed, so `docs/EXPERIMENT_LOG.md`,
  `docs/DECISIONS.md`, `docs/REPRODUCIBILITY.md` and `docs/TODO.md` were not
  updated.

## 2026-06-10 — PCHazard vs DeepHit Example Survival Curves

### Purpose

Generate a thesis-ready example survival-curve figure comparing PCHazard and
DeepHit for reproducibly selected test patients.

### Selection Rule

- Used final seed-42 test predictions from PCHazard and DeepHit.
- Computed the average final risk score between both models for each test
  patient.
- Selected the patients closest to the 10th, 50th and 90th percentiles of the
  average risk distribution.
- Used the same selected patients and curve columns for both models.

### Outputs Created

- `outputs/figures/static/example_survival_curves_pchazard_deephit.png`.
- `outputs/thesis_tables/static/example_survival_curve_patients.csv`.
- `outputs/thesis_tables/static/example_survival_curve_patients.tex`.

### Notes

- The selected low-, medium- and high-risk examples are all censored cases under
  the automatic percentile rule, so the figure should be described as a
  risk-stratified prediction example rather than an event/censoring contrast.
- No models were retrained and no predictions were regenerated.

## 2026-06-10 — Event-Only PCHazard vs DeepHit Example Curves

### Purpose

Generate a companion survival-curve figure using only test patients with an
observed event, so it can be compared against the risk-percentile example that
selected censored patients.

### Selection Rule

- Filtered final seed-42 test predictions to `observed_event == 1`.
- Computed the average final risk score between PCHazard and DeepHit.
- Selected the event patients closest to the 10th, 50th and 90th percentiles of
  average risk among event patients.
- Regenerated both example-curve figures with a common y-axis range `[0, 1]`.

### Outputs Created

- `outputs/figures/static/example_event_survival_curves_pchazard_deephit.png`.
- `outputs/thesis_tables/static/example_event_survival_curve_patients.csv`.
- `outputs/thesis_tables/static/example_event_survival_curve_patients.tex`.

### Notes

- The event-only selection contains 2,228 eligible test event patients.
- The selected examples have observed event times of approximately 3.95, 1.52
  and 1.63 days for low-, medium- and high-risk event strata respectively.
- No models were retrained and no predictions were regenerated.

## 2026-06-11 — static_72h_pycox Pipeline Implementation

### Purpose

Implement a new isolated static pipeline for the revised 72-hour methodology
without modifying or replacing the previous static pipeline.

### Methodological Inputs Reviewed

- `TFG/Nueva_version_experimento.md`.
- `src/models_references/DySurv/Models/Results/Static Benchmarks MIMIC-IV.ipynb`.
- Current repository instructions and static pipeline documentation.

### Notebook Findings

- The DySurv static notebook loads `preprocessed_labels.csv` and
  `preprocessed_flat.csv`.
- It multiplies `actualiculos` by 24, drops rows with `actualiculos > 240`,
  drops `nullheight`, standardizes `age`, `height` and `weight`, leaves several
  binary/one-hot columns unchanged, and evaluates with pycox `EvalSurv`.
- It trains several pycox models, including LogisticHazard, DeepHitSingle,
  neural CoxPH and PCHazard.
- Its split logic is weak for the TFG setting and may overlap train/test
  because validation is sampled again from the full label set.
- It removes long survivors beyond the horizon, which is not used in the new
  TFG methodology.

### Changes Made

- Added `configs/static_72h_data.yaml`.
- Added `configs/static_72h_tuning.yaml`.
- Added `configs/static_72h_evaluation.yaml`.
- Added `scripts/build_static_72h_data.py`.
- Added `scripts/tune_static_72h_models.py`.
- Added `scripts/run_final_static_72h_seeds.py`.
- Added `scripts/evaluate_static_72h_models.py`.
- Added `src/data/static_72h_dataset.py`.
- Added `src/evaluation/static_72h_metrics.py`.
- Added `src/models/static_72h_pycox.py`.
- Added `tests/test_static_72h_pipeline.py`.
- Updated `.gitignore` to ignore generated `outputs/static_72h/`.

### Implemented Models

- Kaplan-Meier via lifelines.
- CoxPH via lifelines.
- DeepSurv-style neural CoxPH via pycox `CoxPH`.
- LogisticHazard via pycox.
- PCHazard via pycox.
- DeepHitSingle via pycox.

### Safeguards

- New outputs are isolated under `outputs/static_72h/`.
- New processed data are isolated under `data/processed/static_72h/`.
- Tuning evaluates only train and validation splits and sets
  `allow_test_metrics: false`.
- Tuning catches failed candidates and records them as failed rather than
  stopping the whole search.
- Final evaluation enforces exactly seeds `42`, `123` and `2026`.
- Preprocessing is fitted only on train and applied to validation/test.
- Long survivors after the 10-day post-72h horizon are censored at 10 days,
  not removed.

### Validation

- Ran
  `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_static_72h_pipeline.py`;
  result: 4 passed.
- Ran
  `C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_static_72h_models.py --config configs/static_72h_tuning.yaml --models coxph deephit_single --dry-run --max-runs 3`;
  result: planned isolated tuning runs under `outputs/static_72h/tuning/`.
- Ran py_compile for the new modules/scripts; result: passed.
- Ran help checks for final and evaluation scripts; result: CLIs loaded.
- Ran `git diff --check`; result: no whitespace errors, only CRLF warnings.

### Documentation Updates

- Added `DEC-009` to `docs/DECISIONS.md`.
- Updated `docs/REPRODUCIBILITY.md` with static_72h commands and outputs.
- Updated `docs/TODO.md` with remaining static_72h execution tasks.
- Did not update `docs/EXPERIMENT_LOG.md` because no real data build, tuning or
  final model experiment was run.

### Next Recommended Action

Run `scripts/build_static_72h_data.py` on the real MIMIC-derived inputs, inspect
the dataset summary, then launch validation-only tuning with `--dry-run` and a
small `--max-runs` smoke before full tuning.

## 2026-06-12 — static_72h_pycox DeepHit/PCHazard Audit Fixes

### Purpose

Audit the new `static_72h_pycox` pipeline before final 3-seed evaluation, with
special focus on pycox DeepHitSingle tail behavior, PCHazard interpolation,
discrete-time cuts, evaluation grids and test-lock discipline.

### Changes Made

- Added a per-split 100-point integration grid for IBS and IBLL/NBLL.
- Kept daily `horizon_times: [1, ..., 10]` for horizon C-index only.
- Added PCHazard `sub=10` before `predict_surv_df`, matching the DySurv static
  notebook.
- Added audit outputs under `outputs/static_72h/audit/` for DeepHitSingle tail
  checks, PCHazard checks, discrete cuts, grids and survival-curve sanity.

### Audit Results

- DeepHitSingle uses pycox native tail behavior; no manual tail category was
  added.
- DeepHitSingle validation survival at 10 days remained positive and
  heterogeneous: min 0.00949, mean 0.66919, max 0.96999, share below `1e-6` 0.
- PCHazard validation Antolini Ctd improved from stale pre-fix 0.40349 to
  0.64926 with `sub=10`; mean horizon C-index was 0.69561.
- PCHazard validation survival curves were monotone and finite, with mean
  survival at 10 days 0.70618.
- LogisticHazard interpolation produced tiny numerical sanity flags
  (`max_survival` just above 1 and a few monotonicity violations around
  numerical tolerance), not a DeepHit/PCHazard blocker.

### Validation

- Ran `py_compile` for modified static_72h modules/scripts; result passed.
- Ran `pytest tests/test_static_72h_pipeline.py`; result 4 passed.
- Ran validation-only audit recalculation for LogisticHazard, PCHazard and
  DeepHitSingle using existing selected hyperparameters; no test split was used.

### Caveat

- A broader audit run including DeepSurv stopped during survival prediction
  because pycox CoxPH attempted to allocate approximately 2.20 GiB for a dense
  train+validation survival matrix.
- Existing PCHazard tuning outputs before this audit are stale and should not be
  used for final selection.

### Documentation Updates

- Added `DEC-010` to `docs/DECISIONS.md`.
- Added `EXP-009` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/REPRODUCIBILITY.md` with static_72h audit outputs and grid
  semantics.
- Updated `docs/TODO.md` to require rerunning validation-only tuning before
  final 3-seed evaluation.

## 2026-06-12 — static_72h Explicit Daily Cuts

### Purpose

Apply the requested minimal change so pycox LogisticHazard and DeepHitSingle use
explicit daily cuts `[0, 1, ..., 10]`, matching the desired 10-day support.

### Changes Made

- `src/models/static_72h_pycox.py` now passes `model_cfg["cuts"]` to
  `label_transform` when present, otherwise preserving the previous
  `num_durations` behavior.
- `configs/static_72h_tuning.yaml` now sets daily cuts for LogisticHazard and
  DeepHitSingle only.

### Validation

- Ran `py_compile` on `src/models/static_72h_pycox.py` and
  `scripts/tune_static_72h_models.py`; result passed.
- Ran dry-run tuning planning for LogisticHazard and DeepHitSingle with
  `--max-runs 2`; result passed.

### Notes

- No model training or final evaluation was run.
- Existing tuning outputs for LogisticHazard and DeepHitSingle are now stale
  because their discretization changed.

## 2026-06-12 — Dynamic 72h Dataset Implementation Discussion

### Purpose

Discuss how to implement the dynamic 72-hour dataset for the new experiment,
using the `static_72h_pycox` cohort/splits as the fixed reference and reviewing
the DySurv/XMI-ICU MIMIC-IV preprocessing logic.

### Context Reviewed

- `AGENTS.md`, `SESSION_NOTES.md`, `docs/TODO.md` and
  `configs/static_72h_data.yaml`.
- `src/models_references/DySurv/Models/Results/Static Benchmarks MIMIC-IV.ipynb`.
- `src/models_references/XMI-ICU/MIMIC_IV-preprocessing/`.
- `src/models_references/XMI-ICU/eICU_preprocessing/`.
- `src/data/static_72h_dataset.py`.
- Existing processed artifacts under `data/processed/static_72h/` and
  `data/processed/mimic_extraction/`.
- Raw MIMIC-IV table headers and file names under `data/raw/mimic-iv-3.1/icu`
  and `data/raw/mimic-iv-3.1/hosp`; no full raw tables were scanned.

### Findings

- The dynamic dataset should be derived from the existing static 72h IDs,
  splits and targets, not by creating a new cohort or split.
- Existing `data/processed/mimic_extraction/` already contains
  `timeseries.csv`, `timeserieslab.csv` and a large
  `preprocessed_timeseries.csv`, so raw extraction from `chartevents` and
  `labevents` is probably unnecessary for the first implementation.
- The original MIMIC-IV reference selects common labs and chart events,
  converts timestamps to minute offsets from ICU admission, pivots variables,
  resamples hourly, creates masks, forward-fills within patient and fills
  remaining missing values.
- For the new methodology, the main adaptation is to clip strictly to
  `0 <= time < 72h`, align to hours `0..71`, fit imputation/scaling on train
  only and save tensors plus audit outputs.

### Documentation Updates

- This session note only. No code/config/data/output changes were made in this
  discussion turn.

## 2026-06-12 — PCHazard LabTrans Fix

### Purpose

Fix the `static_72h_pycox` PCHazard tuning failure caused by passing explicit
daily cuts to `PCHazard.label_transform`, which left `LabTransPCHazard` without
the internal `duc` attribute.

### Changes Made

- Removed explicit `cuts` from the PCHazard grid in `configs/static_72h_tuning.yaml`.
- Added `sub: [10]` to the PCHazard grid so prediction interpolation remains
  explicit.
- Updated `src/models/static_72h_pycox.py` so PCHazard always calls
  `label_transform(num_durations)` while LogisticHazard and DeepHitSingle can
  still use explicit daily cuts.

### Validation

- Ran `py_compile` on `src/models/static_72h_pycox.py` and
  `scripts/tune_static_72h_models.py`; result passed.
- Ran a direct PCHazard `label_transform(10).fit_transform(...)` check; result
  had `has_duc=True` and cuts `[0.0, 1.0, ..., 10.0]`.
- Ran `scripts/tune_static_72h_models.py --config configs/static_72h_tuning.yaml
  --models pchazard --dry-run --max-runs 1`; result passed.

### Notes

- No model training or final evaluation was run.
- Previous failed PCHazard tuning outputs should be rerun after this fix.

## 2026-06-12 — LogisticHazard Explicit Cuts Warning Fix

### Purpose

Remove the pycox warning raised by passing explicit cuts to
`LogisticHazard.label_transform` while preserving daily cuts `[0.0, ..., 10.0]`.

### Changes Made

- Updated `configs/static_72h_tuning.yaml` so LogisticHazard uses
  `num_durations: [11]` and no explicit `cuts`.

### Validation

- Verified `LogisticHazard.label_transform(11).fit_transform(...)` produces
  `out_features=11` and cuts `[0.0, 1.0, ..., 10.0]`.
- Parsed `configs/static_72h_tuning.yaml`; result passed.
- Ran `scripts/tune_static_72h_models.py --config configs/static_72h_tuning.yaml
  --models logistic_hazard --dry-run --max-runs 1`; result passed.

### Notes

- No model training or final evaluation was run.
- Previous LogisticHazard tuning outputs are stale because the discretization
  configuration changed from explicit cuts to equivalent `num_durations=11`.

## 2026-06-12 — Dynamic 72h Dataset Build

### Purpose

Implement and run the `dynamic_72h` dataset builder for the new 72-hour
methodology, using the `static_72h_pycox` cohort, splits and targets as the
fixed reference.

### Changes Made

- Added `configs/dynamic_72h_data.yaml`.
- Added `src/data/dynamic_72h_dataset.py`.
- Added `scripts/build_dynamic_72h_data.py`.
- Added focused tests in `tests/test_dynamic_72h_dataset.py`.
- Added `outputs/dynamic_72h/` to `.gitignore`.

### Dataset Rules Implemented

- Inputs come from `data/processed/static_72h/{train,val,test}_static_72h.parquet`.
- Temporal sources are `timeseries.csv` and `timeserieslab.csv` under
  `data/processed/mimic_extraction/`.
- Only measurements with `0 <= offset_minutes < 4320` are used.
- Measurements are binned into hours `0..71`.
- Repeated patient-feature-hour rows keep the last measurement in that hour.
- Temporal feature selection uses train-patient coverage only, threshold 5%.
- Missingness masks are created before imputation.
- Imputation uses forward-fill within patient/feature, then train medians.
- Scaling uses train-only p05/p95 robust scaling.

### Commands Run

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_dynamic_72h_dataset.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile src/data/dynamic_72h_dataset.py scripts/build_dynamic_72h_data.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/build_dynamic_72h_data.py --config configs/dynamic_72h_data.yaml --dry-run --sample-size 2
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/build_dynamic_72h_data.py --config configs/dynamic_72h_data.yaml --force
```

### Results

- Unit tests: 2 passed.
- `py_compile`: passed.
- Full build completed.
- Selected temporal features: 146.
- Static features: 28.
- Train arrays: `X_seq=(18706, 72, 146)`, `M_seq=(18706, 72, 146)`,
  `X_static=(18706, 28)`, event rate 0.1369.
- Validation arrays: `X_seq=(6236, 72, 146)`, `M_seq=(6236, 72, 146)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Test arrays: `X_seq=(6236, 72, 146)`, `M_seq=(6236, 72, 146)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Used temporal offsets ranged from 0 to 4319 minutes; no timestamp at or
  beyond 72h was used.
- Raw observed temporal fraction before imputation: train 0.1135, validation
  0.1135, test 0.1127.
- Every train/validation patient and 6234/6236 test patients had at least one
  selected temporal measurement.

### Outputs

- Dataset arrays and metadata: `data/processed/dynamic_72h/`.
- Audit outputs: `outputs/dynamic_72h/audit/`.

### Documentation Updates

- Added `DEC-011` to `docs/DECISIONS.md`.
- Added `EXP-010` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/REPRODUCIBILITY.md` with dynamic 72h build commands and
  outputs.
- Updated `docs/TODO.md` with completed dynamic dataset build and pending
  dynamic-model work.

### Next Recommended Action

Adapt and smoke-test DySurv and Dynamic-DeepHit loaders against
`data/processed/dynamic_72h/`. Decide whether `delta_seq` is required before
launching dynamic model training.

## 2026-06-12 — DySurv-Compatible Dynamic Feature Subset

### Purpose

Create a fast reduced dynamic dataset for the first dynamic-model training
attempt, keeping only temporal columns that map to the DySurv reference
time-series variable table.

### Changes Made

- Added `scripts/filter_dynamic_72h_dysurv_features.py`.
- Created `data/processed/dynamic_72h_dysurv_features/` from the existing
  `data/processed/dynamic_72h/` arrays.
- The full 146-feature `dynamic_72h` dataset was left unchanged.

### Method

- Loaded the existing `train/val/test_dynamic_72h.npz` files.
- Selected DySurv-compatible columns from `X_seq` and `M_seq`.
- Preserved `patient_ids`, `X_static`, `duration_eval_days`,
  `duration_rel_days` and `event_eval` unchanged.
- Did not rescan the large temporal CSVs.
- Did not refit imputation or scaling.

### Commands Run

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile scripts/filter_dynamic_72h_dysurv_features.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/filter_dynamic_72h_dysurv_features.py --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -c "<shape/NaN/mask validation>"
```

### Results

- Source temporal features: 146.
- Selected DySurv-compatible temporal features: 76.
- Removed temporal features: 70.
- Missing DySurv table variables: `ALT`, `Bilirubin`, `AST`,
  `Alkaline Phosphatase`.
- Train arrays: `X_seq=(18706, 72, 76)`, `M_seq=(18706, 72, 76)`,
  `X_static=(18706, 28)`, event rate 0.1369.
- Validation arrays: `X_seq=(6236, 72, 76)`, `M_seq=(6236, 72, 76)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Test arrays: `X_seq=(6236, 72, 76)`, `M_seq=(6236, 72, 76)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Validation confirmed no NaNs in `X_seq` and binary `M_seq` values.

### Documentation Updates

- Added `DEC-012` to `docs/DECISIONS.md`.
- Added `EXP-011` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/REPRODUCIBILITY.md` with the subset command and outputs.
- Updated `docs/TODO.md`.

### Next Recommended Action

Use `data/processed/dynamic_72h_dysurv_features/` for the first DySurv-style
loader/training smoke test, while keeping the full `dynamic_72h` dataset for
later comparison.

## 2026-06-12 — dynamic_72h_dysurv_features Additional Reduction

### Purpose

Overwrite `data/processed/dynamic_72h_dysurv_features/` by removing 15
additional chart-derived variables requested for the first dynamic-model
training pass.

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

### Method

- Loaded the existing 76-feature `dynamic_72h_dysurv_features` arrays.
- Sliced `X_seq` and `M_seq` in place to remove the requested columns.
- Rewrote `train/val/test_dynamic_72h.npz`,
  `temporal_feature_columns.json` and
  `dynamic_72h_dysurv_feature_summary.json`.
- Kept patient IDs, static features, durations and events unchanged.

### Results

- Previous subset features: 76.
- Removed features: 15.
- Current subset features: 61.
- Train arrays: `X_seq=(18706, 72, 61)`, `M_seq=(18706, 72, 61)`,
  `X_static=(18706, 28)`, event rate 0.1369.
- Validation arrays: `X_seq=(6236, 72, 61)`, `M_seq=(6236, 72, 61)`,
  `X_static=(6236, 28)`, event rate 0.1369.
- Test arrays: `X_seq=(6236, 72, 61)`, `M_seq=(6236, 72, 61)`,
  `X_static=(6236, 28)`, event rate 0.1369.

### Validation

- Verified all requested features are absent from
  `temporal_feature_columns.json`.
- Verified no NaNs in `X_seq`.
- Verified `M_seq` remains binary.

### Documentation Updates

- Added `DEC-013` to `docs/DECISIONS.md`.
- Added `EXP-012` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/REPRODUCIBILITY.md`.
- Updated `docs/TODO.md`.

## 2026-06-12 — Dynamic 72h Model Pipeline Smoke

### Purpose

Implement the first isolated dynamic 72h model layer for DySurv and
Dynamic-DeepHit, using the 61-feature
`data/processed/dynamic_72h_dysurv_features/` dataset without changing static
pipelines or creating new splits.

### Changes Made

- Added `src/models/dynamic_72h/` with data loading, discretization, losses,
  DySurv, Dynamic-DeepHit, prediction helpers and train entrypoint.
- Added `src/evaluation/dynamic_72h_metrics.py`.
- Added `configs/dynamic_72h_tuning.yaml` and `configs/dynamic_72h_final.yaml`.
- Added `scripts/tune_dynamic_72h_models.py`.
- Added `scripts/run_final_dynamic_72h_seeds.py`.
- Added `scripts/evaluate_dynamic_72h_models.py`.
- Added `tests/test_dynamic_72h_models.py`.

### Method

- Used saved `train/val/test_dynamic_72h.npz` splits directly.
- Default model input mode is `values_plus_mask_plus_static`.
- Current model input shape is `[N, 72, 150]`: 61 values, 61 masks and 28
  repeated static features.
- Target discretization uses daily cuts `[0, 1, ..., 10]` and indices `0..9`.
- Tuning uses train and validation only; test is disabled until final-seed
  runs.

### Commands Run

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile src/models/dynamic_72h/common.py src/models/dynamic_72h/discretization.py src/models/dynamic_72h/data.py src/models/dynamic_72h/losses.py src/models/dynamic_72h/predict.py src/models/dynamic_72h/dysurv.py src/models/dynamic_72h/dynamic_deephit.py src/models/dynamic_72h/train.py src/evaluation/dynamic_72h_metrics.py scripts/tune_dynamic_72h_models.py scripts/run_final_dynamic_72h_seeds.py scripts/evaluate_dynamic_72h_models.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_dynamic_72h_models.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv dynamic_deephit --dry-run --max-runs 2 --sample-size 32 --device cpu
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv dynamic_deephit --max-runs 2 --sample-size 128 --device cpu --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dynamic_deephit --max-runs 1 --sample-size 128 --device cpu --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/run_final_dynamic_72h_seeds.py --config configs/dynamic_72h_final.yaml --model dysurv dynamic_deephit --dry-run --sample-size 32 --device cpu
```

### Smoke Results

- Unit tests: 2 passed.
- `py_compile`: passed.
- DySurv smoke validation Ctd Antolini: 0.6367.
- DySurv smoke validation IBS: 0.6475.
- DySurv smoke validation IBLL/NBLL: 2.5545.
- DySurv smoke mean horizon C-index: 0.6407.
- Dynamic-DeepHit smoke validation Ctd Antolini: 0.7543.
- Dynamic-DeepHit smoke validation IBS: 0.1385.
- Dynamic-DeepHit smoke validation IBLL/NBLL: 0.4416.
- Dynamic-DeepHit smoke mean horizon C-index: 0.7741.
- No test metrics were recorded during tuning.

### Audits

- Split overlap checks passed.
- Target bin indices are in `[0, 9]`.
- Prediction sanity checks passed: no NaNs, values in `[0, 1]`, monotone
  non-increasing survival curves.
- Dynamic-DeepHit PMF sums were approximately 1, CIF was non-decreasing and
  `S(10)` was not forced to zero.

### Caveats

- Smoke used only `sample_size=128`, so metrics are not final evidence.
- DySurv calibration/error metrics were poor in this small smoke and should be
  inspected before final 3-seed evaluation.
- Dynamic-DeepHit uses an internal tail category by default to avoid forced
  zero survival at 10 days; this is an adaptation to the 10-day support.

### Documentation Updates

- Added `DEC-014` to `docs/DECISIONS.md`.
- Added `EXP-013` to `docs/EXPERIMENT_LOG.md`.
- Updated `docs/REPRODUCIBILITY.md` with dynamic model commands.
- Updated `docs/TODO.md`.

### Next Recommended Action

Run full validation-only dynamic tuning, inspect calibration/survival curves
and only then launch final three-seed dynamic evaluation.

## 2026-06-12 — Expanded Dynamic 72h Tuning Grid

### Purpose

Add the approved DySurv and Dynamic-DeepHit hyperparameter combinations to the
dynamic_72h validation-only tuning config using notation that is robust with
the current implementation.

### Changes Made

- Updated `configs/dynamic_72h_tuning.yaml` with the approved DySurv grid.
- Updated `configs/dynamic_72h_tuning.yaml` with the approved
  Dynamic-DeepHit grid.
- Added tuning-script normalization so DySurv `loss_weights` entries are
  expanded to `w_surv`, `w_recon` and `w_kl` before training.
- Kept Dynamic-DeepHit validation for `alpha + beta <= 1`.
- Propagated Dynamic-DeepHit `num_durations` from config instead of relying on
  hardcoded internal slices.

### Validation

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile scripts/tune_dynamic_72h_models.py src/models/dynamic_72h/train.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_dynamic_72h_models.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dysurv --dry-run --max-runs 3 --sample-size 16 --device cpu
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dynamic_72h_models.py --config configs/dynamic_72h_tuning.yaml --model dynamic_deephit --dry-run --max-runs 2 --sample-size 16 --device cpu
```

### Results

- `py_compile`: passed.
- `tests/test_dynamic_72h_models.py`: 2 passed.
- DySurv dry-run planned successfully.
- Dynamic-DeepHit dry-run planned successfully.
- Expanded grid sizes: 384 DySurv candidates and 512 Dynamic-DeepHit
  candidates.
- DySurv `loss_weights` normalization was checked on the first expanded
  candidate.
- No model training, final evaluation or test metrics were run.

### Documentation Updates

- Added `DEC-015` to `docs/DECISIONS.md`.
- Updated `docs/REPRODUCIBILITY.md` with expanded dynamic grid sizes and staged
  execution guidance.
- Did not update `docs/EXPERIMENT_LOG.md` because no experiment was trained.
- Did not update `docs/TODO.md` because priorities did not change.

### Next Recommended Action

Launch full validation-only dynamic tuning on GPU, or run staged chunks with
`--max-runs` first to estimate runtime.
