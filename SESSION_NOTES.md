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

## 2026-06-14 - DySurv tuning aggregate correction

### Changes

- Updated `outputs/dynamic_72h_bis/tuning/dysurv/tuning_results.csv` to replace
  the stale smoke-run `dysurv_cfg_001` row with the selected
  `dysurv_cfg_032` result reconstructed from its tuning metrics and config
  snapshot.
- The corrected row records validation Ctd Antolini `0.782255`, mean horizon
  C-index `0.797566`, IBS `0.142518` and IBLL/NBLL `0.453163`.
- Corrected artifact paths to use `outputs/dynamic_72h_bis`.

### Documentation Updates

- Appended this session note only.
- No experiment was run and no methodology, priorities or execution commands
  changed, so `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md`, `docs/TODO.md` and
  `docs/REPRODUCIBILITY.md` were not updated.

## 2026-06-14 - Full DySurv-faithful results analysis

### Scope

- Audited the completed 16-candidate tuning and final seeds `42`, `123` and
  `2026` under `outputs/dysurv_faithful_72h`.
- Reviewed selection metadata, final summaries, per-seed/per-horizon metrics,
  collapse diagnostics, leakage checks and full/example survival predictions.

### Findings

- Selected config: `dysurv_faithful_cfg_002`; all 16 candidates and all three
  final seeds were non-collapsed.
- Final mean test Ctd `0.777839`, mean horizon C-index `0.776494`, IBS
  `0.251928`, and IBLL/NBLL `0.706178`.
- Test discrimination was stable, whereas IBS/IBLL showed material seed
  variability.
- Full predictions were individualized, bounded and monotone, but mean 10-day
  predicted risk (`0.813`, `0.717`, `0.696`) substantially exceeded the
  observed event indicator rate (`0.137`), indicating overprediction and poor
  absolute calibration.
- The root `dysurv_faithful_audit_report.md` remains stale in its counts and
  still reports zero full tuning/final runs; raw summaries and per-seed files
  were used as source of truth.

### Documentation Updates

- Added EXP-016 to `docs/EXPERIMENT_LOG.md`.
- Marked faithful tuning, review and final seeds complete in `docs/TODO.md`.
- No code, model outputs, configs or methodology were changed, so
  `docs/DECISIONS.md` and `docs/REPRODUCIBILITY.md` were not updated.

## 2026-06-14 - DySurv-faithful tuning resume behavior check

### Finding

- Inspected `scripts/tune_dysurv_faithful_72h.py` before extending the loss
  weight grid.
- The current script does not implement resume/skip behavior: an existing run
  directory is only logged and is trained again, and `tuning_results.csv` is
  rewritten from the rows produced by the current invocation.
- Adding a third `loss_weights` option can also change positional config IDs,
  so completed candidates must be matched by their normalized hyperparameter
  JSON rather than only by `config_id`.

### Recommendation

- Add an explicit resume mode that loads the existing CSV, skips successful
  hyperparameter signatures, assigns new stable IDs after the current maximum,
  appends new rows and recomputes `best_hyperparameters.json` over all rows.
- Do not launch the extended grid with the current script because it would
  retrain and potentially overwrite existing results.

### Documentation Updates

- Appended this inspection note only. No experiment or code/config change was
  performed, so no other documentation file required an update.

## 2026-06-14 - DySurv-faithful resume mode implementation

### Changes

- Added explicit `--resume` support to
  `scripts/tune_dysurv_faithful_72h.py`.
- Completed candidates are matched by normalized hyperparameter JSON, so grid
  ordering changes do not cause retraining.
- New candidates receive IDs after the maximum existing ID; new rows are
  appended logically to the existing results and selection is recomputed over
  the combined history.
- Failed candidates remain eligible for retry, while `--resume` and `--force`
  are explicitly incompatible.
- Added a focused regression test for signature matching and ID continuation.

### Validation

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile scripts\tune_dysurv_faithful_72h.py tests\test_dysurv_faithful_72h.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests\test_dysurv_faithful_72h.py -q
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts\tune_dysurv_faithful_72h.py --config configs\dysurv_faithful_72h.yaml --device cpu --resume --dry-run
```

- Compilation passed.
- Focused tests: 8 passed, including an integration-style resume/append test.
- Resume dry-run skipped all 16 completed combinations and planned no runs.

### Documentation Updates

- Updated `docs/REPRODUCIBILITY.md` with the resume command and semantics.
- Did not update `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md` or
  `docs/TODO.md` because no experiment, methodological decision or priority
  change occurred.

## 2026-06-14 - DySurv-faithful train Kaplan-Meier curve

### Scope and Artifacts

- Computed the descriptive Kaplan-Meier curve directly from
  `data/processed/dysurv_faithful_72h/train_dynamic_72h.npz` using
  `duration_eval_days` and `event_eval`.
- Generated `outputs/figures/dysurv_faithful_72h/kaplan_meier_train.png` and
  `kaplan_meier_train.csv`.

### Findings

- Training cohort size: `18,706`; observed events: `2,561`.
- Kaplan-Meier survival at day 10: `0.696957`, corresponding to cumulative
  event risk approximately `0.303043` after accounting for censoring.
- The raw event-indicator mean `0.136908` must not be interpreted as the
  10-day cumulative event probability because many observations are censored.
- `duration_rel_days` was not used because the faithful prediction target is
  the truncated 0--10 day time measured after the 72-hour landmark.

### Documentation Updates

- Appended this session note only. No model experiment, code/config change or
  methodological decision was made.

## 2026-06-14 - Extended DySurv-faithful per-epoch diagnostics

### Changes

- Extended faithful DySurv epoch reporting without changing training or
  selection logic.
- Added `mean_risk10`, active latent units, KL-per-dimension summaries and
  individual `kl_dim_XX` columns to train/validation epoch diagnostics.
- The visible epoch log now includes validation Ctd, IBS, IBLL, risk10 mean/std,
  active units, total KL and collapse status.
- Defined active units as latent dimensions with between-patient
  `Var(mu_j) > 0.01`; added the threshold to config.

### Validation

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile src\models\dynamic_72h\train_dysurv_faithful.py tests\test_dysurv_faithful_72h.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests\test_dysurv_faithful_72h.py -q
```

- Compilation passed; focused tests: 9 passed.
- Added a focused test for risk mean, active-unit counting and per-dimension KL.

### Documentation Updates

- Added DEC-017 and updated `docs/REPRODUCIBILITY.md`.
- Did not update `docs/EXPERIMENT_LOG.md` or `docs/TODO.md` because no model
  experiment was run and priorities did not change.

### Next Recommended Action

Launch full validation-only dynamic tuning on GPU, or run staged chunks with
`--max-runs` first to estimate runtime.

## 2026-06-12 - DySurv dynamic_72h_bis tuning inspection

### Scope

- Inspected existing validation-only DySurv tuning outputs under
  `outputs/dynamic_72h_bis/tuning/dysurv`.
- No model code, configs, training outputs or metrics files were modified.

### Findings

- Found 86 DySurv candidate directories and 85 `metrics.json` files.
- `dysurv_cfg_086` has no metrics file in the inspected output tree.
- The root `tuning_results.csv` / `best_hyperparameters.json` under the
  inspected folder only summarize `dysurv_cfg_001`, so they should not be used
  as the complete ranking for this run.
- Reconstructed the ranking directly from per-candidate `metrics.json` files.
- Best validation configuration by the configured objective
  `validation_ctd_antolini` maximize, with `validation_ibll` minimize as
  tiebreaker, is `dysurv_cfg_032`.
- `dysurv_cfg_032` validation metrics: Ctd Antolini `0.782255`, mean horizon
  C-index `0.797566`, IBS `0.142518`, IBLL/NBLL `0.453163`.

### Documentation Updates

- Appended this session note only.
- Did not update `docs/EXPERIMENT_LOG.md` because no experiment was run.
- Did not update `docs/DECISIONS.md`, `docs/TODO.md` or
  `docs/REPRODUCIBILITY.md` because no methodology, priority or execution
  instruction changed.

## 2026-06-12 - DySurv final seed 123 diagnostic

### Scope

- Inspected user-provided final DySurv seed 123 artifacts from the Downloads
  folder:
  `tfg-javi-mates-survival-modeling_outputs_dynamic_72h_final_dysurv_seed_123`.
- Reviewed `metrics.json`, `horizon_c_index.csv`, `train_log.csv`,
  `config_snapshot.yaml` and example survival prediction CSVs.
- No code, configs or generated model outputs were modified.

### Findings

- The run used `dysurv_cfg_032` with seed `123`, phase `final`, and
  `include_test: true`.
- Train, validation and test horizon C-index values are exactly `0.5` for all
  horizons from 1 to 10 days.
- Ctd Antolini is reported as `0.0` for all splits.
- IBS/IBLL remain numerically reasonable, for example test IBS `0.144249` and
  test IBLL/NBLL `0.457200`.
- Example survival curves are identical across selected patients; `risk_at_10d`
  has a single value `0.2793454528`.
- Training loss did not explode; best validation loss occurred at epoch 77 with
  `val_loss_total = 0.430403`.
- The KL term collapsed to nearly zero at the best epoch, consistent with a
  seed-specific degenerate solution / latent collapse where the model predicts
  an almost population-level survival curve rather than patient-specific risk.

### Documentation Updates

- Appended this diagnostic note only.
- Did not update `docs/EXPERIMENT_LOG.md` because no experiment was run in this
  session.
- Did not update `docs/DECISIONS.md`, `docs/TODO.md` or
  `docs/REPRODUCIBILITY.md` because no project decision or execution protocol
  changed.

## 2026-06-12 - Dynamic 72h final result tables and survival curve figures

### Scope

- Inspected final dynamic outputs under `outputs/dynamic_72h_bis/final` for
  `dysurv` and `dynamic_deephit`.
- Summarized final test metrics from per-seed `metrics.json` files.
- Applied the user-requested DySurv aggregation rule: exclude seed `123` from
  DySurv mean/std because it was previously diagnosed as degenerate.
- Added a lightweight plotting script for existing dynamic survival curve
  example CSVs.

### Artifacts

- Added `scripts/plot_dynamic_72h_survival_curves.py`.
- Generated figures under `outputs/figures/dynamic_72h_bis`.
- Generated derived result tables under `outputs/thesis_tables/dynamic_72h_bis`.

### Findings

- DySurv final test summary, seeds `42` and `2026` only: Ctd Antolini mean
  `0.756796`, mean horizon C-index `0.770349`, IBS `0.143498`, IBLL/NBLL
  `0.454687`.
- Dynamic-DeepHit final test summary, seeds `42`, `123` and `2026`: Ctd
  Antolini mean `0.790712`, mean horizon C-index `0.796157`, IBS `0.126511`,
  IBLL/NBLL `0.395774`.
- `outputs/dynamic_72h_bis/tuning/dynamic_deephit/best_hyperparameters.json`
  appears stale relative to final outputs: it reports `dynamic_deephit_cfg_001`,
  while final runs use `dynamic_deephit_cfg_005`.

### Validation

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile scripts\plot_dynamic_72h_survival_curves.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts\plot_dynamic_72h_survival_curves.py --outputs-dir outputs\dynamic_72h_bis --figures-dir outputs\figures\dynamic_72h_bis --exclude dysurv:123
```

### Documentation Updates

- Appended this session note.
- Updated `docs/REPRODUCIBILITY.md` with the survival-curve plotting command.
- Did not update `docs/EXPERIMENT_LOG.md` because no model experiment was run.
- Did not update `docs/DECISIONS.md` because no new methodological decision was
  made beyond applying the user-requested exclusion rule.
- Did not update `docs/TODO.md` because priorities did not change.

## 2026-06-14 - DySurv near-constant prediction audit

### Scope

- Audited the original DySurv notebook, current architecture/loss/prediction,
  processed dynamic arrays and existing tuning/final artifacts.
- Ran read-only diagnostics and two controlled 64-patient CPU overfit checks.
- Did not modify model code, configs, datasets or existing model outputs.

### Findings

- No repeated-input, dataloader, batch-dimension or broadcasting bug was found.
- Seeds 42 and 2026 have very narrow test risk10 ranges (`0.003110` and
  `0.004192`); seed 123 is exactly constant.
- The test Kaplan-Meier marginal risk at 10 days is `0.300532`, close to the
  seed 42 predictions.
- KL and reconstruction diagnostics identify posterior collapse: final KL is
  near zero and decoder MSE is no better than a train-mean reconstruction.
- Reconstructing masks and repeated unscaled static features is a major
  adaptation problem; `hour` alone contributes about 59.75% of baseline MSE.
- Tiny controls prove that the architecture can respond to individual and
  perturbed inputs, especially under survival-only training.

### Artifacts

- Added `outputs/dynamic_72h/dysurv_audit_report.md`.
- Added `EXP-014` to `docs/EXPERIMENT_LOG.md`.
- Added blocking/fix tasks to `docs/TODO.md`.
- Did not update `docs/DECISIONS.md` because no implementation decision was
  approved.
- Did not update `docs/REPRODUCIBILITY.md` because execution instructions did
  not change.

### Recommendation

Do not interpret current DySurv final runs as individualized survival curves.
Add collapse diagnostics/checkpoints, correct the reconstruction target, and
rerun tiny-overfit, smoke validation and tuning before final comparison.

## 2026-06-14 - DySurv faithful 72h pipeline implementation

### Scope

- Reviewed the original DySurv notebook at
  `src/models_references/DySurv/Models/Results/DySurv.ipynb`.
- Added a new isolated faithful dataset/model/training/tuning/final/audit
  pipeline without modifying the previous DySurv implementation or outputs.

### Implementation

- Added train-only `ffill -> bfill -> residual median` temporal imputation.
- Standardized static inputs using train-only statistics.
- Removed masks from model input channels.
- Kept repeated standardized static variables as the primary configurable
  input mode.
- Added a reference-capacity LSTM/VAE model with latent dimension 20,
  `[294, 490, 294]` encoder/survival MLPs and a recurrent decoder.
- Reconstruction targets only the 61 temporal clinical variables.
- Added KL warm-up, per-epoch latent/risk diagnostics, collapse flags,
  collapse-aware epoch/config selection, best/last checkpoints, complete
  prediction parquet files and curve-example CSVs.
- Added a 16-candidate training/loss grid and exact final seeds 42, 123, 2026.

### Commands Run

```bash
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m py_compile src/data/dysurv_faithful_72h_dataset.py src/models/dynamic_72h/dysurv_faithful.py src/models/dynamic_72h/train_dysurv_faithful.py scripts/prepare_dysurv_faithful_72h_dataset.py scripts/tune_dysurv_faithful_72h.py scripts/run_final_dysurv_faithful_72h_seeds.py scripts/audit_dysurv_faithful_72h.py
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe -m pytest tests/test_dysurv_faithful_72h.py tests/test_dynamic_72h_models.py -q
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/prepare_dysurv_faithful_72h_dataset.py --config configs/dysurv_faithful_72h.yaml --force
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/audit_dysurv_faithful_72h.py --config configs/dysurv_faithful_72h.yaml --run-tiny-overfit --device cpu
C:\Users\Javi\miniconda3\envs\tfg-survival\python.exe scripts/tune_dysurv_faithful_72h.py --config configs/dysurv_faithful_72h.yaml --max-runs 1 --sample-size 128 --device cpu --force
```

### Results

- Tests: 8 passed across faithful and existing dynamic sanity suites.
- Prepared dataset: train `18706 x 72 x 61`, validation/test
  `6236 x 72 x 61`; all split and finite-value checks passed.
- Tiny-overfit: train survival loss `3.275400 -> 0.219514`, final train
  risk10 std `0.363450`, final validation risk10 std `0.388562`, no final
  collapse flag.
- Weighted smoke selected non-collapsed epoch 8: validation Ctd `0.566922`,
  IBS `0.281065`, IBLL `0.768921`, risk10 std `0.029566`.
- Pure metric-best epoch 14 had slightly higher Ctd `0.575526` but was
  correctly rejected as collapsed (`risk10` range `0.002368`).
- No test data were loaded during tiny-overfit or smoke tuning.
- Full 16-candidate tuning and final three-seed evaluation were not run.

### Documentation Updates

- Added `DEC-016` and `EXP-015`.
- Updated `docs/REPRODUCIBILITY.md` and `docs/TODO.md`.
- Did not modify `docs/PROJECT_HISTORY.md`.

### Next Recommended Action

Run the 16 validation-only candidates on GPU, audit the selected curves and
collapse diagnostics, then launch final seeds only if a stable non-collapsed
candidate is accepted.

## 2026-06-14 - DySurv faithful output tracking audit

### Scope

- Checked whether `outputs/dysurv_faithful_72h/` must be committed for the new
  faithful DySurv pipeline to run after cloning.
- Inspected only filenames, small summary files and code/config path references;
  no patient-level data or heavy artifacts were opened.

### Findings

- The pipeline implementation depends on the new config, scripts, source modules
  and test, not on previously generated output directories.
- Tuning regenerates `tuning_results.csv`, `best_hyperparameters.json`, per-run
  checkpoints, metrics and predictions under `outputs/dysurv_faithful_72h/`.
- The final-seed script reads `best_hyperparameters.json`, so full tuning must run
  first unless an accepted selection manifest is supplied separately.
- The current root `best_hyperparameters.json` has `status: not_run`; committing
  it would not enable final-seed execution.
- `outputs/dysurv_faithful_72h/` is currently untracked and is not explicitly
  covered by `.gitignore`, creating an accidental staging risk.

### Recommendation

- Do not commit the full output directory, checkpoints, predictions or generated
  datasets. Commit the pipeline code/config/test and regenerate outputs.
- Add `outputs/dysurv_faithful_72h/` to `.gitignore` before broad staging.
- If future execution must skip tuning, store only the reviewed selected
  hyperparameters in a small config-owned manifest rather than versioning the
  whole output tree.

### Documentation Updates

- Appended this audit note only.
- Did not update `docs/EXPERIMENT_LOG.md` because no experiment was run.
- Did not update `docs/DECISIONS.md`, `docs/TODO.md` or
  `docs/REPRODUCIBILITY.md` because no new implementation decision, priority or
  execution command was introduced.

## 2026-06-14 - Faithful DySurv curve plotting check

### Scope and Finding

- Checked the curve artifacts written by the faithful DySurv pipeline.
- Each completed final seed writes
  `outputs/dysurv_faithful_72h/final/seed_<seed>/predictions/test_curve_examples.csv`.
- The existing `scripts/plot_dynamic_72h_survival_curves.py` targets the older
  dynamic pipeline's two-file curve format and does not directly accept the
  faithful pipeline's row-based `*_curve_examples.csv` format.

### Documentation Updates

- Appended this note only; no experiment, technical decision, priority or
  repository execution protocol changed.

## 2026-06-14 - DySurv versus Dynamic-DeepHit performance diagnosis

### Scope

- Compared final dynamic results under `outputs/dynamic_72h_bis/final/` and the
  current faithful DySurv summary without retraining or changing model code.
- Reviewed the selected Dynamic-DeepHit configuration, training objective and
  input construction.

### Findings

- Dynamic-DeepHit is stable across seeds: test Ctd `0.7907 +/- 0.0012`, IBS
  `0.1265 +/- 0.0016`, IBLL `0.3958 +/- 0.0039` and mean horizon C-index
  `0.7962 +/- 0.0007`.
- The old DySurv three-seed aggregate is invalid as a normal variability
  summary because seed 123 collapsed (`Ctd=0`, mean horizon C-index `0.5`).
- Excluding seed 123 as previously agreed, old DySurv obtains mean test Ctd
  `0.7568`, mean horizon C-index `0.7703`, IBS `0.1435` and IBLL `0.4547`.
- Dynamic-DeepHit therefore retains a real but moderate advantage over the two
  usable old DySurv seeds: about `+0.0339` Ctd, `+0.0258` mean horizon C-index,
  `-0.0170` IBS and `-0.0589` IBLL.
- Dynamic-DeepHit directly combines PMF negative log-likelihood and pairwise
  ranking loss, has tail support, and has no variational KL bottleneck. DySurv
  must balance survival, reconstruction and KL terms and is susceptible to
  latent/prediction collapse.
- The comparison is not architecture-only: Dynamic-DeepHit uses
  `values_plus_mask_plus_static`, while faithful DySurv intentionally excludes
  measurement masks from model input. Informative ICU measurement patterns may
  contribute to Dynamic-DeepHit performance.
- Against faithful noncollapsed DySurv, the discrimination gap is smaller, but
  the calibration gap is much larger; faithful DySurv currently overpredicts
  risk and has seed-dependent probability scale.

### Documentation Updates

- Appended this diagnostic note only.
- No experiment was run and no technical decision, TODO priority or execution
  protocol changed, so the other project documents were not modified.

## 2026-06-14 - Dynamic-DeepHit and DySurv loss clarification

### Finding

- Dynamic-DeepHit represents event-time probability with a softmax PMF over 10
  daily bins plus an explicit tail category. Without that tail category, the
  softmax would force all probability mass into the evaluated horizon.
- DySurv does not use an explicit tail output because its LogisticHazard head
  predicts conditional hazards. Probability beyond day 10 is represented
  implicitly by the remaining survival probability
  `S(10) = product_t(1 - hazard_t)`.
- Therefore both models account for survival beyond the horizon, but through
  different parameterizations; explicit tail support alone does not establish
  that Dynamic-DeepHit is intrinsically better calibrated.
- Dynamic-DeepHit final loss is `0.4 * longitudinal MSE + 0.1 * ranking loss +
  0.5 * PMF NLL`. Faithful DySurv uses a weighted LogisticHazard NLL,
  reconstruction loss and variational KL divergence with KL warm-up.
- The pairwise ranking term directly rewards temporal risk ordering in
  Dynamic-DeepHit. DySurv has no explicit concordance/ranking term and must also
  preserve a regularized generative latent representation, explaining its
  greater optimization tension and collapse sensitivity.

### Documentation Updates

- Appended this clarification only; no experiment, implementation decision,
  TODO priority or execution protocol changed.

## 2026-06-14 - DySurv loss-weight guidance relative to Dynamic-DeepHit

### Guidance

- DySurv can emphasize temporal probability learning and trajectory structure
  through `w_surv` and `w_recon`, but its weights are not numerically comparable
  to Dynamic-DeepHit because the component losses have different raw scales.
- Dynamic-DeepHit's ranking component has no direct counterpart in DySurv;
  DySurv's KL term regularizes the latent distribution and must not be treated
  as an equivalent replacement.
- A simple conceptually close DySurv trial is `w_surv=0.55`, `w_recon=0.40`,
  `w_kl=0.05`, retaining KL warm-up. A more conservative alternative is
  `0.60/0.30/0.10`.
- `0.50/0.30/0.20` gives substantially more nominal importance to KL and is
  therefore less similar to Dynamic-DeepHit and more exposed to the collapse
  behaviour already observed.
- Comparison should inspect effective contributions `weight * component_loss`,
  not weights alone, together with Ctd, IBS/IBLL and collapse diagnostics.

### Documentation Updates

- Appended this guidance only; no experiment was run and no configuration or
  implementation was changed.

## 2026-06-15 - Faithful dynamic model results analysis

### Scope

- Inspected completed outputs for `outputs/dynamic_deephit_faithful_72h/`,
  `outputs/dysurv_faithful_72h/` and
  `outputs/dysurv_static_faithful_72h/`.
- Read configs, selected hyperparameters, tuning results, final seed summaries,
  per-seed metrics JSONs and complete test survival prediction parquets.
- No retraining, tuning reruns or model-code changes were performed.

### Main Findings

- Dynamic-DeepHit faithful selected `dynamic_deephit_faithful_cfg_002` and is
  the best final model overall: test Ctd `0.780743 +/- 0.003989`, mean horizon
  C-index `0.787014 +/- 0.005717`, IBS `0.121338 +/- 0.001455`, IBLL/NBLL
  `0.385481 +/- 0.004907`; no collapsed seeds.
- DySurv faithful selected `dysurv_faithful_cfg_002` and has similar scalar
  discrimination but much worse calibration: test Ctd `0.777839 +/- 0.002461`,
  mean horizon C-index `0.776494 +/- 0.008187`, IBS `0.251928 +/- 0.032800`,
  IBLL/NBLL `0.706178 +/- 0.083272`; no collapsed seeds.
- DySurv static faithful selected `dysurv_static_faithful_cfg_007` and is a
  weaker static-only baseline: test Ctd `0.683475 +/- 0.000998`, mean horizon
  C-index `0.682671 +/- 0.000665`, IBS `0.165585 +/- 0.020709`, IBLL/NBLL
  `0.499855 +/- 0.047991`; no collapsed seeds.
- The faithful dataset Kaplan-Meier risk at day 10 is about `0.303` train,
  `0.299` validation and `0.301` test. Dynamic-DeepHit final predictions have
  mean day-10 risk around `0.25-0.29`, while DySurv faithful predicts
  `0.70-0.81` and DySurv static `0.45-0.65`, explaining much of the IBS/IBLL
  gap.
- Dynamic-DeepHit tuning was very stable: all 16 candidates completed without
  collapse and validation Ctd ranged only `0.801221-0.805166`.
- DySurv faithful tuning contained one clearly collapsed metric-best candidate
  (`cfg_023`, `std_risk10=0.001069`, `range_risk10=0.007492`) and one near-flat
  but not flagged candidate (`cfg_017`, `std_risk10=0.005885`) with strong
  calibration metrics. The accepted noncollapsed candidate was `cfg_002`.
- DySurv static faithful tuning had no collapse flags. Configurations with
  longer KL warm-up often had better IBS/IBLL but not always the best Ctd.

### Documentation Updates

- Appended this analysis note only. No experiment was run, no technical
  decision was made, and no TODO or reproducibility command changed.
## 2026-06-15 - Dynamic-DeepHit faithful 72h pipeline implementation

### Scope

- Added a separate Dynamic-DeepHit pipeline that reuses
  `data/processed/dysurv_faithful_72h/` and does not overwrite old dynamic or
  faithful DySurv outputs.
- Preserved the reference recurrent embedding, longitudinal next-step head,
  attention, PMF NLL and ranking loss while adding explicit beyond-horizon tail
  support and the faithful input convention without mask channels.
- Added validation-only grid tuning, resume/force controls, collapse-aware
  selection, exact final seeds 42/123/2026, checkpoints, config snapshots,
  epoch diagnostics, complete patient predictions, curve examples and audit
  reporting.

### Validation

- Python compilation passed for all new model, training, script and test files.
- `pytest tests/test_dynamic_deephit_faithful_72h.py tests/test_dysurv_faithful_72h.py -q`:
  16 passed.
- `pytest tests/test_dynamic_deephit_faithful_72h.py tests/test_dynamic_72h_models.py tests/test_deephit_time_dependent_metrics.py -q`:
  11 passed.
- Dry-run expanded the intended 16-candidate grid and isolated smoke paths.
- One 128-patient smoke candidate completed without test evaluation: validation
  Ctd `0.836520`, IBS `0.126745`, IBLL `0.406526`, mean horizon C-index
  `0.863040`, `risk10_std=0.150010`, no collapse.
- Tiny-overfit reduced train PMF NLL from `1.189985` to `0.002844` and retained
  individualized risks. Its first report-generation attempt failed after
  training due to helper definition order; the script was fixed and the audit
  report regenerated without retraining.

### Next Action

- Run the full validation-only grid with `--resume`, review the selected
  candidate's curves/tail diagnostics, and only then run the three final seeds.

### Documentation Updates

- Added DEC-018 and EXP-017.
- Updated `docs/TODO.md` and `docs/REPRODUCIBILITY.md` with pending work and
  commands. `docs/PROJECT_HISTORY.md` was not modified.

## 2026-06-15 - DySurv static faithful 72h implementation

### Scope

- Reviewed the final `DySurv` sections in the GBSG, METABRIC, SUPPORT, NWTCO,
  SAC3 and SAC_ADMIN benchmark notebooks.
- Added an isolated static-only MLP-VAE DySurv pipeline using only `X_static`
  from the exact `dysurv_faithful_72h` split files.
- Added validation-only tuning/resume, collapse-aware selection, exact final
  seeds, config snapshots, best/last checkpoints, per-epoch diagnostics,
  complete predictions, curve examples, dataset identity hashes and audit
  reporting.

### Methodological Adaptation

- Preserved the shared `F -> 3F -> 5F -> 3F`, latent-20, static decoder and
  LogisticHazard structure.
- Kept decoder hidden activations disabled by default, matching the notebooks.
- Corrected repeated notebook defects: malformed encoder syntax, incompatible
  scalar/vector loss weights, stochastic evaluation prediction and
  batch-dependent KL scaling.
- Used deterministic `mu` for evaluation, per-patient mean KL and explicit
  survival/reconstruction/KL weights with warm-up.

### Commands and Results

- Python compilation passed for all six new code/config test targets.
- Focused tests initially found one overly saturated synthetic uniqueness
  assertion; the diagnostic was stopped earlier and the final suites passed:
  18 tests, then 20 tests including existing dynamic model compatibility.
- Dry-run expanded exactly 16 candidates.
- Smoke command: `python scripts/tune_dysurv_static_faithful_72h.py --config configs/dysurv_static_faithful_72h.yaml --max-runs 1 --sample-size 128 --device cpu --force`.
- Smoke validation: Ctd `0.690249`, mean horizon C-index `0.690247`, IBS
  `0.392552`, IBLL `1.092452`, `risk10_std=0.089190`, no collapse, no test.
- Tiny-overfit command: `python scripts/audit_dysurv_static_faithful_72h.py --config configs/dysurv_static_faithful_72h.yaml --run-tiny-overfit --device cpu`.
- Tiny-overfit train survival NLL `3.248966 -> 0.362316`, reconstruction MSE
  `1.026930 -> 0.982480`, final train `risk10_std=0.278752`, no selected
  collapse.

### Next Action

- Run the full 16-candidate validation grid, inspect reconstruction/curves and
  collapse diagnostics, then execute seeds 42, 123 and 2026 only after review.

### Documentation Updates

- Added DEC-019 and EXP-018; updated TODO and reproducibility commands.
- `docs/PROJECT_HISTORY.md` was not modified.

## 2026-06-15 - Final faithful prediction export and experiment log update

### Scope

- Inspected final outputs for Dynamic-DeepHit faithful, temporal DySurv
  faithful and static DySurv faithful.
- Confirmed that complete validation/test prediction parquet files already
  existed for seeds 42, 123 and 2026 in all three pipelines.
- Created standardized derivative files for every model/seed/split:
  `*_survival_curves.parquet` and `*_patient_predictions.csv`.

### Checks

- Verified 10 survival columns per curve file.
- Verified `risk10 = 1 - S(10)` for each patient.
- Verified unique patient IDs and matching patient order across the three
  pipelines for the same split and seed.
- Wrote `outputs/final_faithful_curve_export_audit.json`; no issues were
  reported.

### Results Logged

- Added EXP-019 to `docs/EXPERIMENT_LOG.md` with selected configs, seeds,
  aggregate test metrics, per-seed metrics and final interpretation notes.
- Dynamic-DeepHit faithful remains the best global model among the three final
  faithful runs.
- Temporal DySurv faithful has similar discrimination but much worse
  calibration; static DySurv faithful is weaker in discrimination.

### Documentation Updates

- Updated `docs/EXPERIMENT_LOG.md` and appended this session note.
- Did not update `docs/DECISIONS.md`, `docs/TODO.md` or
  `docs/REPRODUCIBILITY.md` because no methodological decision, priority change
  or new execution protocol was introduced.

## 2026-06-19 — Static time-since-admission flat-feature enrichment

### Purpose

Add a lightweight way to include elapsed hospital time before ICU admission as
a static covariate without rerunning the full direct MIMIC extraction or
time-series generation.

### Changes

- Added `scripts/add_time_since_admission_to_flat_features.py`.
- The script reads the existing `flat_features.csv`, joins raw `icustays` and
  `admissions`, preserves row order and writes
  `flat_features_with_time_since_admission.csv`.
- The script preserves signed raw differences and warns, rather than clamps, if
  ICU `intime` precedes hospital `admittime`.
- Updated `configs/static_72h_data.yaml` and `configs/static_data.yaml` to use
  the enriched flat-feature file and preprocess `time_since_admission_hours` as
  a numeric static feature.
- Updated `docs/REPRODUCIBILITY.md` with the enrichment command.
- Added DEC-021 documenting the data-design decision.

### Validation

- `python scripts/add_time_since_admission_to_flat_features.py` completed with
  93,502 rows and no missing new covariate values.
- The enriched file preserves the original `patientunitstayid` order and adds
  only `time_since_admission_hours`.
- The script reported 449 negative signed differences where ICU `intime`
  precedes hospital `admittime`; values were preserved rather than clamped.
- `py_compile` passed for the new script.
- Static datasets were not rebuilt in this task.

### Next Action

- Rebuild `static_72h` before rerunning static tuning so the new covariate is
  included in model inputs.
## 2026-06-19 - Project Manager planning: parametrizable landmark pipeline

### Context
- The user requested a documentation/planning-only technical PM assessment for generalizing the current 72h landmark survival modeling pipeline to `landmark_hours = 24, 48, 72`.
- Scope explicitly excludes code/config/model changes in this session.

### Work performed
- Reviewed repository governance instructions and current project documentation.
- Reviewed the documented 72h static, dynamic, DySurv-faithful, Dynamic-DeepHit-faithful and static-DySurv-faithful pipeline state.
- Inspected relevant configs, scripts and source modules outside excluded artifact folders to identify reusable components, hardcoded 72h assumptions and likely refactor boundaries.

### Outcome
- Produced an implementation plan for a single parametrizable landmark pipeline preserving the existing 72h pipeline as compatibility reference.
- No experiments were run.
- No source code, configs, model artifacts, data, outputs or checkpoints were modified.

## 2026-06-19 — Technical implementation: parametrizable landmark pipeline layer

### Scope

- Added a CLI-driven landmark layer for `landmark_hours` in `{24, 48, 72}`.
- Preserved existing 72h scripts/configs and kept them usable as compatibility
  entrypoints.
- Used the current DEC-021 data definition with
  `flat_features_with_time_since_admission.csv`; no rollback to older flat
  features was made.

### Implementation

- Added `src/utils/landmark.py` for landmark validation, path resolution,
  suffixes and config snapshots.
- Added generic scripts for static data, dynamic data, DySurv feature filtering,
  faithful dataset preparation, static tuning/final runs, and faithful
  tuning/final runs.
- Made static and dynamic dataset output suffixes configurable while preserving
  current 72h defaults.
- Made faithful dataset preparation and faithful loaders accept landmark-specific
  split filenames instead of requiring 72-step filenames.
- Added `config_used.yaml` snapshots at landmark/family output roots and
  per-run tuning roots where applicable.

### Validation

- `py_compile` passed for the new/modified landmark, data, training and script
  files.
- `pytest tests/test_landmark_pipeline.py tests/test_static_72h_pipeline.py tests/test_dynamic_72h_dataset.py -q`
  passed: 10 tests.
- Dry-runs passed for:
  - `scripts/tune_landmark_static_models.py --landmark-hours 72 --models kaplan_meier --dry-run --max-runs 1`
  - `scripts/tune_landmark_dysurv_faithful.py --landmark-hours 72 --dry-run --max-runs 1 --device cpu`
  - `scripts/tune_landmark_dynamic_deephit_faithful.py --landmark-hours 72 --dry-run --max-runs 1 --device cpu`
  - `scripts/tune_landmark_dysurv_static_faithful.py --landmark-hours 72 --dry-run --max-runs 1 --device cpu`

### Limitations

- Did not rebuild real `landmark_72h` data, dynamic arrays or faithful datasets
  in this task to avoid generating/overwriting heavy artifacts.
- Therefore 72h equivalence is validated at config/unit/dry-run level only;
  real artifact equivalence remains the next required check before 24h/48h.

### Documentation Updates

- Added DEC-022.
- Updated `docs/TODO.md` and `docs/REPRODUCIBILITY.md`.
- Did not modify `docs/PROJECT_HISTORY.md`.

## 2026-06-19 - Static 72h expanded tuning hyperparameter-pruning review

### Scope

- Inspected `outputs/static_72h_tuning_expanded/tuning/` for CoxPH,
  DeepSurv, LogisticHazard, PCHazard and DeepHitSingle.
- Read `configs/static_72h_tuning.yaml`, each model's `tuning_results.csv` and
  `best_hyperparameters.json`.
- No tuning, final evaluation, model code or config edits were run in this
  task.

### Findings

- CoxPH: `l1_ratio=0.0` dominates the top configurations; `l1_ratio` values
  `0.05` and `0.1` slightly but consistently reduce validation Ctd. Penalizer
  `0.01` is the weakest level; a compact grid can keep `penalizer`
  `[0.001, 0.003, 0.005]` and fix `l1_ratio=0.0`.
- DeepSurv: `[64, 32]` is clearly too small and `[128, 64]` is inferior to
  `[128, 64, 32]`. `dropout=0.0` is worst, and `learning_rate=0.0005` is both
  duplicated in the YAML and worse than `0.001`. Weight decay has weak effect;
  `0.0001` is a reasonable fixed value.
- LogisticHazard: `dropout=0.0` is clearly worse. `[64, 32]` does not appear in
  the top configurations; `[128, 64]` and `[128, 64, 32]` should be retained.
  Learning rates are close, but `0.0002` is less represented among top
  configurations. Weight decay differences are minor.
- PCHazard: `dropout=0.4` is strongly favored; `0.2` and mostly `0.3` can be
  dropped. `[256, 128]` is clearly worse than `[64, 32]` and `[128, 64]`.
  Learning rate `0.0002` is weaker on average; weight decay has minor effect.
- DeepHitSingle: `[128, 64, 32]` is the strongest architecture; `[128, 64]` is
  weakest. `dropout=0.2` is clearly inferior, while `0.4` is strongest.
  `alpha=0.1` gives worse probabilistic metrics than `0.2/0.3`; `0.0002` is the
  weakest learning rate.

### Documentation Updates

- Appended this review note only. No experiment was run and no source/config
  change was made.

## 2026-06-19 - Static landmark 24h/48h 16-combination hyperparameter selection

### Scope

- Used `outputs/static_72h_tuning_expanded/tuning/*/tuning_results.csv` to
  select compact hyperparameter grids intended for reuse at 24h and 48h.
- Selection prioritized validation Ctd and used IBS/IBLL as secondary evidence,
  while preserving limited diversity for possible landmark-specific behaviour.
- No configs, code or outputs were modified.

### Recommendations

- CoxPH: use the 15 tested combinations from `penalizer`
  `[0.0003, 0.001, 0.003, 0.005, 0.01]` x `l1_ratio`
  `[0.0, 0.05, 0.1]`, or if exactly 16 are required add a low-risk ridge
  baseline `penalizer=0.0001, l1_ratio=0.0`.
- DeepSurv: use 16 combinations from hidden layers
  `[[128,64], [128,64,32]]`, dropout `[0.1,0.2]`, learning rate
  `[0.0005,0.001]` and weight decay `[0.0001,0.001]`.
- LogisticHazard: use 16 combinations from hidden layers
  `[[128,64], [128,64,32]]`, dropout `[0.1,0.2]`, learning rate
  `[0.0005,0.0008]` and weight decay `[0.0,0.00001]`.
- PCHazard: use 16 combinations from hidden layers
  `[[64,32], [128,64]]`, dropout `[0.3,0.4]`, learning rate
  `[0.0005,0.0008]` and weight decay `[0.0,0.00001]`.
- DeepHitSingle: use 16 combinations from hidden layers
  `[[64,32], [128,64,32]]`, dropout `[0.3,0.4]`, learning rate
  `[0.0005,0.0008]` and alpha `[0.2,0.3]`, keeping sigma `0.1` and
  weight decay `0.0`.

### Documentation Updates

- Appended this selection note only. No experiment was run and no source/config
  change was made.

## 2026-06-19 - Static 72h tuning config grid reduction

### Scope

- Updated `configs/static_72h_tuning.yaml` with the compact grids selected from
  the expanded 72h tuning review for reuse in 24h/48h landmark experiments.

### Changes

- Kept CoxPH at 15 combinations:
  `penalizer=[0.0003,0.001,0.003,0.005,0.01]` x
  `l1_ratio=[0.0,0.05,0.1]`.
- Reduced DeepSurv to 16 combinations using hidden layers
  `[[128,64],[128,64,32]]`, dropout `[0.1,0.2]`, learning rate
  `[0.0005,0.001]` and weight decay `[0.0001,0.001]`.
- Reduced LogisticHazard, PCHazard and DeepHitSingle to 16 combinations each
  using the compact grids selected from validation results.
- Kaplan-Meier remains unchanged as a one-run descriptive baseline.

### Validation

- Parsed `configs/static_72h_tuning.yaml` with PyYAML in `tfg-survival`.
- Combination counts: Kaplan-Meier `1`, CoxPH `15`, DeepSurv `16`,
  LogisticHazard `16`, PCHazard `16`, DeepHitSingle `16`.

### Documentation Updates

- Appended this session note only. No experiment was run and no other project
  document needed an update.

## 2026-06-19 - DySurv faithful 72h tuning config grid reduction

### Scope

- Reviewed `outputs/dysurv_faithful_72h/tuning_results.csv` and updated
  `configs/dysurv_faithful_72h.yaml` to a compact 16-combination grid for
  reuse in 24h/48h landmark experiments.

### Findings

- The equal-weight loss setting `w_surv=w_recon=w_kl=0.333` produced apparently
  attractive IBS/IBLL in some runs but had very low `validation_std_risk10` and
  included a clearly collapsed metric-best candidate
  (`dysurv_faithful_cfg_023`, `std_risk10=0.001069`,
  `range_risk10=0.007492`).
- The selected final 72h candidate used `learning_rate=0.001`,
  `dropout=0.1`, `w_surv=0.70`, `w_recon=0.20`, `w_kl=0.10` and
  `kl_warmup_epochs=50`.
- `kl_warmup_epochs=50` was retained as the safer default because it preserved
  the accepted candidate and avoided the clearest collapsed selection.

### Changes

- Removed the equal-weight `0.333/0.333/0.333` loss setting.
- Retained the previously configured `0.50/0.30/0.20` exploratory setting.
- Added a new reconstruction-emphasizing, low-KL setting:
  `w_surv=0.55`, `w_recon=0.40`, `w_kl=0.05`.
- Fixed `kl_warmup_epochs` to `[50]`.
- Final grid count is exactly 16:
  `2 learning rates x 2 dropouts x 4 loss-weight settings x 1 warm-up`.

### Validation

- Parsed `configs/dysurv_faithful_72h.yaml` with PyYAML in `tfg-survival`.
- Confirmed total tuning combinations: `16`.

### Documentation Updates

- Appended this session note only. No experiment was run and no other project
  document needed an update.

## 2026-06-19 - Dynamic-DeepHit and static DySurv faithful grid refinement

### Scope

- Reviewed `outputs/dynamic_deephit_faithful_72h/tuning_results.csv` and
  `outputs/dysurv_static_faithful_72h/tuning_results.csv`.
- Updated `configs/dynamic_deephit_faithful_72h.yaml` and
  `configs/dysurv_static_faithful_72h.yaml` while keeping each tuning grid at
  exactly 16 combinations.

### Findings and Changes

- Dynamic-DeepHit faithful: all 16 previous candidates completed without
  collapse, but `sigma=0.2` dominated `sigma=0.1` on validation Ctd, mean
  horizon C-index, IBS and IBLL. Fixed `sigma` to `[0.2]`.
- Dynamic-DeepHit faithful: expanded loss-weight options to four balances:
  `(alpha_ranking,beta_nll) = (0.10,0.50), (0.10,0.60), (0.20,0.50),
  (0.20,0.60)`. The two original settings are preserved and two intermediate
  ranking/NLL balances are added.
- Static DySurv faithful: `learning_rate=0.0005` was clearly worse than
  `0.001` on validation Ctd, IBS and IBLL. Fixed learning rate to `[0.001]`.
- Static DySurv faithful: retained the two original loss weights and added two
  nearby survival/KL-low settings around the best observed region:
  `0.75/0.20/0.05` and `0.85/0.10/0.05`.

### Validation

- Parsed both YAML configs with PyYAML in `tfg-survival`.
- Confirmed total combinations:
  `dynamic_deephit_faithful_72h = 16`,
  `dysurv_static_faithful_72h = 16`.

### Documentation Updates

- Appended this session note only. No experiment was run and no other project
  document needed an update.

## 2026-06-19 - Static DySurv learning-rate and KL warm-up adjustment

### Scope

- Updated `configs/dysurv_static_faithful_72h.yaml` per user request while
  preserving the compact 16-combination tuning budget.

### Changes

- Changed static DySurv `learning_rate` from `[0.001]` to
  `[0.001, 0.0005]`.
- Changed static DySurv `kl_warmup_epochs` from `[20, 50]` to `[50]`.
- Static DySurv now keeps 16 combinations:
  `2 learning rates x 2 dropouts x 4 loss-weight settings x 1 warm-up`.

### Loss-Weight Note

- Static DySurv and dynamic DySurv do not use exactly the same loss-weight
  grid.
- Shared settings: `0.70/0.20/0.10` and `0.80/0.15/0.05`
  for `w_surv/w_recon/w_kl`.
- Dynamic DySurv additionally keeps reconstruction-heavy temporal settings:
  `0.50/0.30/0.20` and `0.55/0.40/0.05`.
- Static DySurv additionally keeps high-survival, low-KL settings:
  `0.75/0.20/0.05` and `0.85/0.10/0.05`.

### Validation

- Parsed `configs/dysurv_static_faithful_72h.yaml` and
  `configs/dysurv_faithful_72h.yaml` with PyYAML in `tfg-survival`.
- Confirmed total combinations:
  `dysurv_static_faithful_72h = 16`,
  `dysurv_faithful_72h = 16`.

### Documentation Updates

- Appended this session note only. No experiment was run and no other project
  document needed an update.

## 2026-06-19 - Common DySurv loss-weight grid

### Scope

- Updated `configs/dysurv_faithful_72h.yaml` so temporal DySurv faithful uses
  the same `loss_weights` grid as `configs/dysurv_static_faithful_72h.yaml`.

### Tuning-Based Rationale

- Existing temporal DySurv tuning supported `0.70/0.20/0.10` as the accepted
  non-collapsed candidate and `0.80/0.15/0.05` as a close stable alternative.
- Existing static DySurv tuning favored `0.80/0.15/0.05`, with
  `0.70/0.20/0.10` close behind.
- Equal-weight temporal DySurv candidates were avoided because they included
  suspicious low-risk-dispersion or collapsed behavior.
- The final common grid keeps the two strongest observed regions and two nearby
  high-survival, low-KL variants:
  `0.70/0.20/0.10`, `0.75/0.20/0.05`, `0.80/0.15/0.05`,
  `0.85/0.10/0.05`.

### Validation

- Parsed `configs/dysurv_faithful_72h.yaml` and
  `configs/dysurv_static_faithful_72h.yaml` with PyYAML in `tfg-survival`.
- Confirmed both configs have identical `loss_weights`.
- Confirmed both configs still expand to 16 tuning combinations.

### Documentation Updates

- Added DEC-023 to `docs/DECISIONS.md`.
- Did not update `docs/EXPERIMENT_LOG.md` because no experiment was run.
- Did not update `docs/TODO.md` or `docs/REPRODUCIBILITY.md` because no new
  priority or execution command was introduced.

## 2026-06-19 - Dynamic-DeepHit compact loss grid refinement

### Scope

- Updated `configs/dynamic_deephit_faithful_72h.yaml` to use only the two
  Dynamic-DeepHit loss-weight settings most supported by the completed 72h
  tuning results.

### Changes

- Changed `weight_decay` from `[0.0001]` to `[0.0, 0.0001]`.
- Reduced `loss_weights` from four settings to:
  `alpha_ranking=0.10, beta_nll=0.50` and
  `alpha_ranking=0.20, beta_nll=0.60`.
- Kept `sigma=[0.2]`.

### Validation

- Parsed `configs/dynamic_deephit_faithful_72h.yaml` with PyYAML in
  `tfg-survival`.
- Confirmed total combinations remain 16:
  `2 learning rates x 2 weight-decay values x 2 dropouts x 2 loss settings`.

### Documentation Updates

- Added DEC-024 to `docs/DECISIONS.md`.
- Did not update `docs/EXPERIMENT_LOG.md` because no experiment was run.
- Did not update `docs/TODO.md` or `docs/REPRODUCIBILITY.md` because no new
  priority or execution command was introduced.
## 2026-06-20 - Landmark 24h/48h output audit after full command batch

### Scope

- Inspected generated artifacts under `data/processed/landmark_24h`,
  `data/processed/landmark_48h`, `outputs/landmark_24h` and
  `outputs/landmark_48h` after the user launched the 24h/48h build, tuning and
  final-seed command batch.
- Did not modify model code, configs or experiment outputs.

### Findings

- Landmark static, dynamic, DySurv-feature-filtered and faithful datasets were
  generated for both 24h and 48h.
- General static-model tuning failed at the first Kaplan-Meier candidate for
  both landmarks. A temporary debug rerun of Kaplan-Meier showed the cause:
  `static_file_suffix` was resolved as if it were a filesystem path, producing
  invalid parquet names like `train_C:\...\static_24h.parquet`.
- Temporal DySurv faithful and Dynamic-DeepHit faithful tuning failed for all
  candidates at both landmarks because `validate_faithful_splits` in
  `src/models/dynamic_72h/train_dysurv_faithful.py` still requires exactly 72
  hourly steps.
- Static-only DySurv faithful completed tuning and final seeds at both
  landmarks, with non-collapsed selected candidates and test metrics written to
  `final_seed_results.csv`.

### Documentation Updates

- Appended this session note only. No methodological decision or reproducibility
  command changed during this audit.

## 2026-06-20 - Landmark wrapper bug fixes for static suffix and faithful sequence length

### Scope

- Applied two minimal corrections after auditing failed 24h/48h landmark runs.
- Did not change model architectures, losses, censoring logic, metrics or
  hyperparameter grids.

### Changes

- Updated `scripts/tune_static_72h_models.py` so `static_file_suffix` and
  `file_suffix` are not resolved as filesystem paths. They remain textual
  suffixes such as `static_24h`, `static_48h` and `static_72h`.
- Updated `src/models/dynamic_72h/train_dysurv_faithful.py` so faithful split
  validation no longer requires exactly 72 hourly steps. It now requires
  train/validation/test to share the same temporal length and records the
  observed input hours in the leakage checks.

### Validation

- Ran `python -m py_compile scripts/tune_static_72h_models.py
  src/models/dynamic_72h/train_dysurv_faithful.py` with `tfg-survival`: passed.
- Ran a temporary Kaplan-Meier 24h debug fit using the corrected path resolver:
  completed and returned validation Ctd `0.0`, confirming the static parquet
  path is no longer malformed.
- Loaded small 24h and 48h faithful train/validation/test samples and ran
  `validate_faithful_splits`: returned `24/24/24` and `48/48/48` input hours.
- Ran `python -m pytest tests/test_landmark_pipeline.py -q`: `4 passed`.

### Documentation Updates

- Appended this session note only. No experiment log entry was added because
  only smoke/debug validation was run, not a full model experiment.

## 2026-06-22 - Landmark 24h/48h uploaded output inspection

### Scope

- Inspected newly added `data/processed/landmark_24h`,
  `data/processed/landmark_48h`, `outputs/landmark_24h` and
  `outputs/landmark_48h` artifacts.
- No code, config, data or output artifacts were modified.

### Findings

- Both landmark datasets are present with static, dynamic, filtered dynamic and
  faithful prepared inputs.
- `landmark_24h` static tuning is incomplete: Kaplan-Meier and CoxPH completed,
  but DeepSurv only created `deepsurv_cfg_001` partial training artifacts and no
  `tuning_results.csv`; downstream static models were not reached.
- `landmark_48h` static tuning is complete for Kaplan-Meier, CoxPH, DeepSurv,
  LogisticHazard, PCHazard and DeepHitSingle.
- Faithful model tuning is complete for both landmarks:
  DySurv temporal, Dynamic-DeepHit faithful and DySurv static faithful each have
  16 completed validation candidates and selected hyperparameters.
- No final-seed result files were found for 24h or 48h in the inspected output
  roots; current uploaded artifacts should be treated as validation/tuning
  results only.

### Documentation Updates

- Appended this session note only. No experiment log entry was added because the
  task was an artifact inspection rather than a new experiment run.
## 2026-06-22 - Technical Agent methodology reconstruction report

- Task: reconstructed the implemented MIMIC-IV landmark methodology for the thesis methodology/design section, without modifying model code or running training.
- Scope inspected: cohort extraction, static/dynamic landmark dataset builders, landmark wrappers, current tuning configs, evaluation metrics, and faithful DySurv/Dynamic-DeepHit model implementations.
- Light checks only: inspected configs and metadata artifacts for 24h/48h/72h feature counts, split sizes, and output structure; no experiments or retraining were run.
- Key caveats documented for the report: 24h/48h use `flat_features_with_time_since_admission.csv`; existing 72h processed metadata appears older unless rebuilt; split uniqueness is enforced by ICU stay ID, while subject-level uniqueness is not enforced in the inspected split code; event timing uses the implemented `actualiculos`/hospital mortality proxy.

## 2026-06-23 - Landmark 48h/72h final results analysis

### Scope

- Inspected completed final outputs under `outputs/landmark_48h` and
  `outputs/landmark_72h`.
- Reviewed static model tuning/final outputs, faithful Dynamic-DeepHit,
  temporal DySurv and static DySurv tuning/final outputs, dataset audits,
  selected hyperparameters, per-seed test metrics and horizon C-index metrics.
- No training, tuning reruns, code changes, config changes or output rewrites
  were performed.

### Main Findings

- 48h has a larger eligible cohort than 72h (`27802/9267/9268` vs
  `18706/6236/6236` train/validation/test) and lower observed event rate
  (`~0.120` vs `~0.137`).
- Dynamic-DeepHit faithful is the best overall model at both landmarks:
  test Ctd `0.816153 +/- 0.000765` at 48h and
  `0.788668 +/- 0.002194` at 72h, with the best IBS/IBLL among dynamic models.
- Temporal DySurv faithful is close to Dynamic-DeepHit in discrimination but
  remains worse in probabilistic calibration, especially at 72h.
- Static-only DySurv faithful improves over most classical/static baselines at
  48h but is close to the static neural baselines at 72h.
- Among static pycox/lifelines models, DeepSurv, LogisticHazard and
  DeepHitSingle are close; PCHazard has weaker Antolini Ctd but competitive
  horizon C-index and probabilistic metrics.

### Documentation Updates

- Appended this session note only. No experiment was run, no new technical
  decision was made, and no reproducibility command changed.
## 2026-06-23 - Project Manager repository cleanup strategy for landmark-only GitHub review

### Scope

- Reviewed governance instructions, project documentation, root README, `.gitignore`
  and the visible repository tree outside excluded heavy artifact folders.
- User clarified that the GitHub-facing repository should focus on the current
  parametrizable landmark structure; old static/pre-landmark code and historical
  experiment scaffolding are lower priority than a clean tutor review.

### Findings

- The root `README.md` is stale and still references old commands such as
  `configs/train.yaml`, `scripts/train_static_pipeline.py`,
  `scripts/evaluate.py` and `scripts/run_mimic_pipeline.py`.
- The current active methodology is the landmark layer built around
  `--landmark-hours {24,48,72}`, with `static_72h_*` configs/scripts still used
  as compatibility/base implementation for the landmark wrappers.
- Several pre-landmark static scripts/configs/models remain in the tree and can
  be archived or removed from the GitHub-facing main branch after dependency
  checks.
- Some 72h-named files are not truly obsolete because current landmark wrappers
  import them; they should be renamed/refactored before archival rather than
  moved immediately.

### Outcome

- Produced a cleanup strategy focused on keeping only the current landmark
  project structure visible in GitHub.
- No code, configs, tests, data, outputs or model artifacts were modified.
