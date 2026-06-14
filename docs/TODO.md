# TODO

## How To Use This File

This file tracks current documentation, reproducibility and project-completion
work. It should stay actionable and aligned with [PROJECT_HISTORY.md](PROJECT_HISTORY.md),
[DECISIONS.md](DECISIONS.md), [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Section Guide

- `High Priority`: work that affects thesis validity, reproducibility or final
  claims.
- `Medium Priority`: important cleanup, consolidation or analysis that improves
  quality but does not block the static baseline.
- `Low Priority`: publication polish, archival cleanup and optional robustness
  work.
- `Blocked`: tasks that need data, compute, decisions or Project Manager input.
- `Done Recently`: completed work with dates and enough context to avoid losing
  track of recent progress.

## Prioritization Rules

- Put leakage, invalid metrics, missing reproducibility and final-claim blockers
  in `High Priority`.
- Put dynamic-model implementation and major analysis extensions in `High
  Priority` when they are required for the thesis claim; otherwise keep them in
  `Blocked` until scope is confirmed.
- Put documentation consolidation and artifact cleanup in `Medium Priority`
  unless they block writing or reproduction.
- Put optional polish, extra figures and repository hygiene in `Low Priority`.

## Update Rules

- Preserve unresolved tasks unless they are moved, completed or explicitly
  superseded.
- When completing a task, move it to `Done Recently` with `YYYY-MM-DD`.
- When blocking a task, include the reason.
- Do not mark work done unless there is a local artifact, documented result or
  explicit Project Manager confirmation.
- Technical agents may update tasks they create or complete. The Project
  Manager/Historian owns periodic cleanup and consolidation.

## High Priority

### DeepHit calibration review

- [x] Investigate and implement explicit tail mass beyond prediction horizon.
- [x] Review and correct DeepHit ranking loss implementation.
- [x] Audit and repair censoring/horizon label encoding.
- [x] Re-run DeepHit after implementation fixes and log corrected metrics in `docs/EXPERIMENT_LOG.md`.
- [x] Inspect corrected DeepHit calibration metrics: IBS, IBLL/NBLL and final survival/tail diagnostics.
- [ ] Compare DeepHit outputs against original paper assumptions.
- [ ] Generate corrected DeepHit calibration plots.
- [ ] Compare corrected DeepHit survival curves against PCHazard survival curves.
- [ ] Run a small synthetic DeepHit overfit test.
- [ ] Tune DeepHit hyperparameters after diagnostics.

- [ ] Run the new `static_72h_pycox` data build on real MIMIC-derived inputs.
- [ ] Rerun validation-only tuning for the new `static_72h_pycox` models after
      the 2026-06-12 audit fixes to PCHazard `sub=10` and IBS/IBLL integration
      grids.
- [ ] Run final 3-seed evaluation for the selected `static_72h_pycox` models.
- [ ] Compare `static_72h_pycox` results against future dynamic 72h models on
      the same cohort, split and target definition.

- [ ] Run full validation-only tuning for DySurv and Dynamic-DeepHit on
      `dynamic_72h_dysurv_features`.
- [ ] Inspect dynamic smoke/final candidate calibration and survival curves
      before final 3-seed evaluation.
- [ ] Block final DySurv interpretation until the posterior-collapse findings
      in `outputs/dynamic_72h/dysurv_audit_report.md` are addressed.
- [x] Add DySurv checkpoint/full-prediction saving and latent/risk dispersion
      diagnostics before rerunning validation tuning.
- [x] Revise the DySurv reconstruction target so masks and repeated unscaled
      static covariates do not dominate the VAE objective; rerun tiny-overfit
      and smoke validation in the isolated faithful pipeline.
- [ ] Run the 16-candidate validation-only
      `dysurv_faithful_72h` tuning grid on GPU.
- [ ] Review faithful tuning survival curves and collapse diagnostics, then
      accept or reject the selected validation candidate.
- [ ] Run faithful final seeds 42, 123 and 2026 only after validation review.
- [ ] Run final 3-seed dynamic_72h evaluation after validation tuning.
- [ ] Decide whether dynamic models require `delta_seq` or another
      time-since-last-observed representation before training.
- [ ] Decide whether the thesis requires full dynamic model training, then
      document the execution scope in `docs/DECISIONS.md`.
- [ ] Update root `README.md` so usage commands match the current scripts and configs.

## Medium Priority

- [ ] Backfill `docs/EXPERIMENT_LOG.md` with the final static runs already summarized in `docs/PROJECT_HISTORY.md`.
- [ ] Add formal decision entries for the final 60/20/20 static split, train-only preprocessing fit and model set.
- [ ] Periodically consolidate important `SESSION_NOTES.md` entries into `docs/PROJECT_HISTORY.md` under the permanent Project Manager/Historian role.
- [ ] Verify that all final metric artifacts referenced by configs exist locally after a clean static pipeline run.
- [ ] Consolidate static-vs-dynamic comparison requirements before writing final thesis conclusions.
- [ ] Document the status of temporal parquet artifacts and their intended role in the dynamic pipeline.

## Low Priority

- [ ] Clean or archive historical duplicate artifacts in `basura/` and legacy `models/` after Project Manager approval.
- [ ] Add publication-oriented documentation that distinguishes project-owned code from local reference repositories.
- [ ] Add a short glossary for survival metrics used in the thesis documentation.
- [ ] Review notebooks and mark which are exploratory only versus reproducibility-relevant.

## Blocked

- [ ] Train a complete DySurv pipeline on MIMIC-IV dynamic inputs — blocked
      until DySurv adapter, compute budget and training config are confirmed.
- [ ] Publish or package the repository — blocked until data/output exclusion and reference-code policy are reviewed.

## Done Recently

- [x] Implemented, prepared and smoke-tested the isolated
      `dysurv_faithful_72h` pipeline with recurrent temporal-only
      reconstruction, train-only imputation, collapse-aware selection,
      checkpoints and full predictions — 2026-06-14
- [x] Implemented and built the `dynamic_72h` dataset from the `static_72h_pycox`
      cohort/splits, with strict first-72h temporal filtering, train-only
      feature selection/imputation/scaling and audit outputs — 2026-06-12
- [x] Created a fast DySurv-compatible `dynamic_72h` feature subset with 76
      temporal columns from the existing 146-feature arrays — 2026-06-12
- [x] Overwrote `dynamic_72h_dysurv_features` as a 61-feature reduced subset
      after removing 15 additional chart-derived variables — 2026-06-12
- [x] Implemented and smoke-tested the isolated `dynamic_72h` DySurv and
      Dynamic-DeepHit training/evaluation layer without using test during
      tuning — 2026-06-12
- [x] Audited `static_72h_pycox` DeepHitSingle/PCHazard time grids, added audit
      artifacts, separated IBS/IBLL integration grids from daily horizon
      C-index, and fixed PCHazard prediction interpolation with `sub=10` —
      2026-06-12
- [x] Implemented the isolated `static_72h_pycox` pipeline skeleton, configs,
      scripts, metrics and synthetic validation tests — 2026-06-11
- [x] Generated thesis-ready final static tables, figures and
      `outputs/metrics/final_static_model_comparison.csv` from
      `outputs/final_static/`; the stale `static_model_comparison.csv` remains
      excluded from final reporting — 2026-06-10
- [x] Added CoxPH final-seed results and completed the four-model static
      comparison for CoxPH, DeepHit, DeepSurv and PCHazard — 2026-06-09
- [x] Re-ran CoxPH validation-only tuning with the full penalizer grid
      `[0.0, 0.001, 0.01, 0.1]`; selected `penalizer=0.1` — 2026-06-09
- [x] Ran validation-only static hyperparameter tuning for CoxPH, DeepSurv,
      PCHazard and DeepHit — 2026-06-09
- [x] Ran final static model evaluation with seeds 42, 123 and 2026 after
      validation selection — 2026-06-09
- [x] Analyzed final static three-seed results for DeepHit, DeepSurv and
      PCHazard and logged them in `docs/EXPERIMENT_LOG.md` — 2026-06-09
- [x] Investigated CoxPH smoke-test metric regression and confirmed the new
      pipeline reproduces the old CoxPH benchmark with `penalizer=0.1` —
      2026-06-08
- [x] Prepared validation-only static tuning and final three-seed pipeline with smoke tests — 2026-06-08
- [x] Logged corrected DeepHit run after tail-support/ranking-loss fixes; test IBS improved to 0.1107 and IBLL/NBLL to 0.3531 — 2026-06-08
- [x] Implemented approved DeepHit tail-support, censored-horizon mask and ranking-loss corrections with focused tests — 2026-06-08
- [x] Organized static model metric artifacts under model-specific `outputs/metrics/<model>/` folders — 2026-06-07
- [x] Updated static survival evaluation protocol to avoid final-risk C-index as the primary DeepHit/PCHazard metric — 2026-06-07
- [x] Added formal decision entry for fixed-grid static metric conventions — 2026-06-07
- [x] Reconstructed consolidated project history in `docs/PROJECT_HISTORY.md` — 2026-06-07
- [x] Created documentation governance templates and initialized session notes — 2026-06-07
