# TODO

## High Priority

### GitHub-facing landmark cleanup

- [x] Create safety branch `backup/pre-cleanup-landmark` before cleanup.
- [x] Remove pre-landmark static and old dynamic pipeline code from the active
      GitHub-facing tree.
- [x] Rename active configs, scripts, modules and tests toward `landmark_*`
      naming.
- [x] Rewrite root `README.md` as the guide for the current landmark pipeline.
- [x] Update `docs/REPRODUCIBILITY.md` so commands match the active structure.
- [x] Run the focused landmark test suite after cleanup.
- [x] Review remaining internal `72h` names that are compatibility/function
      names rather than public entrypoints.

### Landmark experiments

- [ ] Confirm which final landmark result set will be cited in the thesis:
      48h/72h only or 24h/48h/72h.
- [ ] Complete or explicitly defer incomplete 24h static tuning/final results.
- [ ] Generate final comparison tables from the selected landmark outputs.
- [ ] Document final model ranking and calibration caveats for the thesis.

## Medium Priority

- [ ] Consolidate the 2026-06-19 to 2026-06-23 landmark implementation and
      results into `docs/PROJECT_HISTORY.md`.
- [ ] Add or update decision entries if the final thesis scope excludes
      archived pre-landmark pipelines.
- [ ] Add a short README section explaining required local MIMIC-derived input
      files without exposing data.
- [ ] Review `docs/EXPERIMENT_LOG.md` for missing final landmark run entries.

## Low Priority

- [ ] Optionally rename remaining internal model class identifiers such as
      `DySurvFaithful72h` to fully generic landmark names.
- [ ] Optionally add a small architecture diagram to the README.
- [ ] Review ignored local folders before publishing or sharing the repository.

## Blocked

- [ ] Publishing the repository publicly remains blocked until data/output
      exclusion is manually verified.

## Done Recently

- [x] Cleaned the GitHub-facing repository around the active parametrizable
      landmark pipeline and preserved the previous state in
      `backup/pre-cleanup-landmark` — 2026-06-23.
- [x] Validated the cleaned landmark tree with `py_compile`, `pytest tests -q`
      and dry-runs of the four main tuning entrypoints — 2026-06-23.
