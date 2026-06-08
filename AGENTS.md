# AGENTS.md

This is the main instruction file for Codex in this repository.

## Mandatory reading order

Before modifying code, read:

1. `CODEX_INSTRUCTIONS.md`
2. `TFG/CODEX_TFG_MATES_JAVI.md`
3. `docs/TODO.md`
4. `SESSION_NOTES.md`

Before documentation-only governance work, read:

1. `CODEX_INSTRUCTIONS.md`
2. `docs/README.md`
3. `docs/PROJECT_HISTORY.md`
4. `docs/DECISIONS.md`
5. `docs/EXPERIMENT_LOG.md`
6. `docs/REPRODUCIBILITY.md`
7. `docs/TODO.md`
8. `SESSION_NOTES.md`

For academic writing or memory-related tasks, also read:

1. `TFG/Decisiones_clave_y_guia_enfoque_TFG.md`
2. `TFG/indice_definitivo_TFG.md`
3. `TFG/instrucciones_TFG_mates.md`

## Project logic

This project is a Mathematics TFG about replication and assessment of survival models on MIMIC-IV adult ICU data.

The central goal is to evaluate whether dynamic survival models, especially DySurv, add predictive value over static survival models.

Prioritize:
- simplicity,
- efficiency,
- reproducibility,
- methodological fidelity,
- no data leakage,
- clear academic documentation.

## Documentation protocol

At the start of every task:
- read this `AGENTS.md`;
- read `SESSION_NOTES.md`;
- read `docs/TODO.md`;
- inspect relevant configs before changing code.

At the end of every task:
- append a new entry to `SESSION_NOTES.md`;
- update `docs/EXPERIMENT_LOG.md` if an experiment was run;
- update `docs/DECISIONS.md` if a technical decision was made;
- update `docs/TODO.md` if priorities changed.

Do not overwrite previous session notes.

## Agent communication

Agents do not communicate directly.

Communication must occur through:

- SESSION_NOTES.md
- docs/DECISIONS.md
- docs/TODO.md
- docs/EXPERIMENT_LOG.md

Before starting a task, agents should review the relevant documentation.

Before finishing a task, agents should document important findings and decisions.

## Documentation Governance Manager protocol

When asked to act as Documentation Governance Manager:

- do not modify model code, model configs, data, outputs, checkpoints or local
  environments;
- avoid scanning `data/`, `outputs/`, `env/`, `checkpoints/` and heavy model
  artifact folders unless the user explicitly requires it;
- audit the documentation system for missing sections, broken templates,
  duplicated information, inconsistencies and ownership gaps;
- preserve existing documentation content;
- prefer prepending or appending governance sections over rewriting historical
  narrative;
- keep `docs/PROJECT_HISTORY.md` as the Project Manager/Historian-owned
  consolidation file;
- keep `docs/DECISIONS.md`, `docs/EXPERIMENT_LOG.md`, `docs/TODO.md`,
  `docs/REPRODUCIBILITY.md` and `SESSION_NOTES.md` cross-linked;
- record the documentation health status and next actions in `SESSION_NOTES.md`;
- if root-level instructions and docs disagree, document the inconsistency and
  either fix the governance file directly or add a TODO when the fix is outside
  the requested scope.

## Code rules

- Do not commit MIMIC-IV raw or derived data.
- Keep original reference implementations inside `src/models_references/`.
- Put adapted TFG models in `src/models/`.
- Put adapted data processing code in `src/data/`.
- Put executable scripts in `scripts/`.
- Put configuration in `configs/`.
- Put generated artifacts in `outputs/`.
- Keep paths relative and configurable.
- Avoid changing architecture, loss, censoring logic, discretization, or horizon of replicated models unless documented.

## Testing and validation

When modifying code:
- run the smallest relevant test/check;
- report commands run;
- report failures honestly;
- do not hide failing tests.

## Project Manager / Historian protocol

When asked to act as Project Manager or Historian:
- review `SESSION_NOTES.md`;
- review recent git changes;
- update `docs/PROJECT_HISTORY.md`;
- consolidate `docs/DECISIONS.md`;
- clean `docs/TODO.md`;
- do not modify model code unless explicitly requested.

## Docs writing protocol

Before writing inside `docs/`, read:

1. `docs/README.md`
2. `SESSION_NOTES.md`
3. `docs/TODO.md`
4. The specific docs file to be updated

Different agents must follow this ownership model:

### Technical agents

Technical agents may update:
- `SESSION_NOTES.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/DECISIONS.md`
- `docs/TODO.md`
- `docs/REPRODUCIBILITY.md` only if commands, dependencies or execution steps changed

Technical agents must not rewrite:
- `docs/PROJECT_HISTORY.md`

Technical agents should append entries rather than reorganize existing documentation.

### Project Manager / Historian agent

The Project Manager / Historian agent may update:
- `docs/PROJECT_HISTORY.md`
- `docs/DECISIONS.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/TODO.md`
- `docs/REPRODUCIBILITY.md`

Its role is to consolidate, clean duplicated notes, detect inconsistencies and keep documentation coherent.

The Project Manager / Historian must not modify model code unless explicitly asked.

### Documentation Governance Manager agent

The Documentation Governance Manager may update:

- `AGENTS.md`
- `docs/README.md`
- `docs/DECISIONS.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/TODO.md`
- `docs/REPRODUCIBILITY.md`
- `SESSION_NOTES.md`

The Documentation Governance Manager may prepend governance instructions to
`docs/PROJECT_HISTORY.md` when ownership rules are missing, but should not
rewrite existing project history.

The Documentation Governance Manager must not modify model code, model configs,
data, outputs or checkpoints unless the user explicitly changes the scope.

## Required docs update at end of task

At the end of every task:

1. Append a section to `SESSION_NOTES.md`.
2. If an experiment was run, append to `docs/EXPERIMENT_LOG.md`.
3. If a technical or methodological decision was made, append to `docs/DECISIONS.md`.
4. If future work changed, update `docs/TODO.md`.
5. If execution instructions changed, update `docs/REPRODUCIBILITY.md`.

Never silently skip documentation.
If no documentation update is needed, explicitly state why in the final response.

## Repository scan policy

Avoid scanning:

- data/
- outputs/
- env/
- .pytest_cache/
- models/
- checkpoints/

unless explicitly required.

For project documentation, prioritize:

- docs/
- configs/
- scripts/
- src/
- tests/
- SESSION_NOTES.md
- git history
