# Documentation protocol

This folder contains consolidated project documentation.

Start here when deciding where to record project knowledge. For reproducible
commands and setup, use [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For the clean
historical narrative, use [PROJECT_HISTORY.md](PROJECT_HISTORY.md).

## Files

### PROJECT_HISTORY.md
Clean chronological history of the project.
Only the Project Manager / Historian agent should edit this file, unless explicitly instructed.

### DECISIONS.md
Important technical and methodological decisions.
Any technical agent may append a new decision, but must not rewrite previous decisions.

### EXPERIMENT_LOG.md
Structured log of experiments actually run.
Any agent that runs an experiment must append an entry here.

### REPRODUCIBILITY.md
Instructions to reproduce the project.
Only update this when commands, dependencies, configs or pipeline steps change.

### TODO.md
Current project tasks.
Any agent may update this, but must preserve priorities and avoid deleting unresolved tasks without explanation.

### SESSION_NOTES.md
Root-level session notes for governance updates and working-session summaries.
Use this for short audit summaries, ownership notes and handoff context.

## General rules

- Prefer appending new sections over rewriting existing content.
- Never delete previous documentation unless explicitly asked.
- Use dates in `YYYY-MM-DD` format.
- Link every experiment to its config, command and output path.
- Document any change that affects methodology, reproducibility or interpretation.


## Documentation ownership rules

### `SESSION_NOTES.md`

- Technical chats: yes.
- Project Manager: yes, but should normally not rewrite existing entries.

### `docs/EXPERIMENT_LOG.md`

- Chat that runs the experiment: yes.
- Project Manager: yes, for consolidation and cleanup while preserving entries.

### `docs/DECISIONS.md`

- Chat that makes a technical or methodological decision: yes, append only.
- Project Manager: yes, for organization and consistency while preserving decision content.

### `docs/TODO.md`

- All chats: yes, but carefully.
- Project Manager: yes, for cleanup, prioritization and removing outdated tasks.

### `docs/REPRODUCIBILITY.md`

- Update only when commands, environment setup, dependencies, configs or pipeline steps change.

### `docs/PROJECT_HISTORY.md`

- Project Manager / Historian only.
- This is the only file where stable project history should be consolidated.

## Cross-linking rules

- Link decisions to related history sections, configs and source files.
- Link experiments to commands, configs, outputs and decisions.
- Link TODO items to decisions or experiment entries when a task follows from them.
- Link reproducibility changes to the command or config that changed.
