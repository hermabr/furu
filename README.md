# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Debug sessions

Set `FURU_OVERLAY` (or `[tool.furu] overlay`) to an absolute path to try a change without touching what is stored:

```bash
FURU_OVERLAY=/tmp/furu-debug uv run python experiment.py
```

Whatever furu would write at a path, it writes at that path inside the overlay: results, working folders, schema lines, plans, records, snapshots and the manager's files. What is already stored is still found and loaded, and never changed or locked, so `rm -rf /tmp/furu-debug` undoes the session. A result in the overlay is used before one under it.

Nothing else differs: a launch still needs committed, pushed code. Every machine that runs a worker must see the overlay, and a launch on Slurm needs a manager started with the same overlay (`FURU_OVERLAY=/tmp/furu-debug uvx furu@<version> manager`), as a session never shares the manager of the store.
