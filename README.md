# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Debug sessions

Set `FURU_OVERLAY` to an absolute folder to try a change without touching what is stored:

```bash
FURU_OVERLAY=/tmp/furu-debug uv run python experiment.py
```

Stored results are still found and loaded, but never changed or locked. Everything the session writes (results, working folders, schema lines, plans, records, snapshots, even the manager's address) goes into the overlay at the path it would otherwise have, and what is in the overlay is read first. `rm -rf /tmp/furu-debug` undoes the session. A session's Slurm launches need a manager of their own, started with the same `FURU_OVERLAY`. Nothing else differs: a launch still needs committed, pushed code, and every machine that runs a worker must see the overlay.
