# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Debug sessions

Set `FURU_OVERLAY` to an absolute path to try a change without touching what is stored:

```bash
FURU_OVERLAY=/tmp/furu-debug uv run python experiment.py
```

Results that are already stored are still found and loaded, and never changed or locked. Everything the session writes (results, working folders, schema lines, plans, records, snapshots) goes into the overlay, at the path it would otherwise have, so `rm -rf /tmp/furu-debug` undoes the session. Nothing else differs: a launch still needs committed, pushed code, and every machine that runs a worker must see the overlay.
