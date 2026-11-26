# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Debug sessions

Set `FURU_OVERLAY` (or `[tool.furu] overlay` in `pyproject.toml`) to try a change without touching what is stored:

```bash
FURU_OVERLAY=/tmp/furu-debug uv run python experiment.py
```

Results that are already stored still load as cache hits and are never written, created, or locked. Everything the session writes (results, working folders, schema lines, remembered lines, plans, provenance records, execution logs, and snapshots) goes into the overlay at the absolute path it would otherwise have, so `rm -rf /tmp/furu-debug` undoes the session. A relative overlay anchors at the main git worktree root, like `furu-data/`. Git and provenance checks are unchanged, and every worker machine must see the overlay path.
