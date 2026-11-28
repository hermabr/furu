# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Trying things against a real store: `FURU_OVERLAY`

```bash
FURU_OVERLAY=/scr/me/try-1 uv run python my_experiment.py
```

With `FURU_OVERLAY` set, furu reads everything the real store holds and writes
only under the overlay folder: results and artifacts (also those with a custom
`Metadata(storage=...)`), working folders and locks, schema and remembered
files, plans, records, code snapshots, and the manager's state. Each path is
mirrored there, so `/data/objects/...` is written to
`/scr/me/try-1/data/objects/...`.

- Reads look in the overlay first, then in the real store.
- The real store is never modified or locked. Delete the overlay folder to undo
  the session; run again with the same folder to go on from it.
- `Spec.delete()` deletes only what the overlay holds. For a result in the real
  store it raises and deletes nothing.
- Only the environment turns an overlay on: `pyproject.toml` may not set it, an
  empty `FURU_OVERLAY=` is unset, and `pytest` runs ignore it.
- The folder is an absolute path to a new or empty folder, or to an overlay an
  earlier session made. Furu refuses any other existing folder. An overlay
  inside a git repository stays out of `git status`.
- Workers inherit the overlay from the script that launches them. A Slurm run
  needs its own manager, started with the same `FURU_OVERLAY`.
