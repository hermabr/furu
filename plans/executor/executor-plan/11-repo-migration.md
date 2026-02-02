# plans/executor-plan/11-repo-migration.md — Repo-wide migration and cleanup

## Scope

- Mechanical rename: `load_or_create()` → `get()`
- Remove any executor submission via `.get(...)` or `.load_or_create(executor=...)`
- Update docs/examples and tests

## Mechanical migration steps

1) Delete `load_or_create` from `Furu` (done in 01).
2) Search & replace all call sites:

```bash
grep -R "load_or_create" -n src tests examples README.md
```

Replace with `.get()`.

3) Update README/examples:
- interactive usage:
  - `obj.get()`
- local parallel:
  - `run_local([...])`
- slurm per-node:
  - `submit_slurm_dag([...], specs=...)`
- slurm pool:
  - `run_slurm_pool([...], specs=..., max_workers_total=..., window_size=...)`

4) Update any docs that mention submitit folder placement or per-object logs.

## API exports

In `src/furu/execution/__init__.py`, export:

- `SlurmSpec`
- `run_local`
- `submit_slurm_dag`
- `run_slurm_pool`

Optionally re-export from `src/furu/__init__.py`.

## Checklist

- [x] Replace all references to `load_or_create` in `src/`
- [x] Replace all references in `tests/`
- [x] Update docs/examples
- [x] Ensure imports and public exports are correct

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Confirm no load_or_create references and document executor APIs in README. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
