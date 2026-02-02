# plans/executor-plan/10-logging.md — Logging updates (“load_or_create” → “get”)

## Scope

Update runtime logging that special-cases the `load_or_create` prefix so it now uses `get`.

Files:
- `src/furu/runtime/logging.py`
- any call sites in `src/furu/core/furu.py` that emit console start lines

## Required changes

- Replace any constant like `_LOAD_OR_CREATE_PREFIX = "load_or_create"` with `"get"`.
- Rename helper like `_strip_load_or_create_decision_suffix` or update its logic.
- Ensure the console prefix printed by `Furu` uses `"get"`.

## Checklist

- [x] Update prefix constant to `"get"`
- [x] Update any prefix-stripping helper
- [x] Update any emitted messages in `Furu` console logs

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Switch console prefix handling from `load_or_create` to `get`. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
