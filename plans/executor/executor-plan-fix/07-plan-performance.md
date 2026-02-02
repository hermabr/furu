# 07 — Planner loop performance: reduce `exists()` churn and log noise

## Why

`build_plan()` calls `obj.exists()`, which runs `_validate()` and can log at INFO. In tight polling loops
(run_local, run_slurm_pool), this can be expensive/noisy.

Relevant files:
- `src/furu/execution/plan.py:33`
- `src/furu/core/furu.py:312` (exists logging/validate)

## Tasks

Choose one of these approaches (A preferred):

- [x] A) Add a quiet existence check used by planners:
  - `obj._exists_quiet()` (no INFO logs)
  - optionally caches validate decisions per tick
  - update plan builder to use it

- [~] OR B) Downgrade exists validation logs to DEBUG so plan polling doesn’t spam.

- [x] Add a micro test (optional):
  - ensure plan building doesn’t emit INFO logs for existence checks (caplog) OR
  - ensure exists calls are bounded per loop tick (if you implement caching)

## Acceptance criteria

- Executor loops do not spam INFO logs and do not repeatedly validate more than necessary.

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Add quiet existence checks for planners, swap build_plan to use them, and test no INFO logs. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
