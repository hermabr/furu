# 08 — Verification tests for v1.1 hardening

This phase ensures all fixes are covered by tests so regressions don’t return.

## Tasks

### Pool infinite-spin prevention
- [x] Test: controller fails fast when `queue/failed` contains entries (Phase 01)

### Pool stale running recovery
- [x] Test: stale running tasks are requeued to todo (Phase 02)

### retry_failed semantics
- [x] Test: with retry_failed enabled, executor retries failed nodes (Phase 03)
- [x] Test: with retry_failed disabled, executor fails fast on failed deps (Phase 03)

### Slurm DAG robustness
- [x] Test: `_job_id_for_in_progress()` fails fast for non-submitit backend (Phase 04)
- [x] Test: `_wait_for_job_id()` robustly records job_id across attempt switch (Phase 04)

### SlurmSpec.extra nested dict
- [x] Test: nested extra passes through submitit_factory (Phase 05)

## Acceptance criteria

- `make test` and `make lint` passes.
- These tests would have failed before the fix.

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Confirmed coverage for all v1.1 hardening tests across phases 01-05. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
