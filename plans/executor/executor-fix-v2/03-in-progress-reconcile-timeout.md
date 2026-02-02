# 03 — IN_PROGRESS hangs: reconciliation + stale timeout

## Problem statement (blocking)
Both `run_local` and `run_slurm_pool` contain logic that sleeps when any dependency is IN_PROGRESS:
- `src/furu/execution/local.py:~130`
- `src/furu/execution/slurm_pool.py:~346`

There is no reconciliation or stale timeout based on `stale_timeout`, so stale attempts can hang the scheduler forever.

## Desired behavior
- If plan contains IN_PROGRESS nodes, schedulers should not sleep forever.
- Before sleeping, they should either:
  - call `StateManager.reconcile(...)` (or equivalent) to update stale heartbeats/state, OR
  - check last heartbeat timestamps and fail after a configured timeout.
- Ideally, reuse existing config `FURU_CONFIG.stale_timeout` (or add executor-specific `in_progress_timeout_sec`).

## Implementation steps
- [x] Identify where `IN_PROGRESS` is blocking scheduling (ready queue empty but pending exists).
- [x] Add a reconciliation step before sleeping:
  - call `StateManager.reconcile(directory)` (if available) for IN_PROGRESS nodes
  - or implement a helper to detect staleness using attempt heartbeat timestamps in state.
- [x] Add a timeout:
  - if an IN_PROGRESS node has not advanced/heartbeated within timeout => mark as FAILED (if retries enabled, classify as TODO) OR raise with clear message.
- [x] Ensure local and pool share the same policy (prefer shared helper in `execution/plan.py` or `execution/util.py`).

## Acceptance criteria
- A stale IN_PROGRESS dependency cannot hang `run_local` or `run_slurm_pool` indefinitely.
- If stale is detected:
  - with retry_failed enabled: it becomes retryable
  - with retry_failed disabled: executor fails with clear error

## Tests
- [x] Create a synthetic IN_PROGRESS state with old heartbeat and assert executor fails or retries within timeout.
- [x] Ensure non-stale IN_PROGRESS sleeps a bounded amount and then re-checks/reconciles.

## Progress log
- Status: [x] done
- Notes:
  - Added shared reconciliation helper with stale heartbeat handling.
  - Added stale IN_PROGRESS coverage for local and slurm pool executors.
