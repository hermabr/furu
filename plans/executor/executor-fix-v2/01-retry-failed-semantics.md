# 01 — Retry failed semantics (planner + local + pool)

## Problem statement (blocking)
Current behavior (per PR review):
- `build_plan()` classifies failed nodes as **FAILED** unconditionally (`src/furu/execution/plan.py:~33`)
- `run_local()` and `run_slurm_pool()` abort if any FAILED node exists (`src/furu/execution/local.py:~97`, `src/furu/execution/slurm_pool.py:~270`)
=> Executor runs can **never retry** failures even when `FURU_CONFIG.retry_failed=True`.
This diverges from interactive `get()` semantics.

## Desired behavior
- If `FURU_CONFIG.retry_failed == True`:
  - Nodes with failed result should be classified as **TODO** (or RETRY) and scheduled again.
- If `FURU_CONFIG.retry_failed == False`:
  - Failed nodes should remain FAILED and executors should fail fast (clear error listing which nodes are failed).
- This behavior must be consistent across:
  - planner (`build_plan`)
  - `run_local`
  - `run_slurm_pool`
  - `submit_slurm_dag` (at minimum: don’t attempt to wire dependents to a FAILED dep without retry)

## Implementation steps
- [x] Update planner classification in `src/furu/execution/plan.py`:
  - When state/result indicates failure:
    - return TODO if retry_failed enabled
    - else return FAILED
  - Make sure any “failed dependency check” in executors matches this logic.
- [x] Update `run_local` and `run_slurm_pool` failure gating:
  - only abort on FAILED nodes (which now only occur when retry_failed is disabled)
- [x] Ensure execution path actually retries:
  - verify `compute_lock` is called with `allow_failed=True` when retries enabled
  - if force-compute paths bypass that, fix them

## Acceptance criteria
- With `retry_failed=True`, executor recomputes a previously failed node and completes.
- With `retry_failed=False`, executor fails fast before running dependents.
- No infinite waiting on a permanently failed node when retries are disabled.

## Tests to add/update
- [x] Add a test creating a failed node state then running `run_local` with retry_failed=True and asserting it recomputes.
- [x] Add a test ensuring `run_local` fails fast with retry_failed=False.
- [x] Add equivalent pool-mode test (small, uses inline worker mode if supported).

## Progress log
- Status: [x] done
- Notes:
  - Planner marks failed nodes TODO when retry_failed enabled; executors only fail fast when retry disabled.
  - Added local/pool retry coverage tests.
