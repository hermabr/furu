# 02 — Slurm pool task completion correctness (do not mark done on failure)

## Problem statement (blocking)
In pool worker loop:
- After `_worker_entry()` returns, code unconditionally marks the task done (`src/furu/execution/slurm_pool.py:~190`).
- When `retry_failed=False`, `_worker_entry` may return early if compute_lock refused due to a **failed artifact**.
- This causes the task to be marked done even though Furu state is failed.
- `completed_hashes` then hides the failure and dependents may run incorrectly.

## Desired behavior
- After executing a task, worker must determine if the node is actually DONE (success marker / state success).
- If result is FAILED (and retry_failed is disabled), worker should:
  - mark task failed (queue/failed) and raise (fail the worker) OR
  - mark task failed and continue (controller will abort on failed queue anyway)
- If result is FAILED but retry_failed is enabled:
  - either mark failed (controller will schedule retry), or requeue (v2: simplest is mark failed and let controller handle retry policy explicitly)

## Implementation steps
- [x] In worker loop (after `_worker_entry()`), read state:
  - If result success => `_mark_done`
  - Else if result failed => `_mark_failed` with reason, and raise (or return non-zero)
  - Else (missing/unknown) => `_mark_failed` and raise
- [x] Consider fixing `_worker_entry()` itself:
  - If compute_lock not acquired because state is FAILED (and retries disabled), raise so job exits non-zero.
  - If not acquired because state is SUCCESS, return (noop).

## Acceptance criteria
- Pool never marks a failed node as done.
- Pool run aborts quickly when a task fails (via failed queue detection) and reports the failure clearly.
- Dependents never run when a required dependency is failed and retries are disabled.

## Tests to add/update
- [x] Create a failed node and run pool with retry_failed=False; ensure task ends in queue/failed and controller aborts.
- [x] Ensure no “done” marker is written for failed tasks.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Added post-worker state check to avoid marking failed/missing tasks as done.
- Notes: `_worker_entry()` now raises when lock acquisition fails due to failed state.
- Notes: Added pool worker test coverage for failed-state task handling.
