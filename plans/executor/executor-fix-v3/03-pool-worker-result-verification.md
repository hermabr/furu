# 03 — Pool worker result verification (avoid false “done”)

## Reported issue (blocking)
Some versions mark tasks done unconditionally after `_worker_entry()`, which can be wrong if:
- compute_lock refused due to FAILED state (retry_failed=False)
- `_worker_entry` returns early/no-op but state is still FAILED
=> task marked done, dependents run incorrectly

## Desired behavior
After `_worker_entry()` returns, worker must:
- read Furu state
- only `_mark_done` if state/result indicates SUCCESS
- otherwise `_mark_failed` with failure_kind set appropriately, and raise (or exit non-zero)

## Implementation steps
- [x] In `pool_worker_main` (src/furu/execution/slurm_pool.py):
  - call `_worker_entry()`
  - read state:
    - success => `_mark_done`
    - failed => `_mark_failed(failure_kind="compute")`
    - missing/unknown => `_mark_failed(failure_kind="protocol")`
  - ensure `_mark_done/_mark_failed` are tolerant to missing task files (FileNotFoundError)
- [x] Consider hardening `_worker_entry()` itself:
  - if lock-not-acquired due to FAILED and retry_failed=False => raise so job fails non-zero
  - if lock-not-acquired due to SUCCESS => noop return

## Acceptance criteria
- A failed artifact is never marked as done in the pool queue.
- Dependents cannot proceed when a required dep is failed and retry is disabled.

## Tests
- [x] Unit/integration test reproducing “lock refused / state failed” and verifying task ends as failed not done.

## Progress log
- Status: [x] done
- Notes:
  - Verified pool worker marks protocol failures for missing/incomplete states after `_worker_entry`.
