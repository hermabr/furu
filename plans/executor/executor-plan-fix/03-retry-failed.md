# 03 — Executors: honor `FURU_CONFIG.retry_failed`

## Why

Executor schedulers currently ignore `FURU_CONFIG.retry_failed`.
`build_plan()` always classifies failed results as FAILED, and `run_local()` / `run_slurm_pool()`
abort immediately. This diverges from interactive `.get()` semantics and prevents executor runs
from recomputing failed nodes even when retry is enabled.

Relevant files mentioned:
- `src/furu/execution/plan.py:33`
- `src/furu/execution/local.py:97`
- `src/furu/execution/slurm_pool.py:270`

## Tasks

- [x] Update plan classification:
  - if node result is FAILED and `FURU_CONFIG.retry_failed` is enabled → treat as TODO/RETRY
  - if retry is disabled → keep FAILED

- [x] Update executor abort checks:
  - do not abort just because a failed node exists if retry is enabled

- [x] Ensure the compute path actually retries failed nodes:
  - any `compute_lock(... allow_failed=...)` must align with retry policy

- [x] Add tests:
  - With retry_failed enabled: create a failed node and confirm run_local retries it and completes.
  - With retry_failed disabled: failed node causes run_local/run_slurm_pool to fail fast.

## Implementation notes

Simplest classifier change:

```py
if isinstance(state.result, _StateResultFailed):
    return "TODO" if FURU_CONFIG.retry_failed else "FAILED"
```

Be careful:
- if you introduce a distinct status "RETRY", update all schedulers accordingly.

## Acceptance criteria

- Executor runs can recompute previously failed nodes when retry_failed is enabled.
- Executor runs fail fast when retry_failed is disabled and a required failed node exists.

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Honor retry_failed in dependency planning/executor guards and cover run_local/run_slurm_pool retries vs fail-fast behavior. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
