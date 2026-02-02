# 02 — retry_failed semantics (planner + local + pool)

## Reported issue (blocking)
Executors ignore `FURU_CONFIG.retry_failed=True` in some versions:
- Planner `_classify()` returns FAILED unconditionally.
- Local/pool abort when any FAILED nodes exist.
=> failures never retried despite config.

## Desired behavior
- When `FURU_CONFIG.retry_failed=True`:
  - planner classifies failed nodes as TODO/RETRY
  - executors do not abort just because a node is failed in state
  - compute paths call compute_lock with `allow_failed=True`
- When `retry_failed=False`:
  - failed nodes remain FAILED
  - executors abort early with clear error listing failed nodes

## Implementation steps
- [x] Update `src/furu/execution/plan.py` classification:
  - failed -> TODO when retry enabled; failed -> FAILED otherwise
- [x] Update abort gates in:
  - `src/furu/execution/local.py`
  - `src/furu/execution/slurm_pool.py`
- [x] Ensure compute paths pass allow_failed based on config (or equivalent)
- [x] Add/confirm docs describing executor retry behavior (and how it differs from pool protocol failures)

## Acceptance criteria
- With retry_failed enabled, local executor recomputes a failed node and completes.
- With retry_failed disabled, executor fails fast with an explicit error listing failed nodes.

## Tests
- [x] Add tests for local retry_failed semantics.
- [x] Add tests for pool retry_failed semantics (compute failure retry).

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
  - Confirmed `_classify()` already routes failed to TODO when retry_failed.
  - 2026-01-23: Verified local/pool abort gates respect retry_failed; pass allow_failed to pool worker entry.
  - 2026-01-23: Documented pool protocol failures as fatal regardless of retry_failed.
  - 2026-01-23: Tests already cover local/pool retry_failed semantics (test_local_executor/test_slurm_pool).
