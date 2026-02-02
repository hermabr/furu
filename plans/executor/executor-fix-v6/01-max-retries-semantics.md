# plans/executor-fix-v6/01-max-retries-semantics.md
# 01 — Retry semantics: max retries vs max attempts (off-by-one)

## Problem
`FURU_MAX_COMPUTE_RETRIES` / `FuruConfig.max_compute_retries` is described as **“max retries”**,
but current executor implementations behave like **“max attempts”** (off-by-one).

Observed symptom: with default 3, you get **2 retries** (3 attempts), not **3 retries** (4 attempts).

## Decision
Treat `max_compute_retries` as **max retries after the first failure**.

Therefore:
- total allowed attempts = `1 + max_compute_retries`
- if `max_compute_retries == 0`, you get exactly 1 attempt (no retries)

## Implementation steps
- [x] Local executor (`src/furu/execution/local.py`)
  - Adjust retry bookkeeping/comparison so `max_compute_retries` means retries, not attempts.
  - Update error messages to avoid confusing “attempt vs retries” wording.
- [x] Pool (`src/furu/execution/slurm_pool.py`)
  - Adjust compute-failure requeue comparison to allow `max_compute_retries` retries.
  - Ensure payload fields and logs communicate attempt count clearly.
- [x] Docs
  - Update README/config docs/comments describing `FURU_MAX_COMPUTE_RETRIES`.
  - If examples mention attempts, fix to “retries”.

## Acceptance criteria
- With `max_compute_retries=3`, a flaky node can fail 3 times and succeed on the 4th attempt.
- With `max_compute_retries=1`, a flaky node can fail once and succeed on the 2nd attempt.
- With `max_compute_retries=0`, a failure is terminal (no requeue).

## Tests
- [x] Add a deterministic “fails N times then succeeds” fixture (local executor).
- [x] Add a pool-level unit test for `_handle_failed_tasks` requeue count:
  - attempt=1 should requeue when max_compute_retries>=1
  - should stop after exhausting retries
- [x] `make lint` and `make test`

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Updated retry comparisons, README wording, and added local/pool tests.
