# plans/executor-fix-v6/03-pool-stale-heartbeat-bounds.md
# 03 — Pool: stale-heartbeat requeues must be bounded (consume attempts)

## Problem
Stale heartbeat requeue currently moves a task back to todo when heartbeat mtime exceeds `stale_sec`,
but does not increment attempt or track retries. A persistent heartbeat issue can therefore requeue
**forever** without reaching `max_compute_retries`.

## Decision
Stale-heartbeat requeue must:
- consume an attempt/retry (increment attempt counter), and
- be bounded by `max_compute_retries` semantics (v6 decision from subplan 01)

Additionally:
- if payload is invalid/unreadable, treat as protocol failure (fail fast)

## Implementation steps
- [x] In `src/furu/execution/slurm_pool.py`:
  - When stale heartbeat is detected:
    - load payload JSON (must be dict)
    - read `attempt` as int (else protocol failure)
    - if retries remaining: increment attempt and requeue
    - if retries exhausted: mark failed with clear reason (protocol) and abort run
  - Consider also tracking `stale_heartbeat_requeues` in payload for better diagnostics.
- [x] Ensure requeue payload updates are atomic (temp + replace), or at least robust against partial writes.
- [x] Ensure this interacts cleanly with compute_lock:
  - duplicate execution should still be prevented by locks/state

## Acceptance criteria
- Stale heartbeat can cause at most `max_compute_retries` retries (plus initial attempt).
- After retries exhausted, controller aborts with a clear message.
- No infinite churn.

## Tests
- [x] Regression test: stale heartbeat triggers bounded requeue and then terminal failure.
- [x] `make lint` and `make test`

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Increment attempts on stale heartbeat requeue; add bounded retry tests.
