# 00-index — Executor Fix v3 master checklist

## Summary of reported issues to address

### Blockers (must fix)
- Pool worker must not mark failed artifacts as done (verify state after `_worker_entry`).
- Pool controller must not spin forever on tasks in `queue/failed` (detect and act).
- Executors must honor `FURU_CONFIG.retry_failed=True` (planner + local + pool; and pool must not contradict this by failing fast on any compute failure).
- IN_PROGRESS waits must reconcile / timeout (no indefinite sleep).
- Slurm DAG must fail fast on IN_PROGRESS deps that are not submitit-backed (avoid timeouts with generic error).
- `paths.py` import/annotation bug: `Path` name error on import (if still present).

### Major behavior hazards
- Pool requeue must not be mtime-only; avoid premature requeue for long tasks.
- Planner stale detection must not treat missing timestamps as stale immediately (upgrade compatibility).
- Missing heartbeat file behavior must be defined (don’t requeue blindly based on task mtime fallback).
- `_exists_quiet()` should not crash the scheduler loop if `_validate()` raises.
- `_set_submitit_job_id()` should update job_id if it differs for the active queued/running attempt.

### Tests/docs gaps
- Explicit tests for retry_failed semantics in executors (local + pool).
- Test: pool aborts on protocol/queue failure (failed queue entry injection).
- Test: stale IN_PROGRESS reconcile/timeout, including missing timestamps.
- Test: missing heartbeat file handling.
- DAG tests: non-submitit in-progress dep fails fast; `_wait_for_job_id` robustness; job_id update mismatch.

## Implementation order (do in sequence)
1) **Policy decision + failure taxonomy** → `01-policy-and-taxonomy.md`
2) **Retry failed semantics end-to-end** → `02-retry-failed-semantics.md`
3) **Pool worker result verification (no false “done”)** → `03-pool-worker-result-verification.md`
4) **Pool failed-queue handling + retry loop** → `04-pool-failed-queue-and-retries.md`
5) **IN_PROGRESS reconcile + stale timeout (and missing timestamp compatibility)** → `05-inprogress-reconcile-timeout.md`
6) **Slurm DAG robustness (backend checks + job_id updates + param merging + run_id)** → `06-slurm-dag-robustness.md`
7) **Pool heartbeat/requeue correctness (missing heartbeat semantics)** → `07-pool-heartbeat-and-requeue.md`
8) **Core polish: `_exists_quiet` exception handling** → `08-core-exists-quiet.md`
9) **Tests + docs** → `09-tests-and-docs.md`

## Master checklist
- [x] 01 Policy/taxonomy decisions documented and encoded
- [x] 02 retry_failed semantics honored by planner + local + pool (and documented)
- [x] 03 pool worker verifies state before marking done; failed state handled correctly
- [x] 04 pool: differentiate compute-fail vs protocol-fail; bounded retry; no infinite spin
- [x] 05 in-progress reconcile/timeout; missing timestamps not treated as immediately stale
- [x] 06 slurm dag: non-submitit in-progress fails fast; job_id update mismatch handled; merge additional params; expose run_id
- [x] 07 pool: heartbeat/requeue semantics safe (no mtime-only premature requeue; missing heartbeat behavior defined)
- [x] 08 core: `_exists_quiet` treats validate exceptions safely
- [x] 09 tests/docs updated; `make test` and `make lint` green

## Definition of done
All items above checked, tests pass, branch pushed, PR exists.

## Progress log (append-only)
| Date | Item | Summary |
|---|---|---|
| 2026-01-23 | 01 | Documented failure taxonomy + retry policy; added payload fields/config. |
| 2026-01-23 | 02 | Verified planner classification respects retry_failed. |
| 2026-01-23 | 02 | Confirmed local/pool retry_failed gates; pass allow_failed into pool workers. |
| 2026-01-23 | 02 | Documented pool protocol failures as fatal regardless of retry_failed. |
| 2026-01-23 | 02 | Marked local/pool retry_failed tests complete. |
| 2026-01-23 | 03 | Ensure pool worker marks protocol failures for missing/incomplete state. |
| 2026-01-23 | 04 | Added failed-queue scanning, retry requeue path, and detailed failure messages. |
| 2026-01-23 | 05 | Added reconcile helper + fallback timestamps; added stale/missing timestamp tests. |
| 2026-01-23 | 06 | Updated slurm DAG job_id refresh + wait handling; verified param merge/run_id. |
| 2026-01-23 | 07 | Added heartbeat grace handling + missing heartbeat tests for pool requeue. |
| 2026-01-23 | 08 | Wrapped _exists_quiet validation errors and added planner coverage. |
| 2026-01-23 | 09 | Verified tests/docs coverage and green make test/lint. |
