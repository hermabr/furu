# 01 — Pool missing heartbeat policy (requeue at least once)

## Decision (requested)
Missing heartbeat should **not immediately** be treated as protocol failure.
We want to **requeue at least once** when heartbeat is missing beyond grace, because early worker crashes or filesystem hiccups can prevent heartbeat creation even though the node is safe to retry.

## Background
Current pool semantics include:
- heartbeat files for claimed running tasks
- a grace window for heartbeat creation
- stale-running requeue based on heartbeat age

Previously, missing heartbeat beyond grace was treated as protocol failure (abort). This is too strict for long cluster runs.

## Desired behavior
For each running task that lacks a heartbeat beyond `heartbeat_grace_sec`:
1) Requeue it once (move back to `queue/todo`) as a **compute retry** or **unknown retry**.
2) Record in payload a counter `missing_heartbeat_requeues` (or `protocol_retry_count`).
3) If missing heartbeat happens again for the same task and exceeds a cap (default 1):
   - then treat as protocol failure (abort), with a clear message.

This preserves “requeue at least once” while still preventing infinite churn on broken workers.

## Implementation steps
- [x] In `src/furu/execution/slurm_pool.py`:
  - Add a field to payload (or sidecar state) tracking missing heartbeat requeues.
  - Modify stale-running scan logic:
    - If heartbeat missing and beyond grace:
      - if counter < 1: requeue (increment counter)
      - else: mark protocol failure
  - Ensure requeue uses atomic file moves and does not crash workers.
- [x] Ensure worker heartbeat thread starts early enough and is robust:
  - If heartbeat cannot be written, worker should treat it as a warning but continue, since controller can requeue.
- [x] Ensure `_mark_done/_mark_failed` tolerate races (task moved by controller).

## Acceptance criteria
- A missing-heartbeat task is requeued at least once.
- If heartbeat never appears, the run eventually aborts with a clear protocol error (not infinite retry).
- No “double execution” occurs beyond what compute_lock already prevents.
- Behavior is covered by deterministic tests.

## Tests
- [x] Add/adjust test: missing heartbeat beyond grace triggers exactly one requeue, not immediate abort.
- [x] Add/adjust test: missing heartbeat twice triggers protocol failure + controller abort.
- [x] Ensure existing stale-running tests still pass.

## Progress log
- Status: [x] done
- Notes: Added missing heartbeat requeue counter and updated stale-running scan + tests.
