# 06 — Pool stale-running requeue redesign (avoid mtime-only duplicates + tolerant mark_done)

## Problems (blocking + recommended)
- Current requeue logic requeues tasks based purely on queue file mtime (`src/furu/execution/slurm_pool.py:~289`).
- Any task longer than `stale_running_sec` (default 900s at ~328) can be requeued while still running.
- If controller requeues a still-running task:
  - worker later calls `_mark_done` / `task_path.replace(...)` on a path that no longer exists (~226)
  - worker crashes; task goes to failed queue; controller aborts (~354)

## Desired behavior
- Do NOT requeue legitimately running tasks based on file mtime alone.
- Requeue should be based on a **heartbeat** or worker liveness signal.
- `_mark_done` / `_mark_failed` should be tolerant to missing files (task already moved), and should not crash the worker in that case.

## Recommended v2 approach (simple + safe)
1) Add a **per-claimed-task heartbeat touch**:
   - When a worker claims a task (moves to running), it periodically touches a heartbeat file:
     - `queue/running/<spec>/<worker_id>/<hash>.hb`
   - Controller considers a task stale only if heartbeat is older than `stale_running_sec`.

2) If you cannot add heartbeats:
   - only requeue running tasks if **no workers exist for that spec**, OR
   - use a much larger default and document it

3) Make mark_done/mark_failed tolerant:
   - if `task_path` missing at mark time:
     - do not crash; log and continue (or mark worker-level warning)

## Implementation steps
- [ ] Implement heartbeat (preferred):
  - worker: touch hb file every N seconds while executing (or before/after `_worker_entry` at minimum)
  - controller: requeue only when hb stale
- [x] Implement heartbeat (preferred):
  - worker: touch hb file every N seconds while executing (or before/after `_worker_entry` at minimum)
  - controller: requeue only when hb stale
- [x] Update `_requeue_stale_running` to consult hb timestamps
- [x] Make `_mark_done` and `_mark_failed` robust to missing `task_path`:
  - catch FileNotFoundError on replace
  - ensure failed queue entry still created when JSON is invalid/missing

## Acceptance criteria
- Long-running tasks are not requeued while active.
- Worker does not crash when controller moves task files.
- If a worker crashes and heartbeat stops, controller eventually requeues safely.

## Tests
- [x] Test: task longer than stale_running_sec does NOT get requeued if heartbeat keeps updating.
- [x] Test: simulate worker crash (no hb updates) => controller requeues.
- [x] Test: mark_done does not crash on missing task_path.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
  - Added heartbeat file updates and stale requeue based on heartbeat mtime.
  - Made mark_done/mark_failed tolerate missing task files and invalid payloads.
  - Added tests for heartbeat behavior and tolerant marking.
