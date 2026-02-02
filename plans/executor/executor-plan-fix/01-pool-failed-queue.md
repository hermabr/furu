# 01 — Slurm pool: fail fast when `queue/failed` contains entries

## Why

The Slurm pool controller can spin forever if a task lands in `queue/failed` **without updating Furu state**
(e.g., spec mismatch, missing payload, deserialize error). The controller does not inspect `queue/failed`, and
ready scheduling continues to think the node is runnable while `_task_known()` prevents requeue.

This must be fixed first because it causes non-terminating runs.

Relevant files mentioned in the bug report:
- `src/furu/execution/slurm_pool.py:180`
- `src/furu/execution/slurm_pool.py:288`
- `src/furu/execution/slurm_pool.py:346`

## Tasks

- [x] Add a controller-side check that scans `run_dir/queue/failed/**` every tick (or at least before enqueue).
  - If any failed entries exist, raise a clear exception showing:
    - task file path(s)
    - parsed payload (best effort)
    - recorded error message if present
  - The error should explain that the run stops to avoid an infinite loop.

- [x] Ensure `_task_known()` semantics do not silently block progress.
  - Minimum v1.1 behavior: `queue/failed` should be treated as **fatal**, not “known and ignore forever”.

- [x] Add a focused unit/integration test:
  - Create a fake run_dir with a failed task file.
  - Run one controller tick (or run_slurm_pool with a short poll interval).
  - Assert it raises quickly with a message that mentions `queue/failed`.

## Implementation notes

Recommended helper:

```py
def _check_failed_queue(run_dir: Path) -> None:
    failed_files = sorted((run_dir / "queue" / "failed").rglob("*.json"))
    if not failed_files:
        return
    # parse a handful and raise
```

Call early in the controller loop, before `ready_todo()`/enqueue.

If you also want to synchronize failures into state, do it as best-effort:
- if payload contains a serialized object, reconstruct and write FAILED state.
- but do not depend on that for termination; still fail fast.

## Acceptance criteria

- If a task exists in `queue/failed`, the controller stops within one poll interval.
- Error message is actionable (shows at least one failed task file and why).

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Added failed-queue detection with payload reporting and a regression test that fails fast on `queue/failed`. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
