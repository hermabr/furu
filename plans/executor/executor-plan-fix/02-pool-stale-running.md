# 02 — Slurm pool: requeue tasks stranded in `queue/running` (worker crash recovery)

## Why

If a worker crashes after claiming a task (moving it from `todo` to `running`) but before completing it,
the task can remain in `queue/running` forever. The controller backlog currently only looks at `queue/todo`,
so it may stop spawning workers and spin forever with “ready nodes but no backlog”.

Relevant files mentioned in the concern:
- `src/furu/execution/slurm_pool.py:155`
- `src/furu/execution/slurm_pool.py:266`

## Tasks

- [x] Implement a controller reconcile step that requeues **stale** running tasks:
  - define parameter `stale_running_sec` (reasonable default for Slurm, e.g. 600–3600s)
  - scan `queue/running/**/<hash>.json`
  - if mtime older than threshold, move back to `queue/todo/<spec_key>/`
  - keep operation atomic (use rename/replace)

- [x] Ensure the reconcile step runs periodically (every tick is OK for v1.1).

- [x] Add a test:
  - create a task in `queue/running/<spec>/...` with old mtime
  - run one controller tick
  - assert it is moved back to `queue/todo/<spec>/`

## Implementation notes

Suggested helper:

```py
def _requeue_stale_running(run_dir: Path, stale_sec: float) -> int:
    ...
```

Edge cases:
- task may have been completed while scanning → ignore FileNotFound.
- if a todo file with same name exists, prefer failing fast or appending a suffix (choose one and test).

## Acceptance criteria

- Stranded running tasks eventually re-enter todo and the run progresses without hanging.

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Requeue stale running tasks via controller helper, add test coverage, and expose stale timeout. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
