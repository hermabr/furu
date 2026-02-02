# plans/executor-plan/09-slurm-pool.md — Slurm worker-pool (“mass-parallel”) via filesystem queue

## Scope

Implement worker-pool mode that:
- reduces scheduler overhead by submitting a small number of longer-lived worker jobs
- workers pull tasks from a shared filesystem queue and run many nodes sequentially
- supports mixed specs by having separate queues/workers per spec key (exact match only)
- has a global `max_workers_total`
- supports window policy (`dfs`/`bfs`/k) over root activation
- logs under `<submitit_root>/workers/<spec_key>/...`

## File: `src/furu/execution/slurm_pool.py`

## Public API (v1)

```py
@dataclass
class SlurmPoolRun:
    run_dir: Path
    submitit_root: Path
    plan: DependencyPlan

def run_slurm_pool(
    roots: list[Furu],
    *,
    specs: dict[str, SlurmSpec],
    max_workers_total: int = 50,
    window_size: str | int = "bfs",
    idle_timeout_sec: float = 60.0,
    poll_interval_sec: float = 2.0,
    submitit_root: Path | None = None,
    run_root: Path | None = None,
) -> SlurmPoolRun:
    ...
```

Validate:
- `specs` must contain `"default"`

## Window policy (v1)

Normalize:
- `"dfs"` → 1
- `"bfs"` → len(roots)
- `int k` → rolling window of k active roots:
  - activate first k roots
  - as soon as a root completes, activate next root, maintaining k until exhausted

## Run directory layout (v1)

Default:
- `<FURU_PATH>/runs/<timestamp>-<random>/`

Inside:

```
queue/
  todo/<spec_key>/
  running/<spec_key>/<worker_id>/
  done/
  failed/
```

Each task file is JSON named `<furu_hash>.json`:

```json
{
  "hash": "abc123...",
  "spec_key": "default",
  "obj": { "type": "...", "fields": { ... } }
}
```

## Orchestrator responsibilities (v1)

Loop until all roots are DONE:

1. Maintain active roots per window policy.
2. Build plan for active roots (v1: rebuild each tick is OK).
3. Abort if any FAILED node is required.
4. Determine ready TODO nodes: deps must be DONE.
5. Enqueue ready nodes into `queue/todo/<spec_key>/` if not already enqueued or completed.
6. Scale workers:
   - compute backlog per spec (count files in todo/<spec>)
   - while total_workers < max_workers_total and any backlog exists:
     - choose spec with largest backlog
     - submit one worker job for that spec
7. Completion:
   - root complete when `root.exists()` becomes true.

## Worker function (v1)

Workers are spec-homogeneous; they only pull from their own spec key directory.

```py
def pool_worker_main(run_dir: Path, spec_key: str, idle_timeout_sec: float, poll_interval_sec: float):
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    last_task_time = time.time()

    while True:
        task = claim_task(run_dir, spec_key, worker_id)  # atomic rename todo -> running
        if task is None:
            if time.time() - last_task_time > idle_timeout_sec:
                return
            time.sleep(poll_interval_sec)
            continue

        last_task_time = time.time()
        try:
            obj = Furu.from_dict(task["obj"])
            if obj._executor_spec_key() != spec_key:
                raise RuntimeError(f"Spec mismatch: task {obj._executor_spec_key()} on worker {spec_key}")
            obj._worker_entry()  # sets executor context; strict get inside
            mark_done(...)
        except Exception as e:
            mark_failed(..., str(e))
            raise
```

**Kill behavior:** Workers exit after idle timeout. Orchestrator does not cancel in v1.

## Mixed-spec dependencies (why this works)

- default deps enqueue under `"default"`
- gpu node remains blocked until deps exist
- once deps exist, gpu node enqueues under `"gpu"`
- gpu worker runs it; inside `_create`, dep `.get()` loads-only

## Checklist

- [x] Implement run directory creation and queue layout
- [x] Implement task enqueue (dedupe by hash)
- [x] Implement claim_task via atomic rename
- [x] Implement worker loop and done/failed marking
- [x] Implement controller loop: plan rebuild, ready detection, worker scaling
- [x] Implement window policy and rolling activation
- [x] Ensure spec mismatch in worker is detected and fails loudly

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Implement slurm pool queue, worker, controller, and tests. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
