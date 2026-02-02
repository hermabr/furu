# plans/executor-plan/07-local-executor.md — Local parallel executor (`run_local`)

## Scope

Implement a local parallel executor that:
- builds a dependency plan from roots
- schedules nodes whose deps are DONE
- executes nodes in a `ThreadPoolExecutor`
- sets executor context for each node computation (so deps are strict)
- respects window policy (`dfs`/`bfs`/k)

## File: `src/furu/execution/local.py`

### Public API (v1)

```py
def run_local(
    roots: list[Furu],
    *,
    max_workers: int = 8,
    window_size: str | int = "bfs",
    poll_interval_sec: float = 0.25,
) -> None:
    ...
```

### Window policy (v1)

Normalize:
- `"dfs"` → 1 active root
- `"bfs"` → all roots active
- `int k` → rolling window: keep up to `k` roots active; activate new roots as previous complete

### Execution entrypoint

For each scheduled node:

```py
token = EXEC_CONTEXT.set(ExecContext(
    mode="executor",
    spec_key=node.obj._executor_spec_key(),
    backend="local",
    current_node_hash=node.obj._furu_hash,
))
try:
    node.obj.get(force=True)   # compute this node
finally:
    EXEC_CONTEXT.reset(token)
```

Dependencies inside `_create()` call `.get()` which is strict in executor mode.

### Scheduler loop sketch (v1)

- Maintain:
  - `active_roots` indices (window policy)
  - `inflight` futures
  - `done_hashes` (when node exists after compute)
- On each tick:
  - build plan from active roots
  - compute ready TODO nodes (deps DONE)
  - submit up to `max_workers - len(inflight)` tasks
  - as tasks finish:
    - on success: mark done
    - on failure: raise and stop

Optional: treat IN_PROGRESS nodes as “not ready” and just poll (v1 can keep it simple).

### Thread safety: signal handlers

Ensure `Furu._setup_signal_handlers` does nothing in worker threads (implemented in 01).

## Checklist

- [x] Implement `run_local` public API
- [x] Implement window normalization (`dfs`/`bfs`/k)
- [x] Implement dynamic ready queue + `ThreadPoolExecutor`
- [x] Set `EXEC_CONTEXT` in worker wrapper and call `get(force=True)`
- [x] Fail-fast on node failure; surface which node failed and where state is stored

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Implement local executor scheduler, window policy, context handling, and tests. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
