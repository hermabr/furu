# plans/executor-plan/02-exec-context.md — Execution context (`contextvars`) and strict mode

## Scope

- Add a runtime execution context that:
  - defaults to interactive mode
  - can be set by local executor threads and Slurm worker entrypoints
  - carries the current worker’s spec key (string) for `force=True` compatibility checks
- Ensure `Furu.get()` can switch behavior based on this context.

## Implementation: `src/furu/execution/context.py`

Create:

```py
from __future__ import annotations
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Literal

Backend = Literal["local", "submitit"]

@dataclass(frozen=True)
class ExecContext:
    mode: Literal["interactive", "executor"]
    spec_key: str | None = None
    backend: Backend | None = None
    current_node_hash: str | None = None

EXEC_CONTEXT: ContextVar[ExecContext] = ContextVar(
    "FURU_EXEC_CONTEXT",
    default=ExecContext(mode="interactive", spec_key=None, backend=None, current_node_hash=None),
)

def in_executor() -> bool:
    return EXEC_CONTEXT.get().mode == "executor"
```

## Required usage patterns

Executors must set context explicitly per worker execution.

### Local parallel worker wrapper example

```py
token = EXEC_CONTEXT.set(ExecContext(
    mode="executor",
    spec_key=node.obj._executor_spec_key(),
    backend="local",
    current_node_hash=node.obj._furu_hash,
))
try:
    node.obj.get(force=True)
finally:
    EXEC_CONTEXT.reset(token)
```

### Submitit / Slurm worker entrypoint example

Inside `Furu._worker_entry()`:

```py
token = EXEC_CONTEXT.set(ExecContext(
    mode="executor",
    spec_key=self._executor_spec_key(),
    backend="submitit",
    current_node_hash=self._furu_hash,
))
try:
    self.get(force=True)
finally:
    EXEC_CONTEXT.reset(token)
```

## Notes on threads

- `contextvars` do not automatically propagate into new threads.
- For local executor, you must set the context inside the worker function itself.
- Avoid relying on implicit propagation.

## Checklist

- [x] Add `execution/context.py`
- [x] Ensure `Furu.get()` reads `EXEC_CONTEXT`
- [x] Ensure `_run_locally(executor_mode=True)` sets executor context around `_create()`
- [x] Ensure `_worker_entry` sets executor context

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Add execution context module and wire it into get, local runs, and submitit workers. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
