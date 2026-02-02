# plans/executor-plan/01-core-get.md — Core: remove `load_or_create`, implement `Furu.get()`

## Scope

- Remove `Furu.load_or_create` entirely (no alias, no compatibility).
- Implement `Furu.get(force: bool = False)` as the only public method.
- Enforce executor strictness via `EXEC_CONTEXT` (see 02-exec-context.md).
- Implement escape hatch `force=True` with **exact spec key match**.
- Ensure local compute can be run inside executor context for local parallel workers.

## Required invariants

- Interactive mode: `obj.get()` computes missing and may compute deps recursively.
- Executor mode: `obj.get()` is load-only; missing raises `FuruMissingArtifact`.
- Executor mode: `obj.get(force=True)` can compute **only if**
  - `obj._executor_spec_key() == EXEC_CONTEXT.spec_key`.

## Edits: `src/furu/core/furu.py`

### 1) Delete `load_or_create` (no compatibility)

- Remove the method definition.
- Delete any code paths that accept an `executor=` argument via `load_or_create` / `get`.
- Submission happens only through new executor modules (`furu.execution.*`).

### 2) Add spec key method

Add to `Furu` base class:

```py
def _executor_spec_key(self) -> str:
    # v1 default
    return "default"
```

Subclasses override this method to request `"gpu"`, `"cpu_big"`, etc.

### 3) Add errors

In `src/furu/errors.py` add:

```py
class FuruMissingArtifact(FuruError):
    pass

class FuruSpecMismatch(FuruError):
    pass
```

### 4) Implement `get(force=...)`

The final code must preserve your current interactive semantics:
- alias resolution / migration
- success marker fast path
- validate / always_rerun invalidation
- compute_lock wait/retry/reconcile
- local compute via `_run_locally`

**Pseudo-structure (final code should map to existing logic, not rewrite it):**

```py
def get(self, *, force: bool = False) -> T:
    from furu.execution.context import EXEC_CONTEXT
    from furu.errors import FuruMissingArtifact, FuruSpecMismatch

    ctx = EXEC_CONTEXT.get()

    # EXECUTOR MODE: strict load-only unless force=True with exact-match
    if ctx.mode == "executor":
        if self.exists():
            return self._load()

        if not force:
            raise FuruMissingArtifact(
                f"Missing artifact {self.__class__.__name__}({self._furu_hash}) in executor mode. "
                f"Requested by {ctx.current_node_hash}. Declare it as a dependency."
            )

        required = self._executor_spec_key()
        if ctx.spec_key is None or required != ctx.spec_key:
            raise FuruSpecMismatch(
                f"force=True not allowed: required={required!r} != worker={ctx.spec_key!r} (v1 exact match)"
            )

        # Allowed: compute self while keeping executor semantics strict for deps
        status, created_here, result = self._run_locally(start_time=time.time(), executor_mode=True)
        if status != "success":
            raise FuruComputeError(...)
        return result if created_here else self._load()

    # INTERACTIVE MODE: preserve old local compute semantics
    return self._get_impl_interactive(force=force)  # name can vary; must be private and clearly “impl”
```

You requested “no `_get_interactive` method” and “no backwards compatibility”.
- A private helper is still fine for structure, but it must be a pure implementation detail.

### 5) Add `executor_mode` to `_run_locally`

Change signature:

```py
def _run_locally(self, start_time: float, *, executor_mode: bool = False) -> tuple[str, bool, T | None]:
    ...
```

Before `_create()` is invoked under the compute_lock, if `executor_mode=True`, set executor context:

```py
from furu.execution.context import EXEC_CONTEXT, ExecContext

token = None
if executor_mode:
    token = EXEC_CONTEXT.set(ExecContext(
        mode="executor",
        spec_key=self._executor_spec_key(),
        backend="local",
        current_node_hash=self._furu_hash,
    ))
try:
    result = self._create()
finally:
    if token is not None:
        EXEC_CONTEXT.reset(token)
```

This makes local-parallel execution safe: deps inside `_create()` are strict.

### 6) Update `_worker_entry` to compute via `get(force=True)`

`_worker_entry` must set executor context and then compute self:

```py
from furu.execution.context import EXEC_CONTEXT, ExecContext

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

This guarantees that within `_create()`, dependency `.get()` calls are strict load-only.

### 7) Guard signal handler installation

`_setup_signal_handlers` must no-op in worker threads:

```py
import threading
if threading.current_thread() is not threading.main_thread():
    return
```

This is required because local parallel executor uses `ThreadPoolExecutor`.

## Checklist

- [x] Remove `load_or_create` from `Furu`
- [x] Add `Furu._executor_spec_key()` default `"default"`
- [x] Add errors `FuruMissingArtifact`, `FuruSpecMismatch`
- [x] Implement executor strict behavior in `get(force=...)`
- [x] Add `executor_mode` to `_run_locally` and set context when computing
- [x] Ensure `_worker_entry` sets context and calls `get(force=True)`
- [x] Guard signal handler install in threads

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Implement `get`, add executor strictness, and update worker context handling. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
| 2026-01-22 | Keep submitit worker lock flow; set executor context around `_create` instead of delegating to `get(force=True)` to preserve scheduler metadata. | Preserve submitit attempt metadata. |
