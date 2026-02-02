# plans/executor-plan/03-specs-and-slurmspec.md — Spec keys `"default"` + SlurmSpec mapping

## Scope

- Define how nodes request resources via **spec keys** (strings).
- Define how Slurm parameters are provided at runtime via a **`specs` mapping**.
- Implement `SlurmSpec` dataclass used by Slurm executors.
- Enforce that executors require `specs["default"]` and error clearly when mappings are missing.

## Spec keys on nodes

Nodes declare spec requirements via:

```py
def _executor_spec_key(self) -> str:
    return "default"
```

Example override:

```py
class TrainGPU(Furu[Path]):
    def _executor_spec_key(self) -> str:
        return "gpu"
```

**Key rule:** spec keys must not affect hashing/serialization.

## SlurmSpec (executor-side)

Implement `src/furu/execution/slurm_spec.py`:

```py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SlurmSpec:
    partition: str | None = None
    gpus: int = 0
    cpus: int = 4
    mem_gb: int = 16
    time_min: int = 60
    extra: dict[str, object] | None = None
```

## Specs mapping required by Slurm executors

Slurm executors accept:

```py
specs: dict[str, SlurmSpec]
```

Required:
- must contain `"default"`
- for any node:
  - `key = obj._executor_spec_key()`
  - `spec = specs[key]` (raise KeyError with helpful message if missing)

Example:

```py
specs = {
    "default": SlurmSpec(partition="cpu", cpus=8, mem_gb=32, time_min=120),
    "gpu":     SlurmSpec(partition="gpu", gpus=1, cpus=8, mem_gb=64, time_min=720),
}
```

## Exact-match enforcement (v1)

- Workers/jobs are created per spec key.
- Worker-pool workers only execute tasks from their matching spec key queue.
- `get(force=True)` in executor contexts is allowed only if object spec key equals worker spec key.

## Checklist

- [x] Add `execution/slurm_spec.py`
- [x] Ensure Slurm executors require `specs["default"]`
- [x] Ensure errors include the missing spec key name and the node hash/class

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Add `SlurmSpec` and spec resolution helper with default/missing key checks. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
