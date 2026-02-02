# plans/executor-plan/04-submitit-paths.md — Submitit logs under `<FURU_PATH>/submitit`

## Scope

- Place submitit logs under `<FURU_PATH>/submitit` by default.
- Allow override via env var `FURU_SUBMITIT_PATH` and optional executor parameters.
- Organize logs:
  - per-node submissions: `<submitit_root>/nodes/<spec_key>/...`
  - worker-pool workers: `<submitit_root>/workers/<spec_key>/...`

## Config changes (`src/furu/config.py`)

Add submitit root resolution:

```py
self.submitit_root = Path(
    os.getenv("FURU_SUBMITIT_PATH", str(self.base_root / "submitit"))
).expanduser().resolve()

def get_submitit_root(self) -> Path:
    return self.submitit_root
```

## Path helpers (`src/furu/execution/paths.py`)

```py
from pathlib import Path
from furu.config import FURU_CONFIG

def submitit_root_dir(override: Path | None = None) -> Path:
    return (override or FURU_CONFIG.get_submitit_root()).resolve()

def submitit_logs_dir(kind: str, spec_key: str, override: Path | None = None) -> Path:
    # kind: "nodes" or "workers"
    root = submitit_root_dir(override)
    return root / kind / spec_key
```

## Checklist

- [x] Add config `FURU_SUBMITIT_PATH` override + default `<FURU_PATH>/submitit`
- [x] Add `execution/paths.py` helpers
- [x] Ensure Slurm executors use these helpers for submitit folder selection

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Add submitit root config and path helper module. |
| 2026-01-22 | Wire Slurm executors to submitit log path helpers. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
