# 05 — Dashboard version should come from package metadata

## Issue
Dashboard sets `__version__ = "0.1.0"` and FastAPI app version `"0.1.0"`, which can drift from the actual installed package.

## Desired behavior
Use installed package metadata:
- `importlib.metadata.version("furu")` (or correct distribution name)
- use that for:
  - `furu.dashboard.__version__`
  - FastAPI app `version=`
  - /health responses if present

## Implementation steps
- [x] Update `src/furu/dashboard/__init__.py`:
  - set `__version__` from `importlib.metadata.version` with a safe fallback
- [x] Update `src/furu/dashboard/main.py` app init `version=...`
- [x] Add a small test to ensure version is non-empty and matches package metadata when available.

## Progress log
- Status: [x] done
- Notes: Dashboard version now tracks package metadata with test coverage.
