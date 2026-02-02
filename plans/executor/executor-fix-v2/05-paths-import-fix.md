# 05 — paths.py import/annotation fix (blocking)

## Problem statement (blocking)
`src/furu/execution/paths.py:~6` uses `Path` in type annotations without importing it or deferring evaluation.
On Python 3.12, annotations are evaluated unless `from __future__ import annotations` is present.
This can cause `NameError: name 'Path' is not defined` on import.

## Fix
Choose one:
- Add `from pathlib import Path`, OR
- Add `from __future__ import annotations` (and still import Path if used at runtime)

## Implementation steps
- [x] Open `src/furu/execution/paths.py`
- [x] Ensure `Path` is imported or annotations are deferred
- [x] Run `python -c "import furu.execution"` to confirm no import error

## Acceptance criteria
- Importing `furu.execution` does not raise NameError.

## Tests
- [x] Add minimal import test if not already present.

## Progress log
- Status: [x] not started / [ ] in progress / [ ] done
- Notes: Added import test; paths already had `Path` import.
