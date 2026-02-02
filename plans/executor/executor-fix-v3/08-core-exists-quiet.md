# 08 — Core: `_exists_quiet` should handle `_validate` exceptions safely

## Reported issue
`_exists_quiet()` calls `_validate()` directly. If validation raises, planner/executor loops can crash.

## Desired behavior
Decide intended semantics:
- If validation exceptions should mean “artifact invalid”: treat as not-exists (return False), possibly logging debug/warn.
- If validation exceptions should be fatal: raise a dedicated `FuruValidationError` with context.
Recommended for executor polling: treat as not-exists, but include diagnostic logs.

## Implementation steps
- [x] Wrap `_validate()` in try/except inside `_exists_quiet()`.
- [x] On exception:
  - return False
  - log at DEBUG/WARN with exception summary and object identifier
- [x] Ensure planner uses `_exists_quiet()` and does not crash due to validate exceptions.

## Tests
- [x] Create a Furu subclass whose `_validate()` raises; ensure planner and executors treat it as not-exists and proceed (or raise the intended dedicated error).

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
- `_exists_quiet()` now logs and treats validation exceptions as missing artifacts.
- Added planner test for validation error handling.
