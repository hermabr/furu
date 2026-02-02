# 09 — Tests and CI-level checks

## Goals
Add targeted regression tests for v5 changes and ensure CI catches packaging/security regressions.

## Tests to add/adjust
- [x] Pool missing heartbeat requeue-at-least-once behavior.
- [x] Public `furu_hash` property test + docs reference test (optional grep).
- [x] Dashboard assets existence via `importlib.resources` test.
- [x] State lock safety regression test (fd None must not unlink).
- [x] SPA traversal test (404).
- [x] Dashboard version metadata test (best-effort; may need conditional if metadata absent in editable mode).
- [x] Ensure `make lint` and `make test` covers these with deterministic timing (mock time where possible).

## Acceptance criteria
- `make lint` and `make test` passes.
- New tests fail on the pre-fix behavior and pass after fixes.
