# plans/executor-fix-v6/05-validate-exception-contract.md
# 05 — Validate exception contract: executor robustness vs strictness

## Problem
`_exists_quiet()` currently:
- returns False on `FuruValidationError`
- **re-raises** unexpected exceptions from `_validate()`

This can crash executor planning/get loops if legacy `_validate()` implementations raise
`FileNotFoundError` / `OSError` / parsing errors instead of returning False or raising `FuruValidationError`.

README may document a “new contract”, but this is a breaking behavior and can be painful in practice.

## Decision (pick one and implement)
Option A (recommended for robustness):
- `_exists_quiet()` should **never crash executor planning**.
- Treat *any* exception from `_validate()` as “invalid” (return False) while logging an exception.
- Keep `exists()` behavior unchanged (or document if changed).

Option B (keep strict contract):
- Keep re-raising in `_exists_quiet()`, but:
  - make the breaking change explicit in CHANGELOG
  - improve error messages to instruct users to raise `FuruValidationError` for invalid artifacts
  - add tests to lock in behavior

## Implementation steps
- [x] Implement the chosen option in `src/furu/core/furu.py` (`_exists_quiet`).
- [x] Ensure executor mode `get(force=True)` still invalidates cached success correctly when validate fails/crashes.
- [x] Update docs:
  - README + any validate guidance
  - add “migration note” for existing `_validate()` implementations
- [x] Add targeted test coverage:
  - a furu with `_validate()` raising `FileNotFoundError` should not hard-crash the executor loop (Option A),
    or should raise with clear message (Option B).

## Acceptance criteria
- Executor planning loops do not unexpectedly crash without actionable diagnostics.
- Behavior is documented clearly (README + CHANGELOG).

## Tests
- [x] Add regression test for `_validate()` raising non-FuruValidationError.
- [x] `make lint` and `make test`

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Treat unexpected validate errors as invalid in executor planning and document behavior.
