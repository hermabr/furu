# 01 — `_exists_quiet()` validation exception semantics

## Feedback
`_exists_quiet()` currently catches any `_validate()` exception and returns False.
Pros: scheduler loops don't crash.
Cons: masks validation bugs and can cause silent recompute loops, hiding real issues.

## Goal
Define and implement a safer policy that:
- doesn't crash polling loops for common/expected invalidation cases
- does surface real validation bugs clearly
- avoids infinite “recompute forever” when validation always throws

## Policy decision (pick one and implement)
A) **Treat validation exceptions as fatal in executors**:
   - raise `FuruValidationError` with context (hash, directory)
   - executors surface error and stop
   - interactive get may still treat it as invalidation (optional)

B) **Treat only known validation errors as non-fatal**:
   - introduce/standardize a `FuruValidationError` (or `FuruArtifactInvalid`) that `_validate()` may raise for “artifact missing/corrupt”
   - `_exists_quiet()` catches only that and returns False
   - all other exceptions propagate as fatal (better than blanket swallow)

C) **Treat all exceptions as invalid, but rate-limit and fail after N repeats**:
   - maintain a per-run counter keyed by furu hash
   - if `_validate()` throws >N times, raise a domain error to prevent silent loops

Recommended: **Option B** (narrowed catch), optionally with Option C’s “N repeats” safeguard.

## Implementation steps
- [x] Add/confirm a domain exception type (e.g., `FuruValidationError`) in `src/furu/errors.py`.
- [x] Update `_exists_quiet()`:
  - catch only the domain validation exception (or a small whitelist)
  - re-raise other exceptions with context
- [x] Update docs for `_validate()` contract:
  - return bool OR raise `FuruValidationError` for “invalid artifact”
  - avoid raising arbitrary exceptions without wrapping

## Acceptance criteria
- Unexpected exceptions in `_validate()` do not silently turn into endless recompute attempts.
- Expected invalidation cases still work without crashing scheduler loops.

## Tests
- [x] A Furu subclass whose `_validate()` raises `FuruValidationError` should be treated as “not exists” by planner/executors.
- [x] A Furu subclass whose `_validate()` raises `RuntimeError` should surface as an error (not swallowed) in `build_plan` or executor loop, with clear context.

## Progress log
- Status: [x] done
- Notes:
  - Added FuruValidationError and narrowed _exists_quiet handling; unexpected validate errors now surface with context logging.
  - Updated README/_validate docs and planner tests for both expected invalidation and unexpected errors.
