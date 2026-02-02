# 02 — Core: public `furu_hash` property

## Requirement (user request)
`furu_hash` must be a **public property**. External callers should not need to use `_furu_hash`.

## Desired behavior
- Add `@property def furu_hash(self) -> str:` returning the stable hash string.
- Update docs + README to reference `.furu_hash` (not `._furu_hash`).
- Keep `_furu_hash` internal (but it can remain as an implementation detail).

## Implementation steps
- [x] In `src/furu/core/furu.py`:
  - Add `@property def furu_hash(self) -> str: return self._furu_hash`
  - Ensure docstring clarifies stability and usage.
- [x] Update any code/docs/tests that refer to `_furu_hash` externally.
  - Internal usages can remain (optional to migrate gradually).
- [x] Consider exposing also `furu_dir` if docs currently point to private methods; optional.

## Acceptance criteria
- No README examples instruct users to access `_furu_hash`.
- Tests cover that `.furu_hash` matches the existing internal hash (and is stable).

## Tests
- [x] Unit test: `obj.furu_hash == obj._furu_hash`.
- [x] Grep-based test (optional): fail if README contains `_furu_hash`.

## Progress log
- Status: [x] done
- Notes: Added public furu_hash property, updated README/tests.
