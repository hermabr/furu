# 00-index — Executor Fix v4 master checklist

## Context
Executor v1 is close. v4 targets remaining correctness/ergonomics issues:
- validation exception handling in `_exists_quiet()`
- consistent error taxonomy (compute vs protocol) in pool mode
- retry behavior consistency and documentation
- clearer diagnostics for lock refusal / failed states
- targeted tests for new behavior

## Implementation order
1) `_exists_quiet` validation exception semantics → `01-exists-quiet-validation.md`
2) Pool error taxonomy (compute vs protocol) → `02-pool-error-taxonomy.md`
3) Requeue cleanup (stale error metadata) → `03-requeue-metadata-cleanup.md`
4) `_worker_entry` diagnostics + error types → `04-worker-entry-errors.md`
5) Local executor retry parity OR explicit docs → `05-local-retry-parity.md`
6) Tests + docs → `06-tests-and-docs.md`

## Master checklist
- [x] 01 `_exists_quiet` handles `_validate()` exceptions with correct semantics (no silent infinite loops)
- [x] 02 Pool worker classifies failures (compute vs protocol) and retries only compute failures when allowed
- [x] 03 Requeued tasks do not retain stale error/failure_kind metadata
- [x] 04 `_worker_entry` raises informative domain errors (not generic RuntimeError) and includes context
- [x] 05 Decide and implement/document local retry parity vs pool retry semantics
- [x] 06 Add/adjust tests and docs; `make test` and `make lint` green

## Definition of done
All items checked, tests pass, branch pushed, PR exists.

## Progress log (append-only)
| Date | Item | Summary |
|---|---|---|
| 2026-01-23 | 01 | Added FuruValidationError, narrowed quiet validation handling, and updated planner tests/docs. |
| 2026-01-23 | 02 | Classified pool protocol vs compute failures, tightened payload handling, and added protocol retry tests. |
| 2026-01-23 | 03 | Cleared stale error metadata on requeue and added unit coverage for payload cleanup. |
| 2026-01-23 | 04 | Raised domain errors on worker lock refusal with context and updated pool/worker tests. |
| 2026-01-23 | 05 | Added bounded local compute retries to mirror pool behavior, plus regression coverage. |
| 2026-01-23 | 06 | Verified executor tests and documented validation/retry semantics in README. |
