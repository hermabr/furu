# 06 — Tests and docs updates (v4)

## Required tests (add if missing)
- [x] `_exists_quiet` exception policy:
  - validate raises domain validation error => treated as not-exists
  - validate raises unexpected error => surfaced clearly
- [x] pool error taxonomy:
  - spec mismatch => protocol => no retries
  - missing payload/invalid JSON => protocol => abort
  - compute failure => compute => retries when allowed
- [x] requeue metadata cleanup: stale error fields cleared
- [x] worker_entry diagnostic errors: include context
- [x] local retry parity decision: tests match chosen behavior

## Docs updates
- [x] `_validate()` contract documented (what exceptions are allowed/expected)
- [x] executor retry semantics documented (and differences between local vs pool if any)
- [x] pool failure taxonomy documented (compute vs protocol)

## Acceptance criteria
- `make test` and `make lint` passes.
- Tests are deterministic (avoid sleeps; use fake time/monkeypatch when possible).
- Docs match behavior.

## Progress log
- Status: [x] done
- Notes:
  - 2026-01-23: Confirmed required tests and updated README validation/retry docs.
