# plans/executor-fix-v6/06-tests.md
# 06 — Tests: targeted regression coverage for v6

## Goal
Add/adjust tests to cover v6 behavior changes without overfitting implementation details.

## Checklist
- [x] Retry semantics tests (local + pool failed queue handling)
- [x] Pool claim timestamp + missing-heartbeat grace regression
- [x] Pool stale-heartbeat bounded requeue regression
- [x] DAG terminal-attempt job_id wait regression
- [x] Validate exception contract regression (chosen behavior)

## Notes
- Prefer deterministic, unit-level tests for small helpers where possible.
- For pool behaviors, filesystem-based tests using tmp paths are OK.

## Commands
- [x] `make lint` and `make test`

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Added targeted regression tests across local, pool, and DAG behavior.
