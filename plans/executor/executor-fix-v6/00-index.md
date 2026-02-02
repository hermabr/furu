# plans/executor-fix-v6/00-index.md
# 00-index — Furu Fix v6 master checklist

## Implementation order (do in sequence)
1) Retry semantics: fix max-retries off-by-one + docs/tests → `01-max-retries-semantics.md`
2) Pool: claim timestamp / heartbeat grace should not use stale mtime → `02-pool-claim-timestamp.md`
3) Pool: bound stale-heartbeat requeues (consume attempts) → `03-pool-stale-heartbeat-bounds.md`
4) DAG: `_wait_for_job_id` should not time out on terminal attempts → `04-dag-terminal-jobid.md`
5) Core/docs: decide `_validate` exception contract + make executor robust → `05-validate-exception-contract.md`
6) Tests: add/adjust targeted regression tests for above → `06-tests.md`
7) Changelog: call out user-visible semantics changes (retry + validate) → `07-changelog.md`

## Master checklist
- [x] 01 Retry semantics: `max_compute_retries` matches “max retries”; local/pool updated; docs/tests updated
- [x] 02 Pool claim timestamp: touch/claimed_at prevents immediate missing-heartbeat requeue; regression test added
- [x] 03 Pool stale-heartbeat bounds: stale requeues consume attempts and are bounded; regression test added
- [x] 04 DAG job_id: `_wait_for_job_id` robust to terminal attempts; regression test added
- [x] 05 Validate contract: decision implemented; executor planning/get does not crash unexpectedly; docs updated
- [x] 06 Tests: coverage added for v6 changes; `make lint` and `make test` green
- [x] 07 Changelog: document behavior changes + migration notes

## Definition of done
All boxes checked, tests pass, branch pushed, PR exists.

## Progress log (append-only)
| Date | Item | Summary |
|---|---|---|
| 2026-01-23 | 01 | Treat max_compute_retries as retries; update local/pool logic, docs, tests. |
| 2026-01-23 | 02 | Touch running mtime on claim, honor claimed_at for grace checks, add regression test. |
| 2026-01-23 | 03 | Bound stale-heartbeat requeues with attempts and regression coverage. |
| 2026-01-23 | 04 | Return submitit job_id even if attempt terminal; add test coverage. |
| 2026-01-23 | 05 | Treat unexpected validate errors as invalid in executor planning; update docs/tests. |
| 2026-01-23 | 06 | Confirm regression coverage for v6 changes and re-run lint/tests. |
| 2026-01-23 | 07 | Update Unreleased changelog for retry, pool heartbeat, DAG job_id, validate changes. |
