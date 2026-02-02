# 00-index — Furu Fix v5 master checklist

## Implementation order (do in sequence)
1) Pool: missing heartbeat requeue-at-least-once policy → `01-pool-missing-heartbeat-policy.md`
2) Core: public `furu_hash` property → `02-core-public-furu-hash.md`
3) Dashboard: bundle dist assets into wheel → `03-dashboard-assets-packaging.md`
4) State: lock release safety (`fd is None` unlink bug) → `04-state-lock-safety.md`
5) Dashboard: version from package metadata → `05-dashboard-version.md`
6) Release workflow: run tests/build before publish → `06-release-workflow.md`
7) Dashboard security: path traversal hardening for SPA serving → `07-dashboard-spa-security.md`
8) Changelog updates for user-visible changes → `08-changelog.md`
9) Tests additions + CI-level checks → `09-tests.md`

## Master checklist
- [x] 01 Pool missing heartbeat: requeue at least once before protocol failure; tests updated
- [x] 02 Core: add `Furu.furu_hash` property; update docs/tests to use it
- [x] 03 Dashboard assets: ensure `frontend/dist` packaged in wheel; add install-time test
- [x] 04 State lock safety: do not unlink locks not owned; regression test
- [x] 05 Dashboard version: use `importlib.metadata.version("furu")` (or package name) in API + __init__
- [x] 06 Release workflow: run build/tests (or `make test-all`) before publishing
- [x] 07 SPA security: prevent path traversal; add test
- [x] 08 Changelog updated for: dashboard, state/metadata changes, executor changes
- [x] 09 Tests: targeted regression tests for above; `make lint` and `make test` green

## Definition of done
All boxes checked, tests pass, branch pushed, PR exists.

## Progress log (append-only)
| Date | Item | Summary |
|---|---|---|
| 2026-01-23 | 01 Pool missing heartbeat | Requeue missing heartbeats once before protocol failure; updated stale-running tests. |
| 2026-01-23 | 02 Core furu_hash | Added public furu_hash property and updated README/tests. |
| 2026-01-23 | 03 Dashboard assets | Included frontend dist assets in uv build and added importlib.resources test. |
| 2026-01-23 | 04 State lock safety | Guarded lock release when acquisition fails and added regression test. |
| 2026-01-23 | 05 Dashboard version | Synced dashboard version with package metadata and added test. |
| 2026-01-23 | 06 Release workflow | Run make build in release workflow before publishing. |
| 2026-01-23 | 07 SPA security | Blocked path traversal in SPA handler and added test. |
| 2026-01-23 | 08 Changelog | Documented dashboard, executor, and lock updates in Unreleased. |
| 2026-01-23 | 09 Tests | Added regression coverage and verified make lint/test. |
