# 02 — Pool failure taxonomy (compute vs protocol) and retry rules

## Feedback
Pool worker currently treats many `_worker_entry()` exceptions as compute failures, but some errors are actually protocol/config problems:
- `FuruMissingArtifact` (dependency missing due to scheduling bug)
- `FuruSpecMismatch` (wrong spec worker)
- deserialization/payload errors
These should fail fast, not be retried as compute failures.

## Goal
Make pool worker classify failures into:
- **compute**: `_create()` failed; Furu state transitions to failed
- **protocol**: payload/queue/config/spec issues; cannot reliably recover by retrying

Only compute failures are eligible for retry when `retry_failed=True` (and bounded retries). Protocol failures always abort run.

## Implementation steps
- [x] Define a helper `classify_pool_exception(exc) -> failure_kind`:
  - protocol: `FuruMissingArtifact`, `FuruSpecMismatch`, JSON decode, KeyError missing payload, TypeError from `from_dict`, etc.
  - compute: `FuruComputeError` or generic Exception raised from inside `_create()` (if state says failed)
- [x] In `pool_worker_main`:
  - if exception occurs before `_worker_entry()` actually ran, mark protocol
  - if `_worker_entry()` ran but state is failed: mark compute
  - if state is missing/unknown: protocol
- [x] Ensure controller `_handle_failed_tasks` only retries compute failures.

## Acceptance criteria
- Protocol failures abort quickly with clear error.
- Compute failures retry only when allowed and bounded.
- No “retry storm” on permanent protocol misconfiguration.

## Tests
- [x] Pool: spec mismatch is protocol -> controller aborts (no retry).
- [x] Pool: missing payload / invalid JSON is protocol -> controller aborts.
- [x] Pool: compute failure is compute -> retried when retry_failed=True.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: 2026-01-23 added pool exception classification, protocol payload handling, and tests for protocol vs compute failures.
