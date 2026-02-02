# 04 — `_worker_entry()` diagnostics and error types

## Feedback
`_worker_entry()` raises generic `RuntimeError` in some lock-refusal cases (e.g., already failed state).
This produces poor operator debugging information.

## Goal
- Use domain errors (`FuruComputeError`, `FuruLockNotAcquired`, `FuruMissingArtifact`, etc.) with context:
  - furu hash
  - directory/state file path
  - attempt status/backend
- Ensure pool/DAG behavior remains correct:
  - success lock refusal => noop
  - failed lock refusal with retry_failed=False => error (job fails)
  - in-progress lock refusal => treat as “already running elsewhere” with explicit message

## Implementation steps
- [x] In `src/furu/core/furu.py` `_worker_entry`:
  - when lock not acquired, read state and branch:
    - success: return
    - failed: raise `FuruComputeError` (or a dedicated `FuruAlreadyFailedError`) including state path/hash
    - in progress: raise a domain error indicating already running elsewhere
- [x] Ensure pool worker classifies these errors as protocol vs compute correctly (ties into subplan 02).

## Acceptance criteria
- Errors include actionable context (hash + directory + state path).
- Slurm DAG jobs don’t exit 0 on failed state when retry is disabled.
- Pool doesn’t misclassify lock-refusal errors.

## Tests
- [x] Unit test: worker_entry on failed state with retry_failed=False raises domain error containing hash/path.
- [x] Pool integration test: lock-refusal failed state leads to protocol/compute classification as intended.

## Progress log
- Status: [x] done
- Notes:
  - Raised domain errors for lock refusal with hash/state context; updated pool/test expectations.
