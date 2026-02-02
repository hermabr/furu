# 01 — Policy and failure taxonomy

This subplan locks policy decisions so fixes are coherent and tests enforce the intended semantics.

## Decisions to encode (must do first)
- [x] Define two classes of task failures in pool mode:
  1) **Compute failure**: `_worker_entry()` executed, Furu state ends in FAILED because `_create()` raised.
     - If `FURU_CONFIG.retry_failed=True`, this is retryable.
     - If `retry_failed=False`, fail fast.
  2) **Protocol/queue failure**: worker cannot execute (spec mismatch, missing payload, invalid JSON, internal worker bug before running).
     - Always fatal: controller aborts immediately.

- [x] Define bounded retry policy for compute failures in pool mode:
  - Introduce `max_compute_retries` (default: 3; env `FURU_MAX_COMPUTE_RETRIES`).
  - Track attempts in task payload (`attempt` counter) starting at 1.
  - When retries exhausted: move to failed queue and abort (even if retry_failed=True).

- [x] Clarify retry_failed scope:
  - It should apply to **Furu artifact FAILED state**, not to protocol failures.
  - It should apply consistently across:
    - `run_local`
    - `run_slurm_pool`
    - optionally `submit_slurm_dag` behavior for previously FAILED artifacts.

## Implementation notes
- Prefer explicit fields in task JSON:
  - `"failure_kind": "compute" | "protocol"`
  - `"attempt": int`
  - `"error": str`
- The controller should inspect failed queue entries and:
  - abort immediately for protocol failures
  - requeue compute failures when retry is enabled and attempts remain
  - otherwise abort

## Acceptance criteria
- Policies are documented in this file AND encoded in code (not just comments).
- There is no contradictory behavior where retry_failed=True but pool aborts on the first compute failure without retry.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
  - 2026-01-23: Added failure_kind + attempt fields in slurm pool payloads; added max_compute_retries config.
