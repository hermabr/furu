# 08 — Tests & docs updates

## Goals
Add targeted tests for newly enforced behavior and document semantic differences.

## Required tests (add if missing)
- [x] paths import test: `import furu.execution` succeeds
- [x] retry_failed semantics:
  - failed state becomes TODO when retry enabled
  - executor aborts when retry disabled
- [x] pool task completion correctness:
  - worker does not mark done on failed state
- [x] in-progress backend check (DAG):
  - IN_PROGRESS non-submitit fails fast
- [x] wait_for_job_id robustness:
  - simulate attempt switch and ensure loop returns correct job id
- [x] pool stale-running heartbeat requeue:
  - long task not requeued when heartbeat updates
  - task requeued when heartbeat stops
- [x] validate-failed-success recompute under executor

## Doc updates
- [x] Note: executor always_rerun semantics are “once per run” due to completed_hashes (if intentional).
- [x] Note: breaking changes / removed per-call override if applicable.

## Acceptance criteria
- `make test` and `make lint` passes.
- New tests are deterministic (avoid tight timing; use fake clocks or large margins).
- Docs mention intentional semantic differences.

## Progress log
- Status: [x] done
- Notes: Verified tests in executor/local/pool/slurm modules and README notes for always_rerun + retry_failed.
