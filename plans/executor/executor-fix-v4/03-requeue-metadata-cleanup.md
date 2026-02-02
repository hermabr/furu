# 03 — Requeue metadata cleanup

## Feedback
`_requeue_failed_task()` retains prior `error` / `failure_kind` fields in the payload.
This can confuse diagnostics because a “fresh attempt” still carries old failure metadata even after requeue.

## Goal
When requeueing a task:
- preserve stable identity fields (hash/spec/obj payload)
- clear stale fields:
  - `error`, `traceback`, `failure_kind`, `failed_at`, `worker_id`, etc.
- increment attempt counter cleanly

## Implementation steps
- [x] Update `_requeue_failed_task()` in `src/furu/execution/slurm_pool.py`:
  - remove stale fields on requeue
  - set `attempt += 1`
  - optionally record `previous_errors` list if you want history (nice-to-have)
- [x] Ensure tests cover the payload after requeue.

## Acceptance criteria
- A retried task payload does not contain stale error/failure_kind fields.
- Attempt count increments deterministically.

## Tests
- [x] Unit test: create a failed task JSON with error fields, requeue it, ensure cleared fields and incremented attempt.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Cleared stale failure metadata during requeue and added unit coverage.
