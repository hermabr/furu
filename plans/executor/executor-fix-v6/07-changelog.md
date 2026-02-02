# plans/executor-fix-v6/07-changelog.md
# 07 — Changelog: call out user-visible semantics changes

## Required notes
- [x] Retry semantics:
  - clarify `FURU_MAX_COMPUTE_RETRIES` meaning (retries vs attempts)
  - include an example of total attempts = 1 + retries
- [x] Pool heartbeat behavior:
  - claim timestamp / grace behavior clarification (if user-tunable knobs are documented)
  - stale heartbeat requeue bounded behavior
- [x] Validate contract:
  - clearly document chosen behavior (and any migration note for existing `_validate()` implementations)

## Acceptance criteria
- Users can understand what changed and why without reading code.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Updated Unreleased changelog bullets for retry, pool heartbeat, DAG job_id, and validate behavior.
