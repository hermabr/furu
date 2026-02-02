# plans/executor-fix-v6/02-pool-claim-timestamp.md
# 02 — Pool: claim timestamp so missing-heartbeat grace is correct

## Problem
Pool missing-heartbeat logic uses the **running task file mtime** to decide whether the worker
has exceeded `heartbeat_grace_sec` without creating a heartbeat.

But when a worker claims a task via rename (`todo → running`), the file mtime can remain **old**
(e.g., if the task sat in todo a long time). That can cause **immediate requeue** before the worker
even has a chance to write a heartbeat.

## Decision
On successful claim, establish a “claimed now” timestamp so grace is measured from claim time.

Preferred approach (minimal + robust):
- After `path.replace(target)` in claim:
  - **touch `target`** (`os.utime`) so mtime reflects claim time
- Optional upgrade:
  - embed `claimed_at` into payload for explicitness and future-proofing

## Implementation steps
- [x] In `src/furu/execution/slurm_pool.py`:
  - Update `_claim_task` to touch the running file immediately after successful replace.
  - If adding `claimed_at`, update payload write using an atomic write (temp + replace).
- [x] Ensure missing-heartbeat grace logic uses:
  - `claimed_at` if present, else `mtime` (fallback)

## Acceptance criteria
- A task with very old todo mtime does **not** get requeued immediately on claim.
- Missing heartbeat beyond grace still triggers the existing policy (requeue-at-least-once, then fail).

## Tests
- [x] Regression test:
  - create a todo task file with old mtime
  - simulate claim (or call `_claim_task`)
  - ensure missing-heartbeat grace logic does not immediately treat it as beyond grace
- [x] `make lint` and `make test`

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Touch running task mtime on claim; add regression test coverage.
