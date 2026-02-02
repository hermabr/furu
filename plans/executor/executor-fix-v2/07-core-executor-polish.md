# 07 — Core executor polish (log spam + validate-failed-success repair)

## Items
1) Reduce INFO log spam:
- Use `_exists_quiet()` (or equivalent) inside executor-mode `get()` to avoid frequent INFO logs from `exists()` (`src/furu/core/furu.py:~365`).

2) Validate-failed-success repair (critical correctness):
- If state/result indicates success but `_validate()` fails, planner may classify TODO.
- Executor force path should invalidate cached success and recompute; otherwise it can get stuck or treat invalid artifacts as done.

3) `_worker_entry` lock-not-acquired semantics:
- If compute_lock not acquired because success => noop
- If not acquired because failed (and retry disabled) => fail the job (exit non-zero) so Slurm afterok does not schedule dependents incorrectly.

## Implementation steps
- [x] Swap executor-mode get to call `_exists_quiet` (or non-logging exists) when checking for availability.
- [x] Add executor-force invalidation when validate fails but state says success:
  - call `_invalidate_cached_success(...)` before computing
- [x] Update `_worker_entry` to distinguish “already succeeded” vs “already failed” on lock refusal.

## Acceptance criteria
- Executor runs do not spam INFO logs on repeated dependency loads.
- Corrupt/invalid artifacts (validate fails) can be repaired via executor compute.
- Slurm jobs for failed nodes do not exit 0 when retry_failed is false.

## Tests
- [x] Add regression test: validate fails with success state => recompute under executor.
- [x] Add regression test: worker_entry on failed state with retry_failed false => raises.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
  - 2026-01-23: use quiet exists in executor get, invalidate cached success on validate failure, adjust worker-entry failure, add executor regression tests.
