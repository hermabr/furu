# 05 — IN_PROGRESS reconcile + stale timeout (and missing timestamp compatibility)

## Reported issues (high-priority + major)
- If any dependency is IN_PROGRESS, schedulers just sleep (local + pool); no reconcile/timeout => dead worker can hang run forever.
- Planner stale detection may treat missing heartbeat/started_at as stale immediately (age=None ⇒ stale).
  This breaks legacy IN_PROGRESS states without timestamps after upgrade.

## Desired behavior
- Before sleeping due to IN_PROGRESS:
  - run reconciliation (`StateManager.reconcile(...)` or equivalent)
  - if still blocked, enforce a stale timeout (configurable; ideally uses `FURU_CONFIG.stale_timeout`)
- Missing timestamps must not be interpreted as immediately stale:
- If heartbeat/started_at missing, treat as "unknown staleness"
  - Prefer fallback to stable fields if present (e.g., state.updated_at) or skip stale eviction for one cycle and warn.

## Implementation steps
- [x] Add a shared helper (plan/util module) used by local and pool:
  - `reconcile_or_timeout_in_progress(plan, stale_timeout_sec)`
- [x] Update `reconcile_in_progress` logic in `src/furu/execution/plan.py`:
  - do not treat missing timestamps as stale
  - fall back to a safer field if available, or skip stale detection with a warning
- [x] Update schedulers:
  - `src/furu/execution/local.py` and `src/furu/execution/slurm_pool.py`
  - call reconcile helper before sleeping
  - raise after timeout with clear message listing stuck nodes

## Acceptance criteria
- No indefinite sleep on IN_PROGRESS.
- Legacy IN_PROGRESS states without timestamps are not immediately preempted on upgrade.

## Tests
- [x] Create IN_PROGRESS with missing timestamps and assert it is not immediately considered stale.
- [x] Create IN_PROGRESS with old heartbeat and assert timeout triggers reconcile/eviction behavior.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
- Added reconcile helper used by local + pool and updated stale detection to fallback to updated_at/missing timestamps.
- Added plan tests for missing timestamps and stale preemption.
