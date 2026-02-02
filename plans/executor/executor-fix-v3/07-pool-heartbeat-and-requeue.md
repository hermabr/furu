# 07 — Pool heartbeat and requeue semantics (missing heartbeat behavior)

## Reported major issue
- Requeue running tasks based on heartbeat mtime, but fallback to task mtime when heartbeat missing.
- If heartbeat creation fails or worker crashes early, long jobs can be requeued prematurely.

## Desired behavior
- Missing heartbeat should not cause immediate or premature requeue based solely on task mtime.
- Define one of these policies (choose and implement):
  A) Heartbeat is mandatory: if missing after a short grace period, treat as protocol failure and abort.
  B) Missing heartbeat is "unknown": skip requeue for one tick (or until a longer timeout), and rely on IN_PROGRESS reconcile/timeout instead.
  C) Use worker liveness: require presence of worker dir + recent worker poll timestamp; only requeue if worker seems dead.

Recommended for v3: **Policy B** (simple, safe):
- For newly claimed tasks, allow a grace period for heartbeat creation.
- If heartbeat missing beyond grace period, mark as protocol failure (or treat as stale only if no worker exists).

## Implementation steps
- [x] Implement `heartbeat_grace_sec` (e.g., 30s).
- [x] In stale requeue logic:
  - if heartbeat exists: use it
  - if heartbeat missing:
    - if task was claimed recently: skip
    - else: treat as protocol failure OR skip and rely on other mechanisms (pick one)
- [x] Ensure `_mark_done/_mark_failed` are robust even if heartbeat is missing.

## Tests
- [x] Missing heartbeat does not cause premature requeue within grace period.
- [x] Missing heartbeat beyond grace period triggers the intended behavior (abort or safe requeue), deterministically.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
- Added heartbeat grace handling in stale requeue logic with protocol failure on missing heartbeat past grace.
- Added tests for missing heartbeat grace and protocol failure behavior.
