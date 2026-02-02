# plans/executor-plan/05-submitit-jobid-race.md — Fix Submitit job_id recording race

## Scope

Fix a race that breaks Slurm DAG wiring and resuming:

- `_submit_once(...)` creates a queued attempt with an attempt id.
- When the worker starts computing under `compute_lock`, the state may transition to a running attempt with a *new* attempt id.
- If watcher logic requires attempt-id equality, it can fail to attach `job_id`.

This breaks:
- `afterok` dependency wiring
- resuming submissions
- observability/debugging

## Required v1 fix

In `src/furu/adapters/submitit.py`, update `watch_job_id` mutation logic:

- Write the job id to the **current** attempt if:
  - attempt exists
  - attempt backend is `"submitit"`
  - attempt status is `"queued"` or `"running"`
  - attempt.scheduler lacks `"job_id"`

**Do not require attempt id equality**.

### Code snippet (v1)

```py
def mutate(state: _FuruState) -> None:
    attempt = state.attempt
    if attempt is None:
        return
    if attempt.backend != "submitit":
        return
    if attempt.status not in {"queued", "running"}:
        return
    if attempt.scheduler.get("job_id"):
        return
    attempt.scheduler["job_id"] = job_id
```

## Checklist

- [x] Implement the watcher mutation change above
- [x] Ensure no other code assumes attempt id equality for job_id attachment
- [x] Add/adjust tests later to cover the queued→running id change scenario

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-23 | Update submitit job-id watcher to attach to current queued/running attempt. |
| 2026-01-22 | Add job id race regression coverage. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
