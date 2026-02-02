# 04 — Slurm DAG robustness: backend checks, job_id update, run_id surfacing

## Why

Problems to fix:
1) `_job_id_for_in_progress()` assumes submitit backend. If dep is IN_PROGRESS locally or other backend,
   current flow times out after 15s with generic error.
2) `_wait_for_job_id()` tries to update state via a specific attempt_id and returns without verifying update,
   which can miss queued→running attempt switches (similar to the watcher race).
3) (Nice-to-have) surface a `run_id` in `SlurmDagSubmission` to make log discovery easy.

Relevant files mentioned:
- `src/furu/execution/slurm_dag.py:78`, `:121` (backend assumption)
- `src/furu/execution/slurm_dag.py:49`, `:61` (job_id update concern)
- `src/furu/execution/slurm_dag.py:86` (run_id suggestion)

## Tasks

### A) Fail fast for non-submitit IN_PROGRESS deps
- [x] In `_job_id_for_in_progress()`:
  - If attempt backend != "submitit", raise a clear error immediately:
    - explain that `afterok` wiring requires submitit job ids
    - suggest waiting, resubmitting, or using pool mode

### B) Make `_wait_for_job_id()` robust to attempt-id switches
- [x] Mirror the watcher behavior:
  - update the *current* queued/running submitit attempt regardless of attempt_id
  - after attempting update, re-read state and verify job_id is present
  - only return when job_id is actually present

### C) Surface run_id
- [x] Generate a `run_id` per DAG submission.
- [x] Include it in return object.
- [x] (Optional but recommended) include `run_id` in submitit log folder path:
  - `<submitit_root>/nodes/<spec_key>/<run_id>/...`

## Acceptance criteria

- Non-submitit IN_PROGRESS deps fail immediately with a helpful error.
- job_id resolution is reliable even if attempt id changes during queued→running transition.
- Users can locate DAG logs via returned run_id (if implemented).

## Progress log

| Date | Summary |
|---|---|
| 2026-01-22 | Fail fast in `_job_id_for_in_progress()` for non-submitit attempts with a clear error message. |
| 2026-01-22 | Update `_wait_for_job_id()` to attach job IDs to the active queued/running submitit attempt and verify before returning. |
| 2026-01-22 | Return Slurm DAG run IDs in `SlurmDagSubmission` and wire them into submitit log paths. |

## Plan changes

| Date | Change | Why |
|---|---|---|
|  |  |  |
