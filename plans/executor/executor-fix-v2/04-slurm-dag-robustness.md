# 04 — Slurm DAG robustness (backend checks, job_id loop, merge params, return run_id)

## Problems (blocking + recommended)
1) Backend mismatch:
- `_job_id_for_in_progress()` waits for submitit job IDs even if attempt backend isn't submitit (`src/furu/execution/slurm_dag.py:~78`).
- This times out with generic error and blocks submissions.

2) Job-id update robustness:
- `_wait_for_job_id()` / state update loop should handle queued→running attempt switch (mirroring watcher behavior).
- Need clearer failure messages and verification the update stuck.

3) Merge slurm_additional_parameters:
- Adding `dependency=afterok:...` should not overwrite user-specified `slurm_additional_parameters` (from SlurmSpec.extra).

4) Expose run_id:
- run id is generated for log folders but not exposed in `SlurmDagSubmission`; returning it helps log discovery.

## Desired behavior
- Fail fast when IN_PROGRESS dep is not submitit backend, with a clear actionable error.
- Job-id retrieval should be robust, updating current queued/running attempt regardless of attempt_id and re-reading to confirm.
- `slurm_additional_parameters` merged with dependency string.
- Return `run_id` in submission result.

## Implementation steps
- [x] Update `_job_id_for_in_progress()`:
  - if attempt.backend != "submitit": raise `FuruExecutionError` with message:
    - "Cannot wire afterok dependencies to non-submitit in-progress nodes. Use pool mode or wait until completed."
- [x] Update `_wait_for_job_id()`:
  - factor a helper that writes job_id onto current queued/running submitit attempt (no attempt_id match)
  - after setting, re-read state and confirm presence before returning
- [x] Merge additional parameters:
  - if spec.extra has slurm_additional_parameters dict, merge dependency into it rather than overwrite
- [x] Extend `SlurmDagSubmission` to include `run_id` and (optionally) submitit_root/log_dir path.

## Acceptance criteria
- Non-submitit IN_PROGRESS dep fails fast without a 15s timeout.
- Job-id state is reliably recorded even across queued→running switch.
- User additional slurm parameters are preserved.
- Users can locate logs via run_id.

## Tests
- [x] Add test: IN_PROGRESS with backend != submitit => immediate error.
- [x] Add test: `_wait_for_job_id` keeps polling until state contains job_id (simulate attempt switch).
- [x] Add test: merging `slurm_additional_parameters` retains original keys.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
  - 2026-01-23: raise FuruExecutionError on non-submitit in-progress deps.
  - 2026-01-23: update _wait_for_job_id to retry job_id write and report last seen id.
  - 2026-01-23: merge dependency into slurm_additional_parameters without overwriting.
  - 2026-01-23: return run_id on SlurmDagSubmission for log discovery.
  - 2026-01-23: added slurm DAG robustness tests for backend mismatch, job_id switch, and parameter merge.
