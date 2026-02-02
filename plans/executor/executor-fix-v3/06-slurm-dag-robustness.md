# 06 — Slurm DAG robustness (backend checks, job_id updates, param merge, run_id)

## Reported issues
- `_job_id_for_in_progress` waits for job IDs even if backend isn't submitit → timeouts and generic errors.
- `_set_submitit_job_id` only writes when missing; if stale job_id exists it won't update → can wire afterok to wrong job.
- Merge `slurm_additional_parameters` rather than overwrite.
- Expose run_id in return type for log discovery.

## Desired behavior
- If dependency is IN_PROGRESS and backend != submitit:
  - fail fast with clear actionable message (use pool mode or wait for completion)
- When job_id differs from current attempt’s job id:
  - update state to match the active queued/running attempt (not only when missing)
- Merge additional params.
- Return run_id.

## Implementation steps
- [x] Update `_job_id_for_in_progress` backend check.
- [x] Update `_set_submitit_job_id(directory, job_id)`:
  - if current attempt is submitit queued/running:
    - if scheduler.job_id missing OR differs: set to provided job_id
- [x] Ensure `_wait_for_job_id` re-reads state and confirms correctness.
- [x] Merge additional parameters with dependency.
- [x] Ensure `SlurmDagSubmission` includes `run_id` and possibly log_dir.

## Tests
- [x] Non-submitit in-progress dep fails fast.
- [x] Job-id mismatch update test (state has old job id; ensure updated).
- [x] Param merge test.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
- Updated submitit job_id tracking to overwrite stale IDs and re-check state after updates.
- Verified non-submitit in-progress behavior, parameter merge, and run_id return.
