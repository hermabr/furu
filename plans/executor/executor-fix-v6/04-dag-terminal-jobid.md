# plans/executor-fix-v6/04-dag-terminal-jobid.md
# 04 — Slurm DAG: avoid job_id timeout when attempt becomes terminal

## Problem
`_wait_for_job_id` currently only writes job_id into state for queued/running attempts.
If an attempt flips to terminal (success/failed) before job_id is written, `_wait_for_job_id`
can **time out**, even though submitit can still provide the job_id.

## Decision
`_wait_for_job_id` must not time out in this case.

Two acceptable solutions (implement at least one):
A) Allow `_set_submitit_job_id` to update job_id for terminal attempts if missing (safe + improves observability).
B) If submitit reports a job_id, `_wait_for_job_id` may return it even if state couldn’t be updated.

## Implementation steps
- [x] In `src/furu/execution/slurm_dag.py`:
  - Relax `_set_submitit_job_id` to also allow terminal statuses (at least when job_id is missing).
  - Update `_wait_for_job_id` so that if submitit yields job_id:
    - it returns job_id without requiring state round-trip success (especially if state is terminal)
  - Add a defensive check: if the node becomes DONE-success while waiting, returning job_id is still OK
    (Slurm dependency on an already-successful job is satisfied immediately).
- [x] Ensure behavior is still correct for the job_id race case (queued→running attempt swap).

## Acceptance criteria
- `_wait_for_job_id` does not time out solely because the attempt became terminal before job_id write.
- DAG wiring continues and submits dependents successfully.

## Tests
- [x] Add a unit test that simulates:
  - terminal submitit attempt in state without job_id
  - adapter/job provides job_id
  - `_wait_for_job_id` returns job_id
- [x] `make lint` and `make test`

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes: Allow terminal job_id updates and return adapter job_id when terminal.
