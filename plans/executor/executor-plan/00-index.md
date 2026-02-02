# plans/executor-plan/00-index.md — v1 index, tracking, and implementation order

> This file is the **single tracker** for executor v1 implementation.
> Each feature has a dedicated subplan with its own checklist + progress log.

## Quick navigation

- 01-core-get.md — remove `load_or_create`, implement `Furu.get()`
- 02-exec-context.md — executor context via `contextvars`
- 03-specs-and-slurmspec.md — spec keys `"default"` and executor-provided mapping rules
- 04-submitit-paths.md — `<FURU_PATH>/submitit` default + override
- 05-submitit-jobid-race.md — fix watcher race (queued→running attempt id)
- 06-planner.md — plan builder (DONE/IN_PROGRESS/TODO/FAILED) + topo/ready helpers
- 07-local-executor.md — `run_local`
- 08-slurm-dag.md — `submit_slurm_dag` per-node afterok
- 09-slurm-pool.md — `run_slurm_pool` filesystem queue
- 10-logging.md — rename console prefix to `"get"`
- 11-repo-migration.md — update docs/examples/tests
- 12-tests.md — testing strategy and concrete tests to add (not required to start implementation)

## Recommended implementation order (keeps diff manageable)

1. **Core API switch**: remove `load_or_create`, implement `get` (01)
2. **Execution context**: `contextvars` + strict mode plumbing (02)
3. **Spec keys**: `_executor_spec_key()` default `"default"`, `SlurmSpec`, mapping rules (03)
4. **Logging + thread safety**: `"get"` prefix and signal handler guard (10 + part of 01)
5. **Submitit root path**: config + helper paths (04)
6. **Submitit job_id race fix** (05)
7. **Planner**: DAG discovery + statuses + topo/ready (06)
8. **Local executor** (07)
9. **Slurm DAG executor** (08)
10. **Slurm worker pool** (09)
11. **Repo migration + cleanup** (11)
12. **Tests** (12)

## Global milestone checklist

- [x] M0: Delete `load_or_create` and update all call sites to `.get()`
- [x] M1: Implement executor context and strict `.get()` semantics
- [x] M2: Implement spec keys with default `"default"` and executor-provided mapping
- [x] M3: Submitit logs under `<FURU_PATH>/submitit` with override
- [x] M4: Fix Submitit job-id watcher race
- [x] M5: Dependency planner with statuses (DONE/IN_PROGRESS/TODO/FAILED)
- [x] M6: Local parallel executor
- [x] M7: Slurm per-node DAG submission via `afterok`
- [x] M8: Slurm worker-pool filesystem queue
- [x] M9: Docs/examples/tests updated and green

## Progress Log (append-only)

| Date | Area | Summary |
|---|---|---|
| 2026-01-23 | — | (start) |
| 2026-01-23 | submitit | Attach submitit job_id to current queued/running attempt without id match. |
| 2026-01-22 | submitit paths | Add submitit root config and path helpers. |
| 2026-01-22 | core/exec/logging | Switch to `get()`, add executor context, update logging prefix. |
| 2026-01-22 | executor | Add `SlurmSpec` and spec resolution helper for executor mappings. |
| 2026-01-22 | planner | Implement dependency planner build/topo/ready helpers. |
| 2026-01-22 | executor | Implement local executor scheduler and tests. |
| 2026-01-22 | executor | Implement Slurm DAG submission and submitit factory. |
| 2026-01-22 | executor | Implement Slurm worker pool queue and controller. |
| 2026-01-22 | docs/tests | Update README executor docs and add executor test coverage. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | (initial) |
