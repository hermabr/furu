# 00 — Index and execution order (Executor v1.1 hardening)

This plan is organized into **phases**. Complete them in order.

## Global policy decisions (must match implementation)

- Pool failure policy: **fail fast** if `queue/failed` contains entries (v1.1).
- Failed retry policy: honor `FURU_CONFIG.retry_failed` in executors.
- Slurm DAG policy: `afterok` wiring requires **submitit-backed job ids** for IN_PROGRESS deps; otherwise fail fast.

## Phase order

1) Phase 01 — Pool fail-fast on failed queue entries (critical infinite-spin fix)  
   File: `plans/executor-plan-fix/01-pool-failed-queue.md`

2) Phase 02 — Pool recovery: requeue stale `queue/running` tasks  
   File: `plans/executor-plan-fix/02-pool-stale-running.md`

3) Phase 03 — Honor `retry_failed` in planning + executors  
   File: `plans/executor-plan-fix/03-retry-failed.md`

4) Phase 04 — Slurm DAG robustness (backend checks + job_id update hardening + run_id)  
   File: `plans/executor-plan-fix/04-slurm-dag-robust.md`

5) Phase 05 — SlurmSpec.extra supports nested submitit configs  
   File: `plans/executor-plan-fix/05-slurmspec-extra.md`

6) Phase 06 — Docs: breaking changes + always_rerun semantics  
   File: `plans/executor-plan-fix/06-docs.md`

7) Phase 07 — Reduce planner polling overhead/log noise  
   File: `plans/executor-plan-fix/07-plan-performance.md`

8) Phase 08 — Verification: add/adjust tests for new behavior  
   File: `plans/executor-plan-fix/08-tests.md`

## Global acceptance criteria

- Pool never spins forever due to `queue/failed` entries or stranded `queue/running`.
- Executors honor `FURU_CONFIG.retry_failed`.
- Slurm DAG fails fast on non-submitit IN_PROGRESS deps and records job ids robustly.
- Tests cover regressions and pass: `make test` and `make lint`.

## Progress log (global)

| Date | Phase | Summary |
|---|---|---|
|  |  |  |
