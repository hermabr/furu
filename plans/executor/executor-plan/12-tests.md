# plans/executor-plan/12-tests.md — Test plan (what to add under `tests/`)

> This file describes the test work to add for executor v1.
> It is included here for completeness, but implementation can start without it.

## Test folder structure to add

```
tests/
  execution/
    __init__.py
    helpers.py
    test_get_executor_semantics.py
    test_plan.py
    test_local_executor.py
    test_slurm_dag.py
    test_slurm_pool.py
    test_submitit_jobid_race.py
```

## Core semantic tests

Validate:
- interactive `.get()` computes recursively
- executor context `.get()` is load-only and missing raises
- executor context `.get(force=True)` computes only on exact spec match
- parent computing must not compute missing dependency

## Planner tests

Validate:
- DONE nodes prune traversal
- IN_PROGRESS nodes prune traversal
- FAILED nodes prune traversal
- topo order correctness

## Local executor tests

Validate:
- computes DAG
- parallelizes independent nodes (loose timing assertion)
- fails on undeclared dependency (because `.get()` inside worker is strict)
- respects spec mismatch on `force=True`

## Slurm DAG submission tests (no real Slurm)

Use fake submitit executor factory / monkeypatch to assert:
- topo submission order
- correct `afterok` dependency strings
- correct submitit folder root under `<submitit_root>/nodes/<spec_key>/...`
- requires `specs["default"]`

## Slurm pool tests (no real Slurm)

Use a thread-based worker launcher:
- controller enqueues tasks per spec key
- workers claim tasks atomically
- cross-spec dependency works (default dep then gpu dependent)
- respects global worker cap
- respects window policy (`dfs`/`bfs`/k)

## Submitit job_id watcher regression test

Validate watcher sets job_id even when attempt id changes between queued and running.

## Checklist

- [x] Add `tests/execution/helpers.py` with minimal Furu nodes
- [x] Add `get()` semantic tests
- [x] Add planner tests
- [x] Add local executor tests
- [x] Add slurm_dag submission tests with fake submitit
- [x] Add slurm_pool tests with thread workers + filesystem queue
- [x] Add watcher race regression test

## Progress Log (append-only)

| Date | Summary |
|---|---|
| 2026-01-23 | (start) |
| 2026-01-22 | Add executor context, planner, local, slurm dag, slurm pool, and job id race tests. |

## Plan Changes (append-only)

| Date | Change | Why |
|---|---|---|
| 2026-01-23 | — | initial |
| 2026-01-22 | Added tests in top-level `tests/` instead of a `tests/execution/` subtree. | Keep new tests adjacent to existing suite structure. |
