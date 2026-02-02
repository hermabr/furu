# 00-index — Executor Fix v2 master checklist

## Objective
Fix blocking correctness issues reported in PR 23 and follow-up review:
- retry semantics (`retry_failed`)
- pool task completion correctness
- IN_PROGRESS hangs (reconcile/timeout)
- Slurm DAG backend/job_id wiring robustness
- path annotation import error
- pool stale-running requeue safety (avoid duplicate execution / worker crashes)
- additional robustness: malformed payload handling, merging slurm_additional_parameters, log noise reduction
- add targeted tests for new behaviors

## Implementation order (do in sequence)
1) **Paths import fix** → `05-paths-import-fix.md`
2) **Retry failed semantics in planner + executors** → `01-retry-failed-semantics.md`
3) **Pool: do not mark done if state is failed / missing** → `02-pool-task-completion-correctness.md`
4) **IN_PROGRESS hangs: reconcile + stale timeout** → `03-in-progress-reconcile-timeout.md`
5) **Slurm DAG: fail fast on non-submitit IN_PROGRESS + job_id robustness + merge params + expose run_id** → `04-slurm-dag-robustness.md`
6) **Pool: stale running requeue redesign (avoid mtime-only duplicates) + tolerant mark_done/mark_failed** → `06-pool-stale-running-requeue.md`
7) **Core: executor-mode get() reduce spam; validate-failed-success recoverability** → `07-core-executor-polish.md`
8) **Docs & tests** → `08-tests-and-docs.md`

## Master checklist
- [x] 05 Paths: fix `Path` annotation NameError
- [x] 01 Planner/executors honor `retry_failed`
- [x] 02 Pool task completion correctness (don’t mark done on failed state)
- [x] 03 IN_PROGRESS reconcile/timeout (no indefinite sleep)
- [x] 04 Slurm DAG robustness: backend check + job_id loop safety + merge additional params + return run_id
- [x] 06 Pool stale-running: safe requeue (no mtime-only duplicates) + tolerant mark_done/mark_failed
- [x] 07 Core polish: `_exists_quiet` in executor get + validate-failed-success repair behavior
- [x] 08 Add/update tests + docs notes

## Definition of done
All items above checked, `make test` and `make lint` green, branch pushed, PR exists.

## Progress log (append-only)
| Date | Item | Summary |
|---|---|---|
| 2026-01-23 | 05 | Verified Path import in execution paths and added import test. |
| 2026-01-23 | 01 | Confirmed retry_failed planner/executor behavior with local + pool tests. |
| 2026-01-23 | 02 | Pool worker now verifies state before marking tasks done. |
| 2026-01-23 | 02 | Added pool worker test for failed-state tasks (no done marker). |
| 2026-01-23 | 03 | Reconciled stale IN_PROGRESS dependencies with timeout and tests. |
| 2026-01-23 | 04 | Slurm DAG now errors on non-submitit in-progress dependencies. |
| 2026-01-23 | 04 | Slurm DAG job_id wait updates state and reports last seen job_id. |
| 2026-01-23 | 04 | Merge dependency with slurm_additional_parameters. |
| 2026-01-23 | 04 | Expose Slurm DAG run_id on submission result. |
| 2026-01-23 | 04 | Added slurm DAG robustness tests for backend mismatch, job_id switch, and parameter merge. |
| 2026-01-23 | 06 | Added pool heartbeats for stale requeue and tolerant task marking with tests. |
| 2026-01-23 | 07 | Executor get uses quiet exists; invalidate cached success on validate failure; worker-entry failure behavior + tests. |
| 2026-01-23 | 08 | Confirmed coverage for executor/pool/dag tests and README notes for retry/always_rerun. |
