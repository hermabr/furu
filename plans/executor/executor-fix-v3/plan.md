# Executor Fix v3 — Ralph Loop Plan

This folder is a **Ralph Wiggum loop work queue** to fix remaining correctness and reliability issues in **executor v1** (PR 23 and follow-up feedback).

Executor v1 adds:
- strict `Furu.get()` semantics controlled by an execution context
- dependency planning (`build_plan`, `ready_todo`, `reconcile_in_progress`)
- local parallel executor
- Slurm executors:
  - per-node DAG (`afterok` dependencies)
  - worker pool (filesystem queue)
- submitit integration + log pathing
- docs/tests

The architecture is solid, but multiple reports identify **blocking correctness** issues and a few **behavioral contradictions** (notably around retry behavior in pool mode).

This plan is designed so an agent can complete fixes incrementally, update checklists, and keep tests green.

## Non-negotiables
1) **Interactive mode**: `obj.get()` computes missing artifacts recursively (existing behavior).
2) **Executor mode**: `obj.get()` is **load-only** unless `force=True`.
3) **Executor `force=True`**: allowed only when `ctx.spec_key == obj._executor_spec_key()` (exact match v1).
4) **Default spec key**: `"default"`.
5) **Executors must not allow dependents to run when dependencies are failed or missing.**
6) **No indefinite waits**: any waiting on `IN_PROGRESS` must reconcile and/or timeout.
7) **Slurm DAG afterok** only wires dependencies to **submitit-backed** job IDs; otherwise fail fast with clear message.

## Key theme in v3
We must resolve a tension:

- Some designs **fail fast on pool failures** (queue/failed => abort).
- But `FURU_CONFIG.retry_failed=True` implies executors should retry failed computations.

v3 introduces explicit differentiation:
- **Compute failure** (node `_create()` raised, state failed): retryable when `retry_failed=True` (with a bounded retry policy).
- **Queue/protocol failure** (spec mismatch, missing payload, invalid JSON, internal worker bug): fatal; abort immediately.

## Update protocol (required)
As you implement changes:
- Update the relevant `NN-*.md` file:
  - mark checkboxes complete
  - add brief notes to its progress log
- Also update `00-index.md` (master checklist).
- Keep diffs small and focused. Prefer completing one subplan per commit.

## Start here
Open `plans/executor-fix-v3/00-index.md` and follow the implementation order.
