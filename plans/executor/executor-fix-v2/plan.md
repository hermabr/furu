# Executor Fix v2 — Ralph Loop Plan

This folder is a **Ralph Wiggum loop work queue** to fix **blocking correctness issues** in PR 23 (executor v1):
- strict `get()` + execution context
- planner
- local executor
- Slurm DAG (afterok) + Slurm worker-pool
- submitit integration + log pathing
- tests/docs

The core architecture is good, but several issues can cause **silent incorrect execution**, **infinite waits**, or **incorrect task completion** in pool/DAG modes.

## How to use this plan
- The **single source of truth** is `plans/executor-fix-v2/00-index.md`.
- Each fix area has its own file `plans/executor-fix-v2/NN-*.md` with:
  - scope & rationale
  - concrete implementation steps
  - acceptance criteria
  - test requirements
  - progress log fields

## Non-negotiables (must preserve)
1) **Interactive mode**: `obj.get()` computes missing artifacts recursively (existing behavior).
2) **Executor mode**: `obj.get()` is **load-only** unless `force=True`, and `force=True` is allowed only when:
   - `ctx.spec_key == obj._executor_spec_key()` (exact match for v1)
3) **Spec keys**: default spec key is `"default"` (not `"cpu"`).
4) **Executors must honor `FURU_CONFIG.retry_failed`** (either retry failures when enabled, or explicitly document executor-only divergence—this plan assumes we honor it).
5) **Slurm DAG dependencies** (`afterok`) must only be wired to submitit-backed job IDs; if a dependency is IN_PROGRESS with a different backend, fail fast with a clear error.
6) **Slurm pool must not silently mark tasks as done** when underlying Furu state is failed or missing.
7) **No infinite waits**: any wait loop involving IN_PROGRESS dependencies must have reconciliation and/or stale timeout behavior.

## Update protocol (required)
When implementing fixes:
- Update the relevant `NN-*.md` file:
  - mark checkboxes as complete
  - add brief notes to the progress log in that file
- Also update `00-index.md` (master checklist) as items complete.
- Keep changes small: implement one subplan per PR-sized chunk when possible.

## Start here
Open `plans/executor-fix-v2/00-index.md` and follow the implementation order.
