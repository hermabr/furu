# plans/executor-fix-v6/plan.md
# Furu Fix v6 — Ralph Wiggum Loop Plan (Executor follow-ups)

This folder is a **Ralph Wiggum loop work queue** for post–executor-fix v5 follow-ups.

Scope is limited to the remaining correctness/robustness issues identified in the updated PR review:

## Executor follow-ups (v6)
1) **Retry count semantics**: `FURU_MAX_COMPUTE_RETRIES` / `FuruConfig.max_compute_retries` is documented as “max retries” but code behaves like “max attempts” (off-by-one).
2) **Pool missing-heartbeat grace**: missing-heartbeat uses task file mtime; after rename, mtime can be old → immediate requeue if todo sat for a while.
3) **Pool stale-heartbeat requeues unbounded**: stale heartbeat requeue does not consume attempts / does not bound retries → can churn forever.
4) **Slurm DAG job_id wait**: `_wait_for_job_id` can time out if attempt flips to terminal before job_id is written into state (even if submitit knows job_id).
5) **Validate exception contract**: `_exists_quiet()` re-raises unexpected `_validate()` exceptions; executor planning loops can crash for legacy validates (breaking behavior). Decide + document.

---

## Non-negotiables (executor behavior)
1) Interactive mode: `obj.get()` computes missing artifacts recursively (existing behavior).
2) Executor mode: `obj.get()` is load-only unless `force=True`.
3) `force=True` is allowed only when `ctx.spec_key == obj._executor_spec_key()` (v1 exact match).
4) Pool and DAG must never treat an invalid/failed artifact as “done”.
5) Pool controller must never spin forever due to heartbeat/requeue logic.

## Guiding principles for these fixes
- Prefer **bounded behavior** over “infinite requeue”.
- Preserve **observability**: errors should be explicit about why a run is aborting.
- Keep diffs small and keep `make lint` and `make test` green.

---

## Update protocol (required)
Each iteration:
- Implement the next smallest unfinished checklist item (see `00-index.md`).
- Update:
  - the relevant subplan `NN-*.md` progress log
  - `00-index.md` master checklist
- Run tests: `make lint` and `make test` (plus any subplan extra commands).
- Commit + push.
