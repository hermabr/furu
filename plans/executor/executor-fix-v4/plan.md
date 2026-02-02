# Executor Fix v4 — Ralph Loop Plan

This folder is a **Ralph Wiggum loop work queue** for additional fixes/refinements to executor v1 after the main v3 blocking issues. v4 focuses on **behavioral correctness, diagnostics, and retry/error taxonomy consistency**.

This plan incorporates feedback about:
- `_exists_quiet()` swallowing validation exceptions too broadly (risk of silent recompute loops masking bugs)
- pool worker error classification (compute vs protocol failures) and retry behavior
- clearing stale error metadata when requeueing failed tasks
- `_worker_entry()` raising generic `RuntimeError` on lock refusal / failed states (poor operator diagnostics)
- parity/documentation differences between local executor and pool executor retry behavior
- adding tests to cover the above (unit + integration)

## Non-negotiables
1) Interactive mode: `obj.get()` computes recursively as before.
2) Executor mode: `obj.get()` is load-only unless `force=True`.
3) `force=True` allowed only when `ctx.spec_key == obj._executor_spec_key()` (exact match v1).
4) Default spec key is `"default"`.
5) Executors must never mark failed/missing artifacts as successfully completed.
6) Executors must not hang indefinitely on `IN_PROGRESS`.

## Update protocol
- Each iteration: implement the next smallest unfinished checklist item.
- Update both:
  - `00-index.md` master checklist
  - the relevant subplan file(s) progress log
- Run `make test` and `make lint` before committing.
- Commit/push from the model; no hooks.

Start with `00-index.md`.
