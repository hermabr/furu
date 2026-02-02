# Furu Fix v5 — Ralph Wiggum Loop Plan (Executor + Dashboard + Release)

This folder is a **Ralph Wiggum loop work queue** for remaining post-PR fixes across:

## Executor v1 follow-ups
- Slurm pool: missing heartbeat policy → **requeue at least once** before treating as protocol failure
- (If still present) any remaining failure taxonomy/diagnostic issues

## Dashboard / packaging / release follow-ups
- **Public `furu_hash` property** (no README/test references to `_furu_hash` as the only accessor)
- **Bundle dashboard frontend dist assets into the wheel** so `furu-dashboard serve` works after install
- Fix `StateManager.update_state` lock-release bug that could delete another process’s lock
- Improve release workflow so it runs build/tests
- Align dashboard version with installed package version (use `importlib.metadata.version`)
- (Security) Prevent SPA route handler path traversal
- Update changelog for user-visible changes

This plan is structured so an agent can implement fixes incrementally with small commits, keeping tests green.

---

## Non-negotiables (executor)
1) Interactive mode: `obj.get()` computes missing artifacts recursively (existing behavior).
2) Executor mode: `obj.get()` is load-only unless `force=True`.
3) `force=True` allowed only when `ctx.spec_key == obj._executor_spec_key()` (exact match v1).
4) Default spec key is `"default"`.
5) Pool and DAG must never mark an invalid or failed artifact as "done".

## Non-negotiables (dashboard/release)
1) `Furu.furu_hash` is a stable, public property returning the hash string.
2) Dashboard assets **ship in the wheel** and can be served without source checkout.
3) Locking is correct: failing to acquire a lock must never unlink another process’s lock.
4) Changelog reflects user-visible changes; release workflow runs tests/build.

---

## Update protocol (required)
- Each iteration: implement the next smallest unfinished checklist item.
- Update:
  - the relevant subplan `NN-*.md` progress log
  - `00-index.md` master checklist
- Run tests: `make lint` and `make test` (and any additional recommended commands in subplans).
- Commit + push from the model; no hooks.

Start with `00-index.md`.
