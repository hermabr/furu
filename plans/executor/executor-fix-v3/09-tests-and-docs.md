# 09 — Tests and docs (v3)

## Required tests (add if missing)
- [x] retry_failed semantics:
  - local: fails once then recomputes with retry_failed=True
  - pool: compute failure retried when retry_failed=True (bounded retries)
- [x] pool failed queue handling:
  - protocol failure in queue/failed aborts immediately
  - compute failure in queue/failed is retried (when allowed) then succeeds or exhausts retries
- [x] in-progress reconcile/timeout:
  - stale in-progress triggers reconcile/timeout behavior
  - missing timestamps do not cause immediate stale eviction
- [x] pool missing heartbeat behavior:
  - missing heartbeat within grace does not trigger requeue
  - missing heartbeat beyond grace triggers the chosen policy deterministically
- [x] slurm dag:
  - non-submitit in-progress dep fails fast
  - job_id mismatch update
  - param merge

## Docs updates
- [x] Document executor retry semantics (including difference between compute failure and protocol failure in pool mode).
- [x] Document always_rerun semantics in executors (once per run) if intentional.

## Acceptance criteria
- `make test` and `make lint` passes.
- Tests are deterministic (avoid real sleeps; mock time where possible).
- Docs reflect the actual behavior implemented.

## Progress log
- Status: [ ] not started / [ ] in progress / [x] done
- Notes:
- Verified coverage across local/pool/slurm dag tests including heartbeat grace behavior.
- README documents retry_failed and always_rerun executor semantics.
