# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Local Executor MVP

Furu includes an in-memory local executor MVP exposed as `furu.execute_local()`
and `furu.InMemoryScheduler`. It splits execution into:

- a dependency planner that starts from one or more final `Furu` artifacts,
  recursively walks declared references from dataclass fields and
  `@furu.dependency`, deduplicates artifacts by `object_id`, and records each
  `parent -> dependency` edge on the parent job's `dependencies` set;
- a small in-memory scheduler with `queued`, `running`, `completed`, and
  `failed` states;
- a worker runner that claims ready jobs and executes claimed artifacts directly.

Queued jobs do not have a separate blocked state. A queued job is claimable only
when every dependency ID is either completed in the scheduler or already present
on disk as a cached Furu result.

Inside a worker-owned artifact, `load_or_create()` has executor semantics:
cached dependencies still load normally, but missing dependencies raise
`BlockedOnDependencies(deps)` before dependency locks are acquired or recursive
inline computation starts. The worker treats this as control flow, registers the
missing dependencies, moves the current job back to `queued`, and retries it
after those dependencies are complete.

Suspended jobs are not resumed from the middle of `create()`. They are aborted
cleanly and rerun from the beginning, so `create()` implementations should be
idempotent or safe to retry. `try_load()` remains load-only and never schedules
work. Calling `load_or_create()` inside `create()` means the dependency is
required and should be scheduled if it is missing.

Cycle detection is intentionally omitted in this MVP. To avoid infinite
rediscovery loops, each scheduler has `max_suspensions_per_job`; a job that
exceeds the limit is marked failed with a clear error.
