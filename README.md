# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Local Executor MVP

`furu.run_local(final_artifacts)` runs final `Furu` artifacts through an in-memory
planner and scheduler backed by a local `ThreadPoolExecutor`. The planner walks
declared dependencies from artifact fields and `@furu.dependency`, deduplicates
artifacts by `object_id`, and schedules graph edges as `parent -> dependency`.

The scheduler keeps jobs in `queued`, `running`, `completed`, or `failed` states.
Queued jobs carry `dependencies: set[str]` and become runnable only after every
dependency ID in that set has completed. Cycle detection is intentionally not
part of this MVP.

Inside executor-managed `create()` calls, `load_or_create()` means "required
dependency": it returns already cached results, but if any requested artifact is
missing it raises `BlockedOnDependencies` before acquiring dependency locks or
computing that dependency inline. The worker treats that as a normal scheduling
event, inserts the missing dependency subgraph, and requeues the current job.
Suspended jobs rerun from the beginning, so `create()` implementations should be
idempotent. Excessive repeated suspensions fail clearly. `try_load()` never
schedules work; use it only for optional cached inputs.
