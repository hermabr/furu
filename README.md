# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Local Distributed Executor MVP

`furu.run_local(final)` runs one or more final artifacts through the MVP local
executor. It uses an in-memory scheduler and a worker loop rather than recursively
computing every cache miss inline.

The planner recursively registers declared artifact references from dataclass
fields and `@furu.dependency` properties. Jobs are deduplicated by `object_id`.
Each parent job stores dependency IDs in `dependencies`; there is no separate
blocked state and no cycle detection in this MVP.

Queued jobs become claimable only when all dependencies are completed by the
scheduler or are already available on disk as cached Furu results. During a
worker-owned `create()`, `load_or_create()` still loads cached dependencies, but
missing dependencies raise `BlockedOnDependencies` instead of acquiring locks or
computing recursively. The worker catches that signal, schedules the missing
dependencies, and retries the current job later.

Suspension does not resume `create()` from the middle. The current attempt is
aborted and the artifact is rerun from the beginning after its dependencies are
ready, so `create()` implementations used with the executor should be safe to
retry. `try_load()` remains load-only and never schedules work. Repeated
suspensions are capped by `max_suspensions_per_job` to keep rediscovery loops
from running forever.
