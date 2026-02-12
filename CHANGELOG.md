# Changelog

## Unreleased

- Rename migration reference field `FuruRef.directory` to `FuruRef.furu_dir`.
- Replace `_executor_spec_key()` with `_executor() -> SlurmSpec`; `run_slurm_pool()` and `submit_slurm_dag()` now derive Slurm resources directly from each node and no longer accept a `specs` mapping.
- Allow `_executor()` to return multiple `SlurmSpec` choices; `submit_slurm_dag` uses the first choice while `run_slurm_pool` workers can claim tasks from any listed spec.
- Route `run_slurm_pool` worker stdout/stderr to each worker queue directory (`queue/running/<spec>/<worker-id>/stdout.log` and `stderr.log`) while preserving existing `slurm_additional_parameters`.
- Fix `furu.execution` exports to expose `resolve_executor_spec`.
- Add `examples/run_local_executor_benchmark.py` to benchmark local executor planning latency and end-to-end runtime on a nested 10k-node DAG.
- Rework `run_local` to build scheduler state once per run and maintain in-memory buckets (`READY`, `BLOCKED`, `IN_PROGRESS_SELF`, `IN_PROGRESS_EXTERNAL`, `FAILED`, `COMPLETED`) with dependency-pending sets, pre-dispatch state rechecks, and configurable external/underutilized/full reconcile polling (`external_poll_interval_sec`, `underutilized_reconcile_interval_sec`, `plan_refresh_interval_sec`).
- Keep sticky `DONE` snapshot semantics per executor run and separate slurm-pool worker health polling cadence (`worker_health_check_interval_sec`).
- Extend `run_slurm_pool(max_workers_total=...)` to accept `int` or `Mapping[SlurmSpec | Literal["total"], int]`, enforce per-spec limits in worker spawning, derive default total from the sum of per-spec limits when `"total"` is absent, and fail fast when active plan specs are missing from mapping mode limits.
- Add independent slurm-pool queue scan and ready-refresh cadences (`done_scan_interval_sec`, `failed_scan_interval_sec`, `stale_scan_interval_sec`, `ready_refresh_interval_sec`) so controller filesystem scans can be throttled without changing worker polling.

## v0.0.8

- Restore Python 3.11 compatibility by using `TypeVar`/`Generic` for `Furu`.
- Replace the `_dependencies()` hook with `@furu_dep` methods (imported from `furu`) for declaring extra dependencies used by dependency discovery and hashing.
- Add support for plain Python dataclasses in `FuruSerializer` hashing and dict round-trips.

## v0.0.7

- Record `furu_package_version` in experiment metadata.
- Store `schema_key` in metadata and expose schema/staleness in the dashboard API.
- Replace migration APIs with schema-keyed, alias-only migration helpers and alias relationship accessors.

## v0.0.6

- Clarify missing git origin guidance with the option to disable git metadata via `FURU_RECORD_GIT=ignore`.
- Add a pytest-friendly test helper (`furu.testing`) for isolated Furu roots.
- Add `furu.testing.override_results` to stub dependency outputs in tests (by object or furu_hash).
- Add `furu.testing.override_results_for` to stub dependency outputs by field path.

## v0.0.5

- Speed up state/plan checks with SUCCESS markers,
  cached `furu_hash`/`furu_dir`, and fewer writes/mkdirs.
- Switch lock files to visible names and use compute-lock
  mtimes for heartbeats (breaking: drop `heartbeat_at`).
- Route per-artifact logs to `.furu/furu.log` and add
  caller locations for get/dependency logs.
- Update git provenance config (`FURU_RECORD_GIT`,
  `FURU_ALLOW_NO_GIT_ORIGIN`) and `.env` docs.

## v0.0.4

- Add stable `Furu.furu_hash` accessor for artifact identity.
- Package dashboard frontend assets and report package version in dashboard metadata and health responses.
- Introduce executor planning plus a local thread-pool executor for dependency graphs.
- Add submitit/Slurm DAG + pool executors with `SlurmSpec`, `FURU_SUBMITIT_PATH`, and bounded retries via `FURU_MAX_COMPUTE_RETRIES`.
- Tighten executor APIs (`get()` only, config-only `retry_failed`) and improve lock/heartbeat error handling.
- Harden dashboard routing, fix lock-release races, and simplify executor error output.

## v0.0.3

- Add dependency discovery via `_dependencies()` and `_get_dependencies()` with recursive traversal and de-duplication, plus `DependencySpec`/`DependencyChzSpec` typing helpers.
- Include direct dependencies in `Furu` hashing to invalidate caches when implicit dependencies change.
- Record migration events with separate namespace/hash fields instead of composite IDs.
- Default to retry failed artifacts (use `FURU_RETRY_FAILED=0` or `retry_failed=False` to keep failures sticky) while enriching compute errors with recorded tracebacks and hints.
- Add detailed compute lock timeout diagnostics with env var overrides and owner context.
- Surface attempt error messages and tracebacks in the dashboard detail view.
- Wrap metadata/signal handler setup failures in `FuruComputeError` for consistent error handling.

## v0.0.2

- Auto-generate dashboard dev dummy data in a temporary Furu root (override with `FURU_DASHBOARD_DEV_DATA_DIR`).
- Replace `FURU_FORCE_RECOMPUTE` with `FURU_ALWAYS_RERUN` to bypass cache for specified classes or `ALL` (must be used alone), validating namespaces on load.
- Switch the build backend from hatchling to uv_build for packaging.
- Add richer compute lock wait logging and defer local locks while queued attempts from other backends are active.
- Default storage now lives under `<project>/furu-data` (pyproject.toml or git root), with version-controlled artifacts in `furu-data/artifacts` and a `FURU_VERSION_CONTROLLED_PATH` override.

## v0.0.1

- Hello Furu
