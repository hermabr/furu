# Changelog

## Unreleased

- Use `SUCCESS.json` markers for `exists()`/planner cache checks and alias resolution to avoid `state.json` reads.
- Move attempt heartbeats to compute-lock mtimes and remove `heartbeat_at` from state and dashboard APIs (breaking).
- Route local executor run logs to each artifact's `.furu/furu.log`.
- Include caller locations for get/dependency logs (including `furu.log` call sites).
- Create each artifact `.furu` directory once during computation setup to avoid repeated mkdirs.
- Skip `state.json` rewrites for no-op state updates (including heartbeat mismatches).
- Use non-hidden lock filenames inside `.furu` (`compute.lock`, `submit.lock`, `state.lock`).
- Document all environment variables in the README.
- Consolidate git metadata env config into `FURU_RECORD_GIT` and `FURU_ALLOW_NO_GIT_ORIGIN`.
- Stop auto-loading `.env` (call `furu.load_env()` if needed).
- Cache `Furu.furu_hash` and `Furu.furu_dir` on instance access.

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
