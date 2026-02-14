# Changelog

## Unreleased

- Make `FuruSerializer.from_dict()` strict by default and fall back only when requested (`strict=False`), while still allowing stale-ref migration to opt into relaxed loading (`strict=False`). ([#55](https://github.com/hermabr/furu/pull/55))
- Ignore non-init dataclass fields during lazy fallback reconstruction so nested schema-migration paths no longer fail with unexpected keyword arguments on stale artifacts. ([#55](https://github.com/hermabr/furu/pull/55))
- Return fallback payloads as attribute-accessible dictionaries in relaxed mode so nested field access like `obj.age` works even when full reconstruction is unavailable. ([#55](https://github.com/hermabr/furu/pull/55))
- Coerce dict-valued nested fields to declared dataclass/chz types during `FuruRef.migrate(..., strict_types=True)` with strict key validation (unknown fields rejected, required fields required), union-branch probing across all members, and explicit `__class__` disambiguation when multiple union members match. ([#55](https://github.com/hermabr/furu/pull/55))
- Limit relaxed deserialization class-import fallback to missing-module cases, so non-module import-time failures now surface instead of being silently swallowed. ([#55](https://github.com/hermabr/furu/pull/55))
- Preserve unknown dataclass fields by downgrading to relaxed fallback dicts (instead of silently dropping keys), and only swallow constructor `TypeError`s in relaxed mode when they indicate signature/schema mismatches. ([#55](https://github.com/hermabr/furu/pull/55))
- Resolve dataclass forward-reference path annotations during deserialization so string path payloads are reconstructed as `pathlib.Path` even when annotations are deferred/quoted. ([#55](https://github.com/hermabr/furu/pull/55))
- Validate constructor kwargs for non-dataclass/chz payloads with `inspect.signature()` before instantiation so relaxed mode falls back predictably on schema mismatches (missing/unexpected args) without relying on broad runtime `TypeError` handling. ([#55](https://github.com/hermabr/furu/pull/55))
- Harden relaxed deserialization fallback by avoiding `find_spec()` crashes for missing parent modules, tolerating unresolved type hints during dataclass reconstruction, and prioritizing payload keys over dict methods for attribute-style fallback access. ([#55](https://github.com/hermabr/furu/pull/55))
- Resolve dataclass field types via annotations during strict migration coercion so union/dataclass checks keep working when field annotations are deferred (`from __future__ import annotations`). ([#55](https://github.com/hermabr/furu/pull/55))
- Validate serialized `__class__` markers before deserialization: strict mode now raises a clear `TypeError` for non-string markers, while relaxed mode safely falls back to attribute-accessible dict payloads. ([#55](https://github.com/hermabr/furu/pull/55))
- Allow `FuruRef.migrate()` transforms to return `furu.MIGRATION_SKIP` for intentional per-ref no-op migrations; the call returns `furu.MIGRATION_SKIPPED`, still verifies original-artifact success, and then exits without strict-type checks or alias writes. ([#56](https://github.com/hermabr/furu/pull/56))
- Improve relaxed deserialization fallback for custom classes by handling positional-only constructor mismatches (including positional-only-as-keyword cases) and by falling back on schema-like constructor errors when signature introspection is unavailable. ([#56](https://github.com/hermabr/furu/pull/56))
- Relax strict migration coercion for nested dataclass/chz dict payloads so omitted fields with defaults are accepted while extra unknown fields are still rejected. ([#56](https://github.com/hermabr/furu/pull/56))
- Fix relaxed deserialization for dataclasses so constructor schema mismatches (including required `InitVar` parameters) fall back to attribute-accessible dict payloads in `strict=False` mode. ([#57](https://github.com/hermabr/furu/pull/57))
- Treat `importlib.util.find_spec()` probe failures as non-importable in relaxed deserialization, so dynamic/spec-less module states now degrade to fallback payloads instead of raising. ([#57](https://github.com/hermabr/furu/pull/57))
- Accept `TransformSkip()` token instances in `FuruRef.migrate()` skip transforms, in addition to the `furu.MIGRATION_SKIP` singleton. ([#57](https://github.com/hermabr/furu/pull/57))

## v0.0.11

- Treat refs that match `schema_key` but fail hydration as stale, so `Furu.all_current()`/`Furu.all_successful()` skip unloadable entries and `Furu.all_stale_refs()` surfaces them for migration. ([#53](https://github.com/hermabr/furu/pull/53))

## v0.0.10

- Replace `Furu.current()`/`Furu.successful()`/`Furu.stale()` with `Furu.all_current()`/`Furu.all_successful()`/`Furu.all_stale_refs()`, with current/successful APIs returning hydrated objects and stale results remaining ref-based for migration workflows.
- Add `Furu.load_existing()` for load-only access to cached artifacts, raising `FuruMissingArtifact` when artifacts are missing or invalid.
- Add `FuruRef.migrate(transform, dry_run=False, strict_types=True)` to migrate one reference via a callback that returns the target object, with optional validation-only dry runs and strict runtime field type enforcement ([#50](https://github.com/hermabr/furu/pull/50)).

## v0.0.9

- Rename migration reference field `FuruRef.directory` to `FuruRef.furu_dir`, and update Slurm executor selection to `_executor()`-declared specs (including multi-spec nodes), dropping `specs` mapping inputs from `run_slurm_pool()`/`submit_slurm_dag()` and exporting `resolve_executor_spec` from `furu.execution`.
- Improve executor observability by recording selected Slurm spec and worker host/node placement in `attempt.scheduler`, and by routing slurm-pool worker stdout/stderr to per-worker queue logs.
- Rework local and slurm-pool scheduling loops to build run state once, keep sticky `DONE` snapshots per run, and expose independent reconcile/scan/health polling cadences.
- Extend `run_slurm_pool(max_workers_total=...)` to support integer totals or per-spec limits with enforced per-spec caps, inferred totals, and fail-fast validation for missing active-spec limits; add `examples/run_local_executor_benchmark.py` for nested 10k-node DAG benchmarking.

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
