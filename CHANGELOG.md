# Changelog

## Unreleased

- Add dependency discovery via `_dependencies()` and `_get_dependencies()` with recursive traversal and de-duplication, plus `DependencySpec`/`DependencyChzSpec` typing helpers.
- Include direct dependencies in `Furu` hashing to invalidate caches when implicit dependencies change.
- Record migration events with separate namespace/hash fields instead of composite IDs.

## v0.0.2

- Auto-generate dashboard dev dummy data in a temporary Furu root (override with `FURU_DASHBOARD_DEV_DATA_DIR`).
- Replace `FURU_FORCE_RECOMPUTE` with `FURU_ALWAYS_RERUN` to bypass cache for specified classes or `ALL` (must be used alone), validating namespaces on load.
- Switch the build backend from hatchling to uv_build for packaging.
- Add richer compute lock wait logging and defer local locks while queued attempts from other backends are active.
- Default storage now lives under `<project>/furu-data` (pyproject.toml or git root), with version-controlled artifacts in `furu-data/artifacts` and a `FURU_VERSION_CONTROLLED_PATH` override.

## v0.0.1

- Hello Furu
