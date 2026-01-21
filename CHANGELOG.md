# Changelog

## Unreleased

- Auto-generate dashboard dev dummy data in a temporary Furu root (override with `FURU_DASHBOARD_DEV_DATA_DIR`).
- Replace `FURU_FORCE_RECOMPUTE` with `FURU_ALWAYS_RERUN` to bypass cache for specified classes or `ALL` (must be used alone), validating namespaces on load.
- Switch the build backend from hatchling to uv_build for packaging.
- Add richer compute lock wait logging and defer local locks while queued attempts from other backends are active.
- Store version-controlled artifacts under `furu-data/artifacts` next to `pyproject.toml`, with a `FURU_VERSION_CONTROLLED_PATH` override.

## v0.0.1

- Hello Furu
