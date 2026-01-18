# Changelog

## v0.2.0
- Add explicit migration alias tracking with overwrite logging and dashboard views for aliased vs original experiments.
- Update state migration handling to store migrated status in state.json and keep migration metadata in migration.json.
- Add migration regression tests for rename/added-field alias cases plus recompute detach events.
- Extend dashboard e2e fixtures/tests to cover migrated experiment tags and original/aliased navigation.
- Warn when migrating experiments that are not in a success state.
- Add `GREN_CACHE_METADATA` for time-based git info caching. Accepts `never`, `forever`, or duration like `5m`, `1h`. Default: `5m`.
- Add `clear_metadata_cache()` to manually invalidate cached metadata.

## v0.1.2
- Add `GREN_FORCE_RECOMPUTE` to force recomputation for selected Gren classes.
- Update release automation to create PRs with version bumps and auto-release on main.
- Improve CI to build frontend assets and install e2e dependencies.

## v0.1.1
- Add `GREN_REQUIRE_GIT` and `GREN_REQUIRE_GIT_REMOTE` flags for git metadata collection.

## v0.1.0
- First public release of gren.
