# Changelog

## v0.1.3
- Add `GREN_CACHE_METADATA` flag to cache git/environment info collection (opt-in, reduces overhead for batch scripts).
- Add `clear_metadata_cache()` to manually invalidate cached metadata.

## v0.1.2
- Add `GREN_FORCE_RECOMPUTE` to force recomputation for selected Gren classes.
- Update release automation to create PRs with version bumps and auto-release on main.
- Improve CI to build frontend assets and install e2e dependencies.

## v0.1.1
- Add `GREN_REQUIRE_GIT` and `GREN_REQUIRE_GIT_REMOTE` flags for git metadata collection.

## v0.1.0
- First public release of gren.
