# 06 — Release workflow should run tests/build before publish

## Issue
Release workflow publishes without running tests/lint/build pipelines, risking broken releases.

## Desired behavior
In `.github/workflows/release.yml`:
- run `make test-all` (or equivalent) before building/publishing
- run `make build` (or at minimum ensure frontend assets build step runs if relevant)
- only publish if tests pass

## Implementation steps
- [x] Update workflow steps to run tests before build/publish
- [x] Ensure it uses the same toolchain as CI (uv/poetry/etc.)
- [x] If dashboard frontend build is required, ensure release workflow produces dist assets.

## Acceptance criteria
- Release workflow fails if tests fail.
- Release workflow produces wheel with dashboard assets included.

## Progress log
- Status: [x] done
- Notes: Release workflow now runs make build to enforce tests/build before publish.
