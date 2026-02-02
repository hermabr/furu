# 07 — Dashboard SPA serving security (path traversal)

## Issue
`serve_spa` constructs `frontend_dir / full_path` without preventing `..` traversal.
This could allow reading files outside dist.

## Desired behavior
- Reject any path with `..` segments or that resolves outside `frontend_dir`.
- Prefer FastAPI `StaticFiles` for static assets + SPA fallback for unknown routes.

## Implementation steps
- [x] Normalize and resolve requested path.
- [x] Ensure resolved path is within frontend_dir; else return 404.
- [x] Add tests for traversal attempts (`/../pyproject.toml`) return 404.

## Acceptance criteria
- Traversal attempts do not escape dist directory.
- Tests cover this.

## Progress log
- Status: [x] done
- Notes: Added path traversal checks in SPA handler with regression test.
