# 03 — Dashboard assets bundled into the wheel

## Requirement (user request)
Dashboard frontend assets must be bundled into the wheel so `furu-dashboard serve` works post-install.

## Background
`get_frontend_dir()` expects something like:
- `furu/dashboard/frontend/dist` to exist within the installed package.

But packaging may not include dist assets by default unless configured as package data.

## Desired behavior
- The wheel includes `src/furu/dashboard/frontend/dist/**`.
- `furu-dashboard serve` can serve the SPA from an installed environment.
- If assets are missing, the CLI should fail fast with a clear error message.

## Implementation steps
- [x] Update build configuration:
  - `pyproject.toml` / packaging backend config to include dist files as package data.
  - If using hatchling/setuptools, add appropriate include rules.
  - Ensure source distributions also include assets where needed.
- [x] Add a packaging verification test:
  - Build wheel in CI (or locally in test) and inspect wheel contents, OR
  - Use `importlib.resources` to assert dist files are accessible at runtime.
- [x] Ensure `get_frontend_dir()` uses `importlib.resources.files(...)` rather than filesystem-relative paths if possible.

## Acceptance criteria
- Wheel contains dist assets.
- `python -c "from furu.dashboard.main import get_frontend_dir; print(get_frontend_dir())"` works in an installed env.
- A minimal smoke test passes that the index.html exists.

## Tests
- [x] Add a test that uses `importlib.resources` to locate `index.html` inside the package.
- [x] If you have a build step in CI, add a wheel inspection step (recommended).

## Progress log
- Status: [x] done
- Notes: Added uv build backend source include and importlib.resources test for frontend assets.
