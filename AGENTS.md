# Repository Guidelines

## Project Structure & Module Organization

- Source code lives in `src/huldra/` (currently mostly in `src/huldra/__init__.py`).
- This repo uses a “src layout”; import the package as `huldra`, not from the repo root.
- There is no `tests/` directory yet. If you add tests, prefer `tests/` at the repo root.

## Build, Test, and Development Commands

This project is managed with `uv` (`uv.lock` is committed).

- Install dependencies: `uv sync`
- Run a one-off command in the locked env: `uv run python -c "import huldra; print(huldra.__name__)"`
- Build distributions (wheel/sdist): `uv build`

If you don’t use `uv`, you can still use standard tooling, but keep `pyproject.toml` and `uv.lock` in sync.

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints for public APIs, and clear docstrings where behavior isn’t obvious.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer small, focused helpers over large “god” functions; this codebase is a library, so API clarity matters.
- No formatter/linter is configured in-repo yet—avoid large, noisy reformat-only diffs.

## Testing Guidelines

- No test framework is set up yet. When adding tests, prefer `pytest` and `tests/test_*.py`.
- Keep tests deterministic and avoid writing to the project root; use temp dirs (e.g., `tmp_path`) and env overrides.

## Commit & Pull Request Guidelines

- Commits use short, imperative subjects (often lowercase), e.g. `fix typing`, `add raw data path to huldra config`.
- Keep commits scoped; separate refactors from behavior changes where practical.
- PRs should include: a clear summary, rationale, and notes on breaking changes or new config/env vars.

## Security & Configuration Tips

- Local configuration is loaded from `.env` (ignored by git). Don’t commit secrets.
- Huldra storage defaults to `./data-huldra/`; override with `HULDRA_PATH`. Other knobs are `HULDRA_*` env vars.
