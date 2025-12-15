# Repository Guidelines

## Project Structure & Module Organization

- Source code lives in `src/huldra/` (package is split across modules; `src/huldra/__init__.py` is a thin re-export layer).
- Key subpackages:
  - `src/huldra/core/`: `Huldra` and `HuldraList`
  - `src/huldra/storage/`: `StateManager`, `MetadataManager`
  - `src/huldra/serialization/`: `HuldraSerializer` (+ optional pydantic support)
  - `src/huldra/runtime/`: scoped logging, `.env` loading, tracebacks
  - `src/huldra/adapters/`: integrations like `SubmititAdapter`
- This repo uses a “src layout”; import the package as `huldra`, not from the repo root.
- Tests live in `tests/` at the repo root (pytest).
- Runnable examples live in `examples/`.

## Build, Test, and Development Commands

This project is managed with `uv` (`uv.lock` is committed).

- Install dependencies: `uv sync`
- Run a one-off command in the locked env: `uv run python -c "import huldra; print(huldra.__name__)"`
- Lint: `uv run ruff check .`
- Type check: `uv run ty check`
- Run tests: `uv run pytest`
- Build distributions (wheel/sdist): `uv build`
- Run examples: `uv run python examples/run_train.py`

If you don’t use `uv`, you can still use standard tooling, but keep `pyproject.toml` and `uv.lock` in sync.

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints for public APIs, and clear docstrings where behavior isn’t obvious.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer small, focused helpers over large “god” functions; this codebase is a library, so API clarity matters.
- Keep diffs focused; avoid large, noisy reformat-only changes.

## Testing Guidelines

- Use `pytest` and `tests/test_*.py`.
- Keep tests deterministic and avoid writing to the project root; use temp dirs (e.g., `tmp_path`) and env overrides.

## Commit & Pull Request Guidelines

- Commits use short, imperative subjects (often lowercase), e.g. `fix typing`, `add raw data path to huldra config`.
- Keep commits scoped; separate refactors from behavior changes where practical.
- PRs should include: a clear summary, rationale, and notes on breaking changes or new config/env vars.

## Security & Configuration Tips

- Local configuration is loaded from `.env` (ignored by git). Don’t commit secrets.
- Huldra storage defaults to `./data-huldra/`; override with `HULDRA_PATH`. Other knobs are `HULDRA_*` env vars.
- Logging uses stdlib `logging` and routes logs to the active holder’s artifact directory during `load_or_create()`; call `huldra.configure_logging()` to install the handler eagerly if desired.
