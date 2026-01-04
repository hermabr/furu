# Repository Guidelines for AI Agents

## Critical Rules

**DO NOT USE** the following patterns in this codebase:
- `typing.Optional` - Use `X | None` instead
- `typing.Any` - Only acceptable when the type is truly unknowable at compile time (e.g., deserializing arbitrary JSON, interacting with untyped third-party libraries). Always prefer protocols, generics, or concrete types.
- `object` as a type annotation - Use specific types, protocols, or generics instead
- `dict` without specific value types - Use Pydantic models, dataclasses, or TypedDict instead of `dict[str, Any]` or `dict[str, object]`. Simple dicts like `dict[str, str]` or `dict[str, int]` should also be avoided if it is possible to know exactly what the keys and structure of the data is.
- `try/except` for error recovery - Prefer happy path and let errors crash; only use try/except for cleanup or resource management
- Backward compatibility shims or aliases - Do not add backward compatibility code; refactor all usages directly

**ALWAYS** use `make` commands rather than running tools directly:
- `make lint` not `uv run ruff check && uv run ty check`
- `make test` not `uv run pytest`
- `make check` for both lint and test

**AFTER** making changes, verify your work:
- For type/import changes: run `make lint`
- For logic/behavior changes: run `make test` or `make test-all` or `make dashboard-test` or `make dashboard-test-e2e`

---

## Project Structure

```
src/huldra/           # Main package (src layout - import as `huldra`)
  core/               # Huldra and HuldraList classes
  storage/            # StateManager, MetadataManager
  serialization/      # HuldraSerializer (+ pydantic support)
  runtime/            # Scoped logging, .env loading, tracebacks
  adapters/           # Integrations (SubmititAdapter)
  dashboard/          # FastAPI dashboard (optional)
tests/                # pytest tests
examples/             # Runnable examples
dashboard-frontend/   # React/TypeScript dashboard frontend
e2e/                  # Playwright end-to-end tests
```

---

## Build, Test, and Lint Commands

This project uses `uv` for dependency management.

### Core Commands (use these)

| Command | Description |
|---------|-------------|
| `make lint` | Run ruff check + ty type checker |
| `make test` | Run pytest on `tests/` |
| `make check` | Run lint + test |
| `make build` | Build wheel/sdist (runs tests first) |
| `make clean` | Remove caches and build artifacts |

### Running a Single Test

```bash
# Run a specific test file
uv run pytest tests/test_huldra_core.py -v

# Run a specific test function
uv run pytest tests/test_huldra_core.py::test_exists_reflects_success_state -v

# Run tests matching a pattern
uv run pytest -k "test_load" -v
```

### Dashboard Commands

| Command | Description |
|---------|-------------|
| `make dashboard-dev` | Start dev servers (backend + frontend) |
| `make dashboard-test` | Run dashboard backend tests |
| `make dashboard-test-e2e` | Run Playwright e2e tests |
| `make dashboard-test-all` | Run all dashboard tests |

### Frontend Commands

| Command | Description |
|---------|-------------|
| `make frontend-lint` | Run frontend TypeScript type checker |
| `make frontend-test` | Run frontend unit tests |
| `make frontend-build` | Build frontend for production |
| `make frontend-generate` | Generate OpenAPI spec and TypeScript client |

---

## Code Style

### Imports

Order imports as: stdlib, third-party, local. Use absolute imports for cross-module references.

```python
import contextlib
from pathlib import Path
import chz
from ..config import HULDRA_CONFIG
```

### Type Annotations

- **Required** on all public APIs (functions, methods, class attributes)
- Use modern syntax: `X | None` not `Optional[X]`
- Use concrete types, not `Any`
- Use generics where reasonable: `class Huldra[T](ABC):`

```python
# Good - specific types
def process(data: dict[str, str]) -> list[int] | None:
    ...

# Good - use Pydantic models or dataclasses for structured data
class UserConfig(BaseModel):
    name: str
    settings: dict[str, str]

def load_config(path: Path) -> UserConfig:
    ...

# Bad - DO NOT USE
def process(data: Dict[str, Any]) -> Optional[List[int]]:
    ...

# Bad - DO NOT USE untyped dicts
def load_config(path: Path) -> dict[str, Any]:  # NO - use a model
    ...
```

### Naming Conventions

- `snake_case` for functions, variables, module names
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Private/internal names prefixed with `_` (e.g., `_HuldraState`, `_iso_now`)

### Error Handling

**Prefer happy path with early errors over defensive checks.** Don't wrap code in if-statements to handle error cases - let it crash or raise explicitly.

```python
# Good - assume happy path, crash if invariant violated
data = json.loads(path.read_text())

# Good - explicit early error instead of nested ifs
if not condition:
    raise ValueError("condition must be true")
# proceed with happy path...

# Bad - defensive if-checks for non-happy paths
if path.exists():
    data = json.loads(path.read_text())
else:
    data = {}  # NO - hides bugs, use happy path

# Bad - swallowing errors
try:
    data = json.loads(path.read_text())
except Exception:
    data = {}  # NO - this hides bugs
```

**Only use try/except for:**
1. Resource cleanup (use `contextlib.suppress` for ignoring cleanup errors)
2. Converting exceptions to domain-specific errors
3. Explicit user-facing error messages

### Formatting

- 4-space indentation
- Line length: follow ruff defaults
- Use trailing commas in multi-line structures
- Prefer small, focused functions over large ones

---

## Testing Guidelines

- All tests in `tests/` directory using pytest
- Use `tmp_path` fixture for temporary directories
- Use `huldra_tmp_root` fixture (from `conftest.py`) for isolated Huldra config
- Keep tests deterministic - no writing to project root
- Test functions named `test_<description>`

```python
def test_exists_reflects_success_state(huldra_tmp_root) -> None:
    obj = Dummy()
    assert obj.exists() is False
    obj.load_or_create()
    assert obj.exists() is True
```

---

## Commit Guidelines

- Short, imperative subjects (often lowercase)
- Examples: `fix typing`, `add raw data path to huldra config`
- Keep commits scoped; separate refactors from behavior changes

---

## Environment & Configuration

- Local config from `.env` (gitignored); don't commit secrets
- Storage defaults to `./data-huldra/`; override with `HULDRA_PATH`
- Python version: >=3.12
