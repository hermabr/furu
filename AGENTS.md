# Repository Guidelines for AI Agents

## Critical Rules

**Always update `CHANGELOG.md`** for any user-visible API or package experience change.
- Keep the top changelog section as `## Unreleased`; `make release-*` replaces it with `## vX.Y.Z` when creating release PRs.

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

- The project was renamed from huldra to furu. If I ever refer to huldra, assume I mean furu.
- `furu_hash` is globally unique; do not build composite IDs (e.g., namespace + `furu_hash`).

## Workflow

- For feature requests, default to creating a new branch, making changes, committing when it makes sense, pushing the branch, and opening a pull request when the work is done.
- You may use git actions (commit, push, PR) without asking for permission unless the user explicitly forbids it.
- Never push to `main` or merge a pull request unless the user explicitly asks.
- When creating a pull request, write a detailed body that explains what changed, notes new or updated API behavior, and includes usage examples when relevant.

---

## Project Structure

```
src/furu/           # Main package (src layout - import as `furu`)
  core/               # Furu and FuruList classes
  storage/            # StateManager, MetadataManager
  serialization/      # FuruSerializer (+ pydantic support)
  runtime/            # Scoped logging, .env loading, tracebacks
  adapters/           # Integrations (SubmititAdapter)
  dashboard/          # FastAPI dashboard (optional)
tests/                # pytest tests
examples/             # Runnable examples
dashboard-frontend/   # React/TypeScript dashboard frontend
e2e/                  # Playwright end-to-end tests
```

---

## Frontend Notes

- Use shadcn/ui for frontend components.

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
uv run pytest tests/test_furu_core.py -v

# Run a specific test function
uv run pytest tests/test_furu_core.py::test_exists_reflects_success_state -v

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
from ..config import FURU_CONFIG
```

### Type Annotations

- **Required** on all public APIs (functions, methods, class attributes)
- Use modern syntax: `X | None` not `Optional[X]`
- Use concrete types, not `Any`
- Use generics where reasonable: `class Furu[T](ABC):`

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
- Private/internal names prefixed with `_` (e.g., `_FuruState`, `_iso_now`)

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
- Use `furu_tmp_root` fixture (from `conftest.py`) for isolated Furu config
- Keep tests deterministic - no writing to project root
- Test functions named `test_<description>`

```python
def test_exists_reflects_success_state(furu_tmp_root) -> None:
    obj = Dummy()
    assert obj.exists() is False
    obj.load_or_create()
    assert obj.exists() is True
```

### Test Coverage Requirements

**ALWAYS write extensive tests when adding new features.** Tests should cover:

1. **Happy path** - The feature works as expected with valid inputs
2. **Edge cases** - Empty inputs, boundary values, None/null values
3. **Filter combinations** - When adding filters, test each filter individually AND in combination
4. **Error cases** - Invalid inputs, missing data, malformed requests
5. **Integration** - Test the full stack (API endpoints, not just internal functions)

For dashboard features specifically:
- Add tests in `tests/dashboard/test_scanner.py` for scanner/filtering logic
- Add tests in `tests/dashboard/test_api.py` for API endpoint behavior
- Update `dashboard-frontend/src/api.test.ts` for frontend schema validation

Example test structure for a new filter:
```python
def test_filter_by_new_field(furu_tmp_root) -> None:
    """Test filtering by the new field."""
    # Setup: create experiments with different field values
    # Test: filter returns only matching experiments
    # Verify: correct count and correct experiments returned

def test_filter_by_new_field_no_match(furu_tmp_root) -> None:
    """Test filter returns empty when no experiments match."""

def test_filter_by_new_field_combined(furu_tmp_root) -> None:
    """Test new filter works in combination with existing filters."""
```

### Dashboard Test Performance

**Use module-scoped fixtures for read-only tests.** Creating experiments is slow, so:

1. **Prefer `populated_furu_root`** (module-scoped) over `temp_furu_root` (function-scoped)
2. **Extend `_create_populated_experiments()`** in `conftest.py` when you need new test data
3. **Only use `temp_furu_root`** when tests must mutate state or need isolated data

```python
# Good - uses shared fixture, fast
def test_filter_by_backend(client: TestClient, populated_furu_root: Path) -> None:
    response = client.get("/api/experiments?backend=local")
    assert response.json()["total"] == 3  # Uses pre-created data

# Slow - creates experiments per test (avoid unless necessary)
def test_something(client: TestClient, temp_furu_root: Path) -> None:
    create_experiment_from_furu(...)  # Slow!
```

---

## Commit Guidelines

- Short, imperative subjects (often lowercase)
- Examples: `fix typing`, `add raw data path to furu config`
- Keep commits scoped; separate refactors from behavior changes

---

## Environment & Configuration

- Local config from `.env` (gitignored); don't commit secrets
- Storage defaults to `<project>/furu-data/` (pyproject.toml or git root); override with `FURU_PATH`.
  Use `FURU_VERSION_CONTROLLED_PATH` to override where version-controlled artifacts are stored.
- Python version: >=3.12
