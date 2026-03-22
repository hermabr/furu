# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Configuration

Project defaults can live in `pyproject.toml`:

```toml
[tool.furu]
debug = false
data_dir = ".furu/data"
```

Runtime overrides win over `pyproject.toml`:

```python
from furu import configure

configure(data_dir="/tmp/furu-data", debug=True)
```
