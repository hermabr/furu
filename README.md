# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Usage

```python
import furu


class Embedding(furu.Furu[list[float]]):
    text: str

    @classmethod
    def _create_batched(cls, objs: list["Embedding"]) -> list[list[float]]:
        return [[float(len(obj.text))] for obj in objs]


single = Embedding(text="hello")
batch = [Embedding(text="hello"), Embedding(text="world"), Embedding(text="hello")]

assert single.load_or_create() == [5.0]
assert furu.load_or_create(batch) == [[5.0], [5.0], [5.0]]
```

Each object still keeps its own artifact directory, `result.pkl`, `metadata.json`, and `run.log`. Batched creation only changes how unresolved objects are computed, not how results are stored.
