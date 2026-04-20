# furu

> **Note:** `v0.0.x` is alpha and may (will) include breaking changes.

## Installation

```bash
uv add furu # or pip install furu
```

## Result Bundles

Furu now persists completed results as versioned JSON-first bundles under
`<data_dir>/result/`:

```text
<data_dir>/
├── result/
│   ├── manifest.json
│   └── artifacts/
│       └── ...
└── .furu/
    ├── metadata.json
    ├── compute.lock
    └── run.log
```

`result.pkl` is no longer used.

Plain JSON-compatible results stay inline in `result/manifest.json`:

```python
import furu


class BuildSummary(furu.Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "metrics": {"accuracy": 0.91},
            "tags": ["baseline", "v1"],
        }
```

Large values can be externalized explicitly:

```python
class Train(furu.Furu[dict[str, object]]):
    def _create(self) -> dict[str, object]:
        return {
            "embeddings": furu.lazy([[1.0, 2.0], [3.0, 4.0]]),
        }
```

Field annotations can choose a codec:

```python
from dataclasses import dataclass
from typing import Annotated

import furu


@dataclass(frozen=True)
class Output:
    payload: Annotated[list[int], furu.SaveWith("furu.json.v1")]
```

Custom codecs are registered through `_result_config()`:

```python
class MyCodec:
    codec_id = "my-value.v1"

    def dump(self, value, ctx):
        ...

    def load(self, ctx, meta):
        ...


class BuildValue(furu.Furu[object]):
    def _create(self) -> object:
        ...

    def _result_config(self) -> furu.ResultConfig:
        config = furu.ResultConfig.default()
        config.registry.register_type(MyValue, MyCodec())
        return config
```
