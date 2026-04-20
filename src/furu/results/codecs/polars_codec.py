from __future__ import annotations

import importlib
from typing import Any

from furu.results.errors import ResultCodecError
from furu.utils import JsonValue


class PolarsDataFrameCodec:
    codec_id = "polars.dataframe.parquet.v1"

    def dump(self, value: Any, ctx) -> JsonValue | None:
        pl = importlib.import_module("polars")

        if not isinstance(value, pl.DataFrame):
            raise ResultCodecError(
                f"Codec {self.codec_id} expected polars.DataFrame, got {type(value).__name__}"
            )

        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        value.write_parquet(ctx.artifact_dir / "data.parquet")
        return {
            "rows": int(value.height),
            "columns": list(value.columns),
            "schema": {name: str(dtype) for name, dtype in value.schema.items()},
        }

    def load(self, ctx, meta: JsonValue | None) -> Any:
        pl = importlib.import_module("polars")

        return pl.read_parquet(ctx.artifact_dir / "data.parquet")
