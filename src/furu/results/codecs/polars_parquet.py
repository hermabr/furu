from __future__ import annotations

from pathlib import Path

from furu.utils import JsonValue


class PolarsParquetCodec:
    codec_id = "polars.dataframe.parquet.v1"

    def dump(
        self,
        value: object,
        artifact_dir: Path,
        ctx: object,
    ) -> dict[str, JsonValue] | None:
        del ctx
        import polars as pl

        if not isinstance(value, pl.DataFrame):
            raise TypeError(
                f"{self.codec_id} expected polars.DataFrame, got {type(value).__name__}"
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "data.parquet"
        value.write_parquet(path)
        return {
            "height": value.height,
            "width": value.width,
        }

    def load(
        self,
        artifact_dir: Path,
        meta: dict[str, JsonValue] | None,
        ctx: object,
    ) -> object:
        del meta, ctx
        import polars as pl

        path = artifact_dir / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing Polars artifact at {path}")
        return pl.read_parquet(path)
