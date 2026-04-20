from __future__ import annotations

from typing import TYPE_CHECKING, cast

from furu.results.codecs import _fsync_path
from furu.results.context import DumpContext, LoadContext
from furu.results.nodes import ExternalNode, ManifestValue, make_external_node
from furu.results.rules import SaveSpec
from furu.utils import JsonValue, class_label

if TYPE_CHECKING:
    import polars as pl


class PolarsParquetCodec:
    codec_id = "polars_parquet"

    def dump(self, value: object, ctx: DumpContext, spec: SaveSpec) -> ManifestValue:
        frame = cast("pl.DataFrame", value)
        artifact_path = ctx.artifact_dir / "data.parquet"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(artifact_path)
        _fsync_path(artifact_path)

        meta: JsonValue = {
            "shape": [int(frame.height), int(frame.width)],
            "columns": list(frame.columns),
        }
        return make_external_node(
            serializer=self.codec_id,
            artifact_dir=ctx.artifact_dir_rel,
            lazy=bool(spec.lazy),
            python_type=class_label(type(frame)),
            meta=meta,
        )

    def load(self, node: ManifestValue, ctx: LoadContext) -> object:
        import polars as pl

        external = cast(ExternalNode, node)
        return pl.read_parquet(
            ctx.resolve_artifact_dir(external["artifact_dir"]) / "data.parquet"
        )
