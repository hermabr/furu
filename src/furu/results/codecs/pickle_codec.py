from __future__ import annotations

import pickle
from typing import cast

from furu.results.codecs import _write_bytes
from furu.results.context import DumpContext, LoadContext
from furu.results.nodes import ExternalNode, ManifestValue, make_external_node
from furu.results.rules import SaveSpec
from furu.utils import class_label


class PickleCodec:
    codec_id = "pickle"

    def dump(self, value: object, ctx: DumpContext, spec: SaveSpec) -> ManifestValue:
        _write_bytes(ctx.artifact_dir / "data.pkl", pickle.dumps(value))
        return make_external_node(
            serializer=self.codec_id,
            artifact_dir=ctx.artifact_dir_rel,
            lazy=bool(spec.lazy),
            python_type=class_label(type(value)),
        )

    def load(self, node: ManifestValue, ctx: LoadContext) -> object:
        external = cast(ExternalNode, node)
        return pickle.loads(
            (
                ctx.resolve_artifact_dir(external["artifact_dir"]) / "data.pkl"
            ).read_bytes()
        )
