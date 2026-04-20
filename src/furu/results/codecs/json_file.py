from __future__ import annotations

import json
from typing import cast

from furu.results.codecs import _write_json
from furu.results.context import DumpContext, LoadContext
from furu.results.nodes import ExternalNode, ManifestValue, make_external_node
from furu.results.rules import SaveSpec


class JsonFileCodec:
    codec_id = "json_file"

    def dump(
        self,
        value: ManifestValue,
        ctx: DumpContext,
        spec: SaveSpec,
    ) -> ManifestValue:
        _write_json(ctx.artifact_dir / "data.json", value)
        return make_external_node(
            serializer=self.codec_id,
            artifact_dir=ctx.artifact_dir_rel,
            lazy=bool(spec.lazy),
            python_type="builtins.object",
        )

    def load(self, node: ManifestValue, ctx: LoadContext) -> object:
        external = cast(ExternalNode, node)
        path = ctx.resolve_artifact_dir(external["artifact_dir"]) / "data.json"
        return ctx.load(
            cast(ManifestValue, json.loads(path.read_text(encoding="utf-8")))
        )
