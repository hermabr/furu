from __future__ import annotations

import json
from typing import Any

from furu.utils import JsonValue


class JsonTreeCodec:
    codec_id = "furu.json-tree.v1"

    def dump(self, value: Any, ctx) -> JsonValue | None:
        from furu.results.walker import _dump_json_tree_with_new_encoder

        meta = _dump_json_tree_with_new_encoder(value, ctx)
        data_path = ctx.artifact_dir / "data.json"
        data_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return None

    def load(self, ctx, meta: JsonValue | None) -> Any:
        from furu.results.walker import _load_json_tree_from_path

        return _load_json_tree_from_path(ctx.artifact_dir / "data.json", ctx)
