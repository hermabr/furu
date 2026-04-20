from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from furu.results.nodes import WRAPPER_KEY
from furu.utils import JsonValue


class JsonFileCodec:
    codec_id = "json.file.v1"

    def dump(
        self,
        value: object,
        artifact_dir: Path,
        ctx: object,
    ) -> dict[str, JsonValue] | None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload_path = artifact_dir / "data.json"
        payload_path.write_text(
            json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return None

    def load(
        self,
        artifact_dir: Path,
        meta: dict[str, JsonValue] | None,
        ctx: object,
    ) -> object:
        del meta, ctx
        payload_path = artifact_dir / "data.json"
        if not payload_path.exists():
            raise FileNotFoundError(f"missing JSON artifact at {payload_path}")
        return json.loads(payload_path.read_text(encoding="utf-8"))


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return cast(JsonValue, value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "json.file.v1 only supports dictionaries with string keys"
                )
            if key == WRAPPER_KEY:
                raise TypeError(
                    "json.file.v1 cannot store dictionaries containing the reserved '$furu' key"
                )
            out[key] = _json_value(item)
        return out
    raise TypeError(
        f"json.file.v1 only supports JSON-compatible values, got {type(value).__name__}"
    )
