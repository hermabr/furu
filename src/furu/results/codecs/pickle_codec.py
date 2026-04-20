from __future__ import annotations

import pickle
from typing import Any

from furu.utils import JsonValue


class PickleCodec:
    codec_id = "python.pickle.v1"

    def dump(self, value: Any, ctx) -> JsonValue | None:
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        with (ctx.artifact_dir / "data.pkl").open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def load(self, ctx, meta: JsonValue | None) -> Any:
        with (ctx.artifact_dir / "data.pkl").open("rb") as f:
            return pickle.load(f)
