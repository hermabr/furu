from __future__ import annotations

import importlib
from typing import Any

from furu.results.errors import ResultCodecError
from furu.utils import JsonValue


class NumpyArrayCodec:
    codec_id = "numpy.ndarray.npy.v1"

    def dump(self, value: Any, ctx) -> JsonValue | None:
        np = importlib.import_module("numpy")

        if not isinstance(value, np.ndarray):
            raise ResultCodecError(
                f"Codec {self.codec_id} expected numpy.ndarray, got {type(value).__name__}"
            )
        if value.dtype.hasobject:
            raise ResultCodecError(
                f"Codec {self.codec_id} rejects object-dtype arrays at save time"
            )

        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(ctx.artifact_dir / "data.npy", value, allow_pickle=False)
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }

    def load(self, ctx, meta: JsonValue | None) -> Any:
        np = importlib.import_module("numpy")

        return np.load(ctx.artifact_dir / "data.npy", allow_pickle=False)
