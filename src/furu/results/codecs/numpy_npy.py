from __future__ import annotations

from pathlib import Path

from furu.utils import JsonValue


class NumpyNpyCodec:
    codec_id = "numpy.ndarray.npy.v1"

    def dump(
        self,
        value: object,
        artifact_dir: Path,
        ctx: object,
    ) -> dict[str, JsonValue] | None:
        del ctx
        import numpy as np

        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"{self.codec_id} expected numpy.ndarray, got {type(value).__name__}"
            )
        if value.dtype.hasobject:
            raise TypeError(
                "numpy.ndarray.npy.v1 does not support object-dtype arrays; "
                "use an explicit custom codec if you need pickle-based semantics"
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "data.npy"
        with path.open("wb") as f:
            np.save(f, value, allow_pickle=False)
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }

    def load(
        self,
        artifact_dir: Path,
        meta: dict[str, JsonValue] | None,
        ctx: object,
    ) -> object:
        del meta, ctx
        import numpy as np

        path = artifact_dir / "data.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing NumPy artifact at {path}")
        with path.open("rb") as f:
            return np.load(f, allow_pickle=False)
