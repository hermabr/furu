from __future__ import annotations

import os
from typing import cast

from furu.results.context import DumpContext, LoadContext
from furu.results.nodes import ExternalNode, ManifestValue, make_external_node
from furu.results.rules import SaveSpec
from furu.utils import JsonValue, class_label


class NumpyNpyCodec:
    codec_id = "numpy_npy"

    def dump(self, value: object, ctx: DumpContext, spec: SaveSpec) -> ManifestValue:
        import numpy as np

        array = cast("np.ndarray", value)
        if array.dtype.hasobject:
            raise TypeError(
                f"{ctx.path.display()} cannot be saved with numpy_npy because object-dtype arrays require pickle support"
            )

        artifact_path = ctx.artifact_dir / "data.npy"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())

        meta: JsonValue = {
            "shape": [int(size) for size in array.shape],
            "dtype": str(array.dtype),
        }
        return make_external_node(
            serializer=self.codec_id,
            artifact_dir=ctx.artifact_dir_rel,
            lazy=bool(spec.lazy),
            python_type=class_label(type(array)),
            meta=meta,
        )

    def load(self, node: ManifestValue, ctx: LoadContext) -> object:
        import numpy as np

        external = cast(ExternalNode, node)
        return np.load(
            ctx.resolve_artifact_dir(external["artifact_dir"]) / "data.npy",
            allow_pickle=False,
        )
