from __future__ import annotations

import dataclasses
import importlib
import json
import os
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from furu.locking import LockLostError
from furu.utils import fully_qualified_name

FURU_TAG = "$furu"
NUMPY_NDARRAY_NPY_CODEC = "$furu.numpy.ndarray.npy"


def is_complete(*, result_dir: Path) -> bool:
    return (result_dir / "manifest.json").is_file()


def save_result(
    *,
    result: Any,
    result_dir: Path,
    has_lock: Callable[[], bool],
    lock_path: Path,
) -> None:
    if not has_lock():
        raise LockLostError(f"lost lock at {lock_path} before writing final result")

    parent = result_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = parent / f"{result_dir.name}.tmp.{secrets.token_hex(8)}"
    tmp_dir.mkdir(parents=True, exist_ok=False)

    renamed = False
    try:
        manifest = _walk(result, root_dir=tmp_dir, rel_path=())
        with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if not has_lock():
            raise LockLostError(
                f"lost lock at {lock_path} before renaming temporary result"
            )
        os.rename(tmp_dir, result_dir)
        renamed = True
        if not has_lock():
            raise LockLostError(
                f"lost lock at {lock_path} after renaming temporary result"
            )
    except BaseException:
        if not renamed:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def load_result(*, result_dir: Path) -> Any:
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    return _unwalk(manifest, root_dir=result_dir)


def _check_key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError(
            f"furu result dict keys must be str, got {type(key).__name__}: {key!r}"
        )
    if key == "":
        raise ValueError("furu result dict keys must not be empty")
    if "/" in key:
        raise ValueError(f"furu result dict key must not contain '/': {key!r}")
    if key == FURU_TAG:
        raise ValueError(f"furu result dict key {FURU_TAG!r} is reserved")
    return key


def _arr_idx(i: int, n: int) -> str:
    width = max(1, len(str(max(0, n - 1))))
    return f"arr_idx_{i:0{width}d}"


def _walk(value: Any, *, root_dir: Path, rel_path: tuple[str, ...]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        n = len(value)
        return [
            _walk(item, root_dir=root_dir, rel_path=(*rel_path, _arr_idx(i, n)))
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        n = len(value)
        return {
            FURU_TAG: {
                "kind": "tuple",
                "items": [
                    _walk(
                        item,
                        root_dir=root_dir,
                        rel_path=(*rel_path, _arr_idx(i, n)),
                    )
                    for i, item in enumerate(value)
                ],
            }
        }
    if isinstance(value, dict):
        return {
            _check_key(k): _walk(v, root_dir=root_dir, rel_path=(*rel_path, k))
            for k, v in value.items()
        }
    if isinstance(value, PydanticBaseModel):
        return {
            FURU_TAG: {
                "kind": "pydantic",
                "type": fully_qualified_name(type(value)),
                "fields": {
                    _check_key(k): _walk(v, root_dir=root_dir, rel_path=(*rel_path, k))
                    for k, v in value.model_dump().items()
                },
            }
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            FURU_TAG: {
                "kind": "dataclass",
                "type": fully_qualified_name(type(value)),
                "fields": {
                    _check_key(f.name): _walk(
                        getattr(value, f.name),
                        root_dir=root_dir,
                        rel_path=(*rel_path, f.name),
                    )
                    for f in dataclasses.fields(value)
                },
            }
        }
    saved = _try_save_ndarray(value, root_dir=root_dir, rel_path=rel_path)
    if saved is not None:
        return saved
    raise TypeError(
        f"unsupported type for furu result at "
        f"{'/'.join(rel_path) if rel_path else '<root>'}: {type(value).__name__}"
    )


def _try_save_ndarray(
    value: Any, *, root_dir: Path, rel_path: tuple[str, ...]
) -> dict[str, Any] | None:
    try:
        import numpy as np  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError:
        return None
    if not isinstance(value, np.ndarray):
        return None
    rel = ("artifacts", *rel_path)
    out_dir = root_dir.joinpath(*rel)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data.npy", value, allow_pickle=False)
    return {
        FURU_TAG: {
            "kind": "external",
            "codec": NUMPY_NDARRAY_NPY_CODEC,
            "path": "/".join(rel),
            "meta": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
        }
    }


def _unwalk(value: Any, *, root_dir: Path) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_unwalk(item, root_dir=root_dir) for item in value]
    if isinstance(value, dict):
        if FURU_TAG in value:
            tag = value[FURU_TAG]
            kind = tag["kind"]
            if kind == "tuple":
                return tuple(_unwalk(x, root_dir=root_dir) for x in tag["items"])
            if kind == "external":
                codec = tag["codec"]
                if codec == NUMPY_NDARRAY_NPY_CODEC:
                    import numpy as np  # noqa: PLC0415  # ty: ignore[unresolved-import]

                    return np.load(
                        root_dir / tag["path"] / "data.npy", allow_pickle=False
                    )
                raise ValueError(f"unknown furu external codec: {codec!r}")
            if kind in ("dataclass", "pydantic"):
                cls = _resolve_type(tag["type"])
                return cls(
                    **{
                        k: _unwalk(v, root_dir=root_dir)
                        for k, v in tag["fields"].items()
                    }
                )
            raise ValueError(f"unknown furu kind: {kind!r}")
        return {k: _unwalk(v, root_dir=root_dir) for k, v in value.items()}
    raise TypeError(f"unsupported value in furu manifest: {type(value).__name__}")


def _resolve_type(fqn: str) -> type:
    module_name, _, qualname = fqn.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"invalid type fully qualified name: {fqn!r}")
    return getattr(importlib.import_module(module_name), qualname)
