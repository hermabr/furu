from __future__ import annotations

import importlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from furu.locking import LockLostError
from furu.utils import JsonValue, fully_qualified_name

FURU_KEY = "$furu"
NUMPY_CODEC = "$furu.numpy.ndarray.npy"

type HasLock = Callable[[], bool]


def save_result(obj: Any, path: Path, *, has_lock: HasLock, lock_path: Path) -> None:
    _require_lock(has_lock, lock_path, "before writing final result")
    tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    try:
        manifest = _encode(obj, tmp_path, ())
        _write_json(tmp_path / "manifest.json", manifest)
        _fsync_dir(tmp_path)
        _require_lock(has_lock, lock_path, "right before publishing final result")
        tmp_path.rename(path)
        _require_lock(has_lock, lock_path, "after publishing final result")
    except BaseException:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        raise


def load_result(path: Path) -> Any:
    return _decode(
        json.loads((path / "manifest.json").read_text(encoding="utf-8")), path
    )


def _encode(obj: Any, root: Path, path: tuple[str, ...]) -> JsonValue:
    match obj:
        case None | bool() | int() | float() | str():
            return obj
        case list():
            width = len(str(max(len(obj) - 1, 0)))
            return [
                _encode(x, root, (*path, f"arr_idx_{i:0{width}d}"))
                for i, x in enumerate(obj)
            ]
        case tuple():
            return {
                FURU_KEY: {
                    "kind": "tuple",
                    "items": [
                        _encode(x, root, (*path, str(i))) for i, x in enumerate(obj)
                    ],
                }
            }
        case dict():
            return {_valid_key(k): _encode(v, root, (*path, k)) for k, v in obj.items()}
        case PydanticBaseModel():
            return {
                FURU_KEY: {
                    "kind": "pydantic",
                    "type": fully_qualified_name(type(obj)),
                    "fields": {
                        _valid_key(k): _encode(v, root, (*path, k))
                        for k, v in obj.__dict__.items()
                    },
                }
            }
        case x if is_dataclass(x) and not isinstance(x, type):
            return {
                FURU_KEY: {
                    "kind": "dataclass",
                    "type": fully_qualified_name(type(x)),
                    "fields": {
                        _valid_key(f.name): _encode(
                            getattr(x, f.name), root, (*path, f.name)
                        )
                        for f in fields(x)
                    },
                }
            }
        case _ if _is_numpy_array(obj):
            artifact_path = Path("artifacts", *path)
            data_path = root / artifact_path / "data.npy"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            _numpy_save(data_path, obj)
            return {
                FURU_KEY: {
                    "kind": "external",
                    "codec": NUMPY_CODEC,
                    "path": artifact_path.as_posix(),
                    "meta": {"dtype": str(obj.dtype), "shape": list(obj.shape)},
                }
            }
        case _:
            raise TypeError(f"unsupported result value: {type(obj).__name__}")


def _decode(obj: JsonValue, root: Path) -> Any:
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_decode(x, root) for x in obj]
        return obj
    if set(obj) != {FURU_KEY}:
        return {_valid_key(k): _decode(v, root) for k, v in obj.items()}

    spec = obj[FURU_KEY]
    if not isinstance(spec, dict):
        raise ValueError("invalid furu result marker")
    match spec.get("kind"):
        case "tuple":
            return tuple(_decode(x, root) for x in _required_list(spec, "items"))
        case "external":
            if spec.get("codec") != NUMPY_CODEC:
                raise ValueError(f"unknown result codec: {spec.get('codec')}")
            return _numpy_load(root / str(spec["path"]) / "data.npy")
        case "pydantic":
            validate = getattr(_import_type(str(spec["type"])), "model_validate")
            return validate(
                {
                    k: _decode(v, root)
                    for k, v in _required_dict(spec, "fields").items()
                },
            )
        case "dataclass":
            return _import_type(str(spec["type"]))(
                **{
                    k: _decode(v, root)
                    for k, v in _required_dict(spec, "fields").items()
                }
            )
        case kind:
            raise ValueError(f"unknown furu result kind: {kind}")


def _valid_key(key: Any) -> str:
    if not isinstance(key, str) or key == "" or "/" in key or key == FURU_KEY:
        raise ValueError(f"invalid result key: {key!r}")
    return key


def _required_dict(spec: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"invalid furu result {key}")
    return value


def _required_list(spec: dict[str, JsonValue], key: str) -> list[JsonValue]:
    value = spec.get(key)
    if not isinstance(value, list):
        raise ValueError(f"invalid furu result {key}")
    return value


def _import_type(name: str) -> type:
    module_name, _, class_name = name.rpartition(".")
    if not module_name:
        raise ValueError(f"invalid type reference: {name}")
    return getattr(importlib.import_module(module_name), class_name)


def _is_numpy_array(obj: Any) -> bool:
    return type(obj).__module__ == "numpy" and type(obj).__name__ == "ndarray"


def _numpy_save(path: Path, obj: Any) -> None:
    import numpy as np

    with path.open("wb") as f:
        np.save(f, obj)
        f.flush()
        os.fsync(f.fileno())


def _numpy_load(path: Path) -> Any:
    import numpy as np

    return np.load(path, allow_pickle=False)


def _write_json(path: Path, obj: JsonValue) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_lock(has_lock: HasLock, lock_path: Path, when: str) -> None:
    if not has_lock():
        raise LockLostError(f"lost lock at {lock_path} {when}")
