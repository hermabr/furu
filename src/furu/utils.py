import hashlib
import importlib
import json
import os
import socket
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import JsonValue

type JsonFields = dict[str, JsonValue]


def class_label(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def fully_qualified_name(value: object) -> Any:  # TODO: clean up this method
    main = sys.modules.get("__main__")
    main_module_name = None
    if main is not None:
        spec = getattr(main, "__spec__", None)
        spec_name = getattr(spec, "name", None)
        if isinstance(spec_name, str) and spec_name and spec_name != "__main__":
            main_module_name = spec_name

    if isinstance(value, str):
        module_name, _, attr_name = value.rpartition(".")
        if not module_name or not attr_name:
            raise ValueError(f"Expected fully qualified name, got {value!r}")

        if module_name == main_module_name:
            if main is not None and hasattr(main, attr_name):
                return getattr(main, attr_name)

        return getattr(importlib.import_module(module_name), attr_name)

    mod = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(mod, str) or not isinstance(qualname, str):
        raise TypeError(
            f"Expected a fully qualified name or nameable object, got {value!r}"
        )

    if "<locals>" in qualname:
        raise ValueError("Cannot serialize local classes")
    elif "." in qualname:
        raise ValueError("Cannot serialize nested classes")
    elif isinstance(value, type) and issubclass(value, Enum):
        raise ValueError(
            "TODO: support this in the future"
        )  # return f"{mod}.{qualname}.{obj.name}"

    if mod == "__main__":
        if main_module_name is None:
            raise ValueError(
                "Cannot serialize objects from __main__ module. "
                "Run the file as `python -m package.module`."
            )
        mod = main_module_name

    return f"{mod}.{qualname}"


def object_id_from_parts(
    *,
    fully_qualified_name: str,
    schema_hash: str,
    artifact_hash: str,
) -> str:
    return f"{fully_qualified_name}:{schema_hash}:{artifact_hash}"


def _stable_json_dump(x: JsonValue) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _hash_dict_deterministically(obj: JsonValue) -> str:
    json_str = _stable_json_dump(obj)

    return hashlib.blake2s(
        json_str.encode(),
        digest_size=10,  # TODO: make this digest size configurable and include a script for estimating likelihood of crashing. right now, i think there is a 1e-08 chance of a collision with 155M items with the same schema and namespace
    ).hexdigest()


def nfs_safe_unique_name(path: Path, *, name: str | None = None) -> Path:
    stem = f"{path.name}.{socket.getfqdn()}.{os.getpid()}.{uuid.uuid4().hex}"
    if name is not None:
        stem = f"{stem}.{name}"
    return path.with_name(stem)


def write_private_file(path: Path, contents: str, *, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(contents)
    path.chmod(mode)
