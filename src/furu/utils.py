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

_MAIN_MODULE_OVERRIDE: str | None = None


def class_label(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def set_main_module(module_name: str) -> None:
    if (
        not module_name
        or module_name == "__main__"
        or not all(part.isidentifier() for part in module_name.split("."))
    ):
        raise ValueError(f"Expected a valid module name, got {module_name!r}")

    global _MAIN_MODULE_OVERRIDE
    _MAIN_MODULE_OVERRIDE = module_name


def _module_name_from_package_file(path: Path) -> str | None:
    path = path.resolve()

    if path.suffix != ".py":
        return None

    if path.name == "__init__.py":
        parts: list[str] = []
    else:
        parts = [path.stem]

    package_dir = path.parent
    found_package = False

    while (package_dir / "__init__.py").is_file():
        found_package = True
        parts.insert(0, package_dir.name)
        package_dir = package_dir.parent

    if not found_package or not parts:
        return None

    if not all(part.isidentifier() for part in parts):
        return None

    return ".".join(parts)


def _running_main_module_name() -> str | None:
    if _MAIN_MODULE_OVERRIDE is not None:
        return _MAIN_MODULE_OVERRIDE

    main = sys.modules.get("__main__")
    if main is None:
        return None

    spec = getattr(main, "__spec__", None)
    spec_name = getattr(spec, "name", None)
    if isinstance(spec_name, str) and spec_name and spec_name != "__main__":
        return spec_name

    main_file = getattr(main, "__file__", None)
    if isinstance(main_file, str):
        return _module_name_from_package_file(Path(main_file))

    return None


def fully_qualified_name(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if "<locals>" in qualname:
        raise ValueError("Cannot serialize local classes")
    elif "." in qualname:
        raise ValueError("Cannot serialize nested classes")
    elif (isinstance(tp, type) and issubclass(tp, Enum)) or isinstance(tp, Enum):
        raise ValueError(
            "TODO: support this in the future"
        )  # return f"{mod}.{qualname}.{obj.name}"
    elif mod == "__main__":
        main_module_name = _running_main_module_name()
        if main_module_name is None:
            raise ValueError(
                "Cannot serialize objects from __main__ module. "
                "Run the file as `python -m package.module`, put it in a regular "
                "package, or call `furu.set_main_module(...)`."
            )
        mod = main_module_name
    return f"{mod}.{qualname}"


def resolve_qualified_name(qualified_name: str) -> Any:
    module_name, _, attr_name = qualified_name.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Expected fully qualified name, got {qualified_name!r}")

    if module_name == _running_main_module_name():
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, attr_name):
            return getattr(main, attr_name)

    return getattr(importlib.import_module(module_name), attr_name)


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
