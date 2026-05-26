import hashlib
import json
import os
import socket
import sys
import uuid
from enum import Enum
from pathlib import Path

from pydantic import JsonValue

type JsonFields = dict[str, JsonValue]


def class_label(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _module_name_from_src_path(path: Path) -> tuple[str, Path] | None:
    for parent in path.parents:
        if parent.name != "src":
            continue

        parts = list(path.relative_to(parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts and all(part.isidentifier() for part in parts):
            return ".".join(parts), parent
    return None


def _module_name_from_package_path(path: Path) -> tuple[str, Path] | None:
    if path.suffix != ".py":
        return None

    parts = [] if path.name == "__init__.py" else [path.stem]
    current = path.parent
    while (current / "__init__.py").is_file():
        parts.insert(0, current.name)
        current = current.parent

    if parts:
        return ".".join(parts), current
    return _module_name_from_src_path(path)


def _main_module_name_for(tp: type) -> str:
    main_module = sys.modules.get("__main__")
    if main_module is None or getattr(main_module, tp.__qualname__, None) is not tp:
        raise ValueError("Cannot serialize objects from __main__ module")

    spec = getattr(main_module, "__spec__", None)
    spec_name = getattr(spec, "name", None)
    if isinstance(spec_name, str) and spec_name != "__main__":
        sys.modules.setdefault(spec_name, main_module)
        return spec_name

    main_file = getattr(main_module, "__file__", None)
    if main_file is None:
        raise ValueError("Cannot serialize objects from __main__ module")

    resolved = _module_name_from_package_path(Path(main_file).resolve())
    if resolved is None:
        raise ValueError("Cannot serialize objects from __main__ module")

    module_name, import_root = resolved
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
    sys.modules.setdefault(module_name, main_module)
    return module_name


def fully_qualified_name(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if "<locals>" in qualname:  # TODO: allow overwriting
        raise ValueError("TODO: msg")
    elif mod == "__main__":
        mod = _main_module_name_for(tp)
    elif "." in qualname:
        raise ValueError("TODO: msg")
    elif (isinstance(tp, type) and issubclass(tp, Enum)) or isinstance(tp, Enum):
        raise ValueError(
            "TODO: support this in the future"
        )  # return f"{mod}.{qualname}.{obj.name}"
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
