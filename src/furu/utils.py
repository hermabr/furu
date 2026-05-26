import hashlib
import json
import keyword
import os
import socket
import sys
import uuid
from enum import Enum
from functools import cache
from pathlib import Path
from types import ModuleType

from pydantic import JsonValue

type JsonFields = dict[str, JsonValue]


def class_label(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _valid_module_parts(parts: list[str]) -> bool:
    return all(part.isidentifier() and not keyword.iskeyword(part) for part in parts)


def _module_name_from_package_path(path: Path) -> tuple[str, Path] | None:
    module_parts = [] if path.name == "__init__.py" else [path.stem]
    package_dir = path.parent

    current = package_dir
    while (current / "__init__.py").is_file():
        module_parts.insert(0, current.name)
        current = current.parent

    if not module_parts or not _valid_module_parts(module_parts):
        return None
    if current == package_dir:
        return None
    return ".".join(module_parts), current


def _module_name_from_src_path(path: Path) -> tuple[str, Path] | None:
    for root in path.parents:
        if root.name != "src":
            continue
        relative = path.relative_to(root)
        parts = list(relative.with_suffix("").parts)
        if parts[-1:] == ["__init__"]:
            parts = parts[:-1]
        if parts and _valid_module_parts(parts):
            return ".".join(parts), root
    return None


def _module_name_from_import_path(path: Path) -> tuple[str, Path] | None:
    roots = {Path.cwd().resolve()}
    roots.update(
        Path(entry or os.getcwd()).resolve()
        for entry in sys.path
        if isinstance(entry, str)
    )

    candidates: list[tuple[str, Path]] = []
    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1:] == ["__init__"]:
            parts = parts[:-1]
        if parts and _valid_module_parts(parts):
            candidates.append((".".join(parts), root))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: len(candidate[0].split(".")))


def _insert_import_root(root: Path) -> None:
    root_str = str(root)
    resolved_roots = {
        str(Path(entry or os.getcwd()).resolve())
        for entry in sys.path
        if isinstance(entry, str)
    }
    if str(root.resolve()) not in resolved_roots:
        sys.path.insert(0, root_str)


def _alias_main_module(module_name: str) -> None:
    main_module = sys.modules.get("__main__")
    if not isinstance(main_module, ModuleType):
        return
    sys.modules.setdefault(module_name, main_module)


@cache
def _main_module_name() -> str:
    main_module = sys.modules.get("__main__")
    if not isinstance(main_module, ModuleType):
        raise ValueError("Cannot serialize objects from __main__ module")

    spec = getattr(main_module, "__spec__", None)
    spec_name = getattr(spec, "name", None)
    if isinstance(spec_name, str) and spec_name != "__main__":
        _alias_main_module(spec_name)
        return spec_name

    main_file = getattr(main_module, "__file__", None)
    if not isinstance(main_file, str):
        raise ValueError("Cannot serialize objects from __main__ module")

    path = Path(main_file).resolve()
    for resolver in (
        _module_name_from_package_path,
        _module_name_from_src_path,
        _module_name_from_import_path,
    ):
        if resolved := resolver(path):
            module_name, import_root = resolved
            _insert_import_root(import_root)
            _alias_main_module(module_name)
            return module_name

    raise ValueError(
        "Cannot serialize objects from __main__ module; run the file from an "
        "importable package or a src/ layout"
    )


def fully_qualified_name(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if mod == "__main__":
        mod = _main_module_name()
    if "<locals>" in qualname:  # TODO: allow overwriting
        raise ValueError("TODO: msg")
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
