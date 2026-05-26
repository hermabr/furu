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


def fully_qualified_name(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if "<locals>" in qualname:  # TODO: allow overwriting
        raise ValueError("TODO: msg")
    elif "." in qualname:
        raise ValueError("TODO: msg")
    elif mod == "__main__":
        main_module = sys.modules.get("__main__")
        if main_module is None:
            raise ValueError("Cannot resolve __main__ module name: __main__ is missing")
        spec_name = getattr(getattr(main_module, "__spec__", None), "name", None)
        if isinstance(spec_name, str) and spec_name != "__main__":
            mod = spec_name
        else:
            main_file = getattr(main_module, "__file__", None)
            if main_file is None:
                raise ValueError(
                    "Cannot resolve __main__ module name: __main__.__file__ is not set"
                )
            try:
                cwd = Path.cwd().resolve()
                path = Path(main_file).resolve().relative_to(cwd)
            except ValueError:
                raise ValueError(
                    f"Cannot resolve __main__ module name: {main_file!r} is outside "
                    f"the current working directory {cwd}"
                ) from None
            parts = path.with_suffix("").parts
            if parts[:1] == ("src",):
                parts = parts[1:]
            if not parts or not all(part.isidentifier() for part in parts):
                raise ValueError(
                    f"Cannot resolve __main__ module name: {path} is not a valid module path"
                )
            mod = ".".join(parts)
        sys.modules[mod] = main_module
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
