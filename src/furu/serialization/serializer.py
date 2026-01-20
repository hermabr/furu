import datetime
import enum
import hashlib
import importlib
import json
import pathlib
import textwrap
from pathlib import Path
from typing import Any

import chz

from ..errors import _FuruMissing
from pydantic import BaseModel as PydanticBaseModel


# Type alias for JSON-serializable values. We use Any here because this serialization
# library handles arbitrary user-defined objects that we cannot know at compile time.
JsonValue = Any


class FuruSerializer:
    """Handles serialization, deserialization, and hashing of Furu objects."""

    CLASS_MARKER = "__class__"

    @staticmethod
    def get_classname(obj: object) -> str:
        """Get fully qualified class name."""
        classname = obj.__class__.__module__
        if classname == "__main__":
            raise ValueError("Cannot serialize objects from __main__ module")

        if isinstance(obj, enum.Enum):
            return f"{classname}.{obj.__class__.__qualname__}:{obj.name}"
        return f"{classname}.{obj.__class__.__qualname__}"

    @classmethod
    def to_dict(cls, obj: object) -> JsonValue:
        """Convert object to JSON-serializable dictionary."""
        if isinstance(obj, _FuruMissing):
            raise ValueError("Cannot serialize Furu.MISSING")

        if chz.is_chz(obj):
            result = {cls.CLASS_MARKER: cls.get_classname(obj)}
            for field_name in chz.chz_fields(obj):
                result[field_name] = cls.to_dict(getattr(obj, field_name))
            return result

        if isinstance(obj, pathlib.Path):
            return str(obj)

        if isinstance(obj, (list, tuple)):
            return [cls.to_dict(v) for v in obj]

        if isinstance(obj, dict):
            return {k: cls.to_dict(v) for k, v in obj.items()}

        return obj

    @classmethod
    def from_dict(cls, data: JsonValue) -> JsonValue:
        """Reconstruct object from dictionary."""
        if isinstance(data, dict) and cls.CLASS_MARKER in data:
            module_path, _, class_name = data[cls.CLASS_MARKER].rpartition(".")
            data_class = getattr(importlib.import_module(module_path), class_name)

            kwargs = {
                k: cls.from_dict(v) for k, v in data.items() if k != cls.CLASS_MARKER
            }

            path_types = (Path, pathlib.Path)

            if chz.is_chz(data_class):
                for name, field in chz.chz_fields(data_class).items():
                    if field.final_type in path_types and isinstance(
                        kwargs.get(name), str
                    ):
                        kwargs[name] = pathlib.Path(kwargs[name])
            return data_class(**kwargs)

        if isinstance(data, list):
            return [cls.from_dict(v) for v in data]

        if isinstance(data, dict):
            return {k: cls.from_dict(v) for k, v in data.items()}

        return data

    @classmethod
    def compute_hash(cls, obj: object, verbose: bool = False) -> str:
        """Compute deterministic hash of object."""

        def canonicalize(item: object) -> JsonValue:
            if isinstance(item, _FuruMissing):
                raise ValueError("Cannot hash Furu.MISSING")

            if chz.is_chz(item):
                fields = chz.chz_fields(item)
                return {
                    "__class__": cls.get_classname(item),
                    **{
                        name: canonicalize(getattr(item, name))
                        for name in fields
                        if not name.startswith("_")
                    },
                }

            if isinstance(item, dict):
                filtered = item
                if cls.CLASS_MARKER in item:
                    filtered = {
                        k: v
                        for k, v in item.items()
                        if not (isinstance(k, str) and k.startswith("_"))
                        or k == cls.CLASS_MARKER
                    }
                return {k: canonicalize(v) for k, v in sorted(filtered.items())}

            if isinstance(item, (list, tuple)):
                return [canonicalize(v) for v in item]

            if isinstance(item, Path):
                return str(item)

            if isinstance(item, enum.Enum):
                return {"__enum__": cls.get_classname(item)}

            if isinstance(item, (set, frozenset)):
                return sorted(canonicalize(v) for v in item)

            if isinstance(item, (bytes, bytearray, memoryview)):
                return {"__bytes__": hashlib.sha256(item).hexdigest()}

            if isinstance(item, datetime.datetime):
                return item.astimezone(datetime.timezone.utc).isoformat(
                    timespec="microseconds"
                )

            if isinstance(item, (str, int, float, bool)) or item is None:
                return item

            if isinstance(item, PydanticBaseModel):
                return {
                    "__class__": cls.get_classname(item),
                    **{k: canonicalize(v) for k, v in item.model_dump().items()},
                }

            raise TypeError(f"Cannot hash type: {type(item)}")

        canonical = canonicalize(obj)
        json_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))

        if verbose:
            print(json_str)

        return hashlib.blake2s(json_str.encode(), digest_size=10).hexdigest()

    @classmethod
    def to_python(cls, obj: object, multiline: bool = True) -> str:
        """Convert object to Python code representation."""

        def to_py_recursive(item: object, indent: int = 0) -> str:
            if isinstance(item, _FuruMissing):
                raise ValueError("Cannot convert Furu.MISSING to Python")

            pad = "" if not multiline else " " * indent
            next_indent = indent + (4 if multiline else 0)

            if chz.is_chz(item):
                cls_path = cls.get_classname(item)
                fields = []
                for name, field in chz.chz_fields(item).items():
                    fields.append(
                        f"{name}={to_py_recursive(getattr(item, name), next_indent)}"
                    )

                if multiline:
                    inner = (",\n" + " " * next_indent).join(fields)
                    return f"{cls_path}(\n{pad}    {inner}\n{pad})"
                return f"{cls_path}({', '.join(fields)})"

            if isinstance(item, enum.Enum):
                return cls.get_classname(item)

            if isinstance(item, pathlib.Path):
                return f"pathlib.Path({str(item)!r})"

            if isinstance(item, datetime.datetime):
                iso = item.astimezone(datetime.timezone.utc).isoformat(
                    timespec="microseconds"
                )
                return f"datetime.datetime.fromisoformat({iso!r})"

            if isinstance(item, (bytes, bytearray, memoryview)):
                hex_str = hashlib.sha256(item).hexdigest()
                return f"bytes.fromhex({hex_str!r})"

            if isinstance(item, list):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                return f"[{items}]"

            if isinstance(item, tuple):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                comma = "," if len(item) == 1 else ""
                return f"({items}{comma})"

            if isinstance(item, set):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                return f"{{{items}}}"

            if isinstance(item, frozenset):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                return f"frozenset({{{items}}})"

            if isinstance(item, dict):
                kv_pairs = [
                    f"{to_py_recursive(k, next_indent)}: {to_py_recursive(v, next_indent)}"
                    for k, v in item.items()
                ]

                if multiline:
                    joined = (",\n" + " " * (indent + 4)).join(kv_pairs)
                    return f"{{\n{pad}    {joined}\n{pad}}}"
                else:
                    return "{" + ", ".join(kv_pairs) + "}"

            return repr(item)

        result = to_py_recursive(obj, indent=0)
        if multiline:
            result = textwrap.dedent(result).strip()
        return result
