import datetime
import dataclasses
import contextlib
import enum
import hashlib
import importlib
import json
import pathlib
import textwrap
from pathlib import Path
from typing import Any, Protocol, Sequence, cast, runtime_checkable

import chz
from chz.util import MISSING as CHZ_MISSING, MISSING_TYPE

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

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            result = {cls.CLASS_MARKER: cls.get_classname(obj)}
            for field in dataclasses.fields(obj):
                result[field.name] = cls.to_dict(getattr(obj, field.name))
            return result

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
    def from_dict(cls, data: JsonValue, *, strict: bool = True) -> JsonValue:
        """Reconstruct object from dictionary.

        Args:
            data: Serialized payload.
            strict: Raise on reconstruction mismatch when True; return dict fallback
                when False.
        """
        if isinstance(data, dict) and cls.CLASS_MARKER in data:
            module_path, _, class_name = data[cls.CLASS_MARKER].rpartition(".")
            if strict:
                data_class = getattr(importlib.import_module(module_path), class_name)
            else:
                data_class = None
                with contextlib.suppress(Exception):
                    data_class = getattr(
                        importlib.import_module(module_path), class_name
                    )

                if data_class is None:
                    return {
                        k: cls.from_dict(v, strict=strict)
                        for k, v in data.items()
                        if k != cls.CLASS_MARKER
                    }

            kwargs = {
                k: cls.from_dict(v, strict=strict)
                for k, v in data.items()
                if k != cls.CLASS_MARKER
            }

            path_types = (Path, pathlib.Path)
            fallback = kwargs

            def _strict_or_fallback() -> JsonValue:
                if strict:
                    raise TypeError(f"cannot reconstruct {data_class}")
                return fallback

            if chz.is_chz(data_class):
                init_kwargs = dict(kwargs)
                for name, field in chz.chz_fields(data_class).items():
                    if name not in kwargs:
                        if field._default is CHZ_MISSING and isinstance(
                            field._default_factory, MISSING_TYPE
                        ):
                            return _strict_or_fallback()
                        continue

                    if field.final_type in path_types and isinstance(
                        init_kwargs.get(name), str
                    ):
                        init_kwargs[name] = pathlib.Path(init_kwargs[name])
            elif dataclasses.is_dataclass(data_class):
                dataclass_fields = dataclasses.fields(data_class)
                init_kwargs = {
                    field.name: kwargs[field.name]
                    for field in dataclass_fields
                    if field.init and field.name in kwargs
                }

                for field in dataclass_fields:
                    if not field.init:
                        continue
                    if field.name not in init_kwargs and (
                        field.default is dataclasses.MISSING
                        and field.default_factory is dataclasses.MISSING
                    ):
                        return _strict_or_fallback()

                for field in dataclass_fields:
                    if not field.init:
                        continue

                    field_value = init_kwargs.get(field.name)
                    if isinstance(field_value, str) and field.type in path_types:
                        init_kwargs[field.name] = pathlib.Path(field_value)
            else:
                init_kwargs = kwargs

            try:
                return data_class(**init_kwargs)
            except TypeError:
                if strict:
                    raise
                return fallback

        if isinstance(data, list):
            return [cls.from_dict(v, strict=strict) for v in data]

        if isinstance(data, dict):
            return {k: cls.from_dict(v, strict=strict) for k, v in data.items()}

        return data

    @classmethod
    def compute_hash(cls, obj: object, verbose: bool = False) -> str:
        """Compute deterministic hash of object."""

        @runtime_checkable
        class _DependencyHashProvider(Protocol):
            def _dependency_hashes(self) -> Sequence[str]: ...

        def _has_required_fields(
            data_class: type[object],
            data: dict[str, JsonValue],
        ) -> bool:
            if not chz.is_chz(data_class):
                return False
            for field in chz.chz_fields(data_class).values():
                name = field.logical_name
                if name in data:
                    continue
                if field._default is not CHZ_MISSING:
                    continue
                if not isinstance(field._default_factory, MISSING_TYPE):
                    continue
                return False
            return True

        def canonicalize(item: object) -> JsonValue:
            if isinstance(item, _FuruMissing):
                raise ValueError("Cannot hash Furu.MISSING")

            if chz.is_chz(item):
                fields = chz.chz_fields(item)
                result = {
                    "__class__": cls.get_classname(item),
                    **{
                        name: canonicalize(getattr(item, name))
                        for name in fields
                        if not name.startswith("_")
                    },
                }
                if isinstance(item, _DependencyHashProvider):
                    dependency_hashes = list(item._dependency_hashes())
                    if dependency_hashes:
                        result["__dependencies__"] = dependency_hashes
                return result

            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                return {
                    "__class__": cls.get_classname(item),
                    **{
                        field.name: canonicalize(getattr(item, field.name))
                        for field in dataclasses.fields(item)
                        if not field.name.startswith("_")
                    },
                }

            if isinstance(item, dict):
                mapping_item = cast(dict[str, JsonValue], item)
                class_marker = mapping_item.get(cls.CLASS_MARKER)
                if isinstance(class_marker, str):
                    config = mapping_item
                    module_path, _, class_name = class_marker.rpartition(".")
                    module = importlib.import_module(module_path)
                    data_class = getattr(module, class_name, None)
                    if (
                        data_class is not None
                        and hasattr(data_class, "_dependency_hashes")
                        and _has_required_fields(data_class, config)
                    ):
                        return canonicalize(cls.from_dict(config))
                filtered = mapping_item
                if class_marker is not None:
                    filtered = {
                        k: v
                        for k, v in mapping_item.items()
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
