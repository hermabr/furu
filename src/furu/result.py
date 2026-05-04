"""Stage 1 result persistence.

A persisted result is a *bundle directory* with this layout::

    bundle_dir/
        manifest.json
        artifacts/
            <mirrored logical path>/
                data.npy

``manifest.json`` directly contains the persisted root value. There is no
envelope. A result is considered complete when ``manifest.json`` exists.

Stage 1 supports JSON-native scalars/lists/dicts, dataclass instances,
Pydantic model instances, and an external NumPy codec.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, cast

import pydantic

from furu.utils import JsonValue, fully_qualified_name


# The reserved key used inside ``manifest.json`` to mark wrapper objects.
WRAPPER_KEY: Final[str] = "$furu"

# Subdirectory inside a bundle that holds external artifacts.
ARTIFACTS_DIR_NAME: Final[str] = "artifacts"

# Filename containing the manifest tree.
MANIFEST_FILE_NAME: Final[str] = "manifest.json"

# Artifact subdirectory used when the bundle root is a single external value.
_ROOT_ARTIFACT_NAME: Final[str] = "root"


# A logical path inside a result tree. Each element is the literal string
# used as a directory component under ``artifacts/``. List indices are
# formatted with zero-padded width based on the surrounding list length.
LogicalPath = tuple[str, ...]


def _path_display(path: LogicalPath) -> str:
    if not path:
        return "<root>"
    return "/".join(path)


def _path_artifact_dir(path: LogicalPath) -> Path:
    if not path:
        return Path(ARTIFACTS_DIR_NAME) / _ROOT_ARTIFACT_NAME
    return Path(ARTIFACTS_DIR_NAME, *path)


def _is_safe_path_segment(text: str) -> bool:
    """Stage 1 path safety rule for dict keys and field names.

    Keep this simple. Stage 1 does not support escaping, slugging, hashing,
    or recovery for unusual keys.
    """

    if text == "" or text == "." or text == "..":
        return False
    if "/" in text or "\\" in text:
        return False
    if "\x00" in text:
        return False
    return True


# ---------------------------------------------------------------------------
# Codec protocol and built-in codecs
# ---------------------------------------------------------------------------


class ResultCodec(Protocol):
    """An external result codec.

    A codec knows how to dump a value to a directory and load it back.
    Stage 1 has only built-in codecs; there is no public registration API.
    """

    codec_id: ClassVar[str]

    @classmethod
    def dependencies_available(cls) -> bool:
        """Whether this codec can run in the current environment."""
        ...

    @classmethod
    def matches(cls, value: object) -> bool:
        """Whether this codec should be used for ``value``."""
        ...

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        """Persist ``value`` under ``artifact_dir`` and return manifest meta."""
        ...

    @classmethod
    def load(cls, *, artifact_dir: Path, meta: JsonValue) -> object:
        """Reconstruct the value from ``artifact_dir`` and ``meta``."""
        ...


class NumpyNpyCodec:
    """Persist NumPy arrays as ``data.npy`` files."""

    codec_id: ClassVar[str] = "numpy.ndarray.npy"

    @classmethod
    def dependencies_available(cls) -> bool:
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def matches(cls, value: object) -> bool:
        import numpy as np

        return isinstance(value, np.ndarray)

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        import numpy as np

        assert isinstance(value, np.ndarray)
        if value.dtype.hasobject:
            raise ValueError(
                f"Unsupported result value at {_path_display(path)}:\n"
                "numpy object-dtype arrays are not supported by the default npy codec."
            )

        artifact_dir.mkdir(parents=True, exist_ok=False)
        np.save(artifact_dir / "data.npy", value, allow_pickle=False)

        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    @classmethod
    def load(cls, *, artifact_dir: Path, meta: JsonValue) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


def default_codecs() -> list[type[ResultCodec]]:
    """Return the default codec list."""

    return [NumpyNpyCodec]


def _codec_for_value(
    value: object,
    codecs: list[type[ResultCodec]],
) -> type[ResultCodec] | None:
    for codec in codecs:
        if codec.dependencies_available() and codec.matches(value):
            return codec
    return None


def _codec_by_id(
    codec_id: str,
    codecs: list[type[ResultCodec]],
) -> type[ResultCodec]:
    for codec in codecs:
        if codec.codec_id == codec_id:
            if not codec.dependencies_available():
                raise ValueError(
                    f"result codec {codec_id!r} is unavailable because its "
                    "dependencies are not installed"
                )
            return codec
    raise ValueError(f"unknown result codec: {codec_id}")


# ---------------------------------------------------------------------------
# Dump walker
# ---------------------------------------------------------------------------


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> JsonValue:
    # Order matters: bool is a subclass of int, so check it explicitly.
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value

    if isinstance(value, list):
        return _dump_list(
            cast(list[object], value),
            path=path,
            bundle_dir=bundle_dir,
            codecs=codecs,
        )

    if isinstance(value, dict):
        return _dump_dict(
            cast(dict[object, object], value),
            path=path,
            bundle_dir=bundle_dir,
            codecs=codecs,
        )

    # Pydantic and dataclass instances are *structural* containers. Pydantic
    # is checked first because a model can technically also be a dataclass
    # subclass in some configurations.
    if isinstance(value, pydantic.BaseModel):
        return _dump_pydantic(value, path=path, bundle_dir=bundle_dir, codecs=codecs)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dump_dataclass(value, path=path, bundle_dir=bundle_dir, codecs=codecs)

    codec = _codec_for_value(value, codecs)
    if codec is not None:
        return _dump_external(value, codec=codec, path=path, bundle_dir=bundle_dir)

    raise ValueError(
        f"Unsupported result value at {_path_display(path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by Furu Stage 1."
    )


def _dump_list(
    value: list[object],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> list[JsonValue]:
    width = max(len(str(len(value))), 1)
    return [
        _dump_value(
            item,
            path=(*path, f"{i:0{width}d}"),
            bundle_dir=bundle_dir,
            codecs=codecs,
        )
        for i, item in enumerate(value)
    ]


def _dump_dict(
    value: dict[object, object],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Unsupported result value at {_path_display(path)}:\n"
                f"dict result keys must be strings in Stage 1; got {type(key).__name__} key {key!r}."
            )
        if key == WRAPPER_KEY:
            raise ValueError(
                f"Unsupported result value at {_path_display(path)}:\n"
                f"dict keys named {WRAPPER_KEY!r} are reserved by Furu result persistence."
            )
        if not _is_safe_path_segment(key):
            raise ValueError(
                f"Unsupported result path at {_path_display((*path, key))}:\n"
                "dict key cannot be used as an artifact path segment."
            )
        out[key] = _dump_value(
            child,
            path=(*path, key),
            bundle_dir=bundle_dir,
            codecs=codecs,
        )
    return out


def _dump_dataclass(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> dict[str, JsonValue]:
    type_name = fully_qualified_name(type(value))
    fields_out: dict[str, JsonValue] = {}
    for field in dataclasses.fields(cast(Any, value)):
        if not _is_safe_path_segment(field.name):
            raise ValueError(
                f"Unsupported result path at {_path_display((*path, field.name))}:\n"
                "dataclass field name cannot be used as an artifact path segment."
            )
        fields_out[field.name] = _dump_value(
            getattr(value, field.name),
            path=(*path, field.name),
            bundle_dir=bundle_dir,
            codecs=codecs,
        )
    return {
        WRAPPER_KEY: {
            "kind": "dataclass",
            "type": type_name,
            "fields": fields_out,
        }
    }


def _dump_pydantic(
    value: pydantic.BaseModel,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> dict[str, JsonValue]:
    type_name = fully_qualified_name(type(value))
    fields_out: dict[str, JsonValue] = {}
    # Walk actual field values via ``getattr``, not ``model_dump()``, so that
    # non-JSON leaves (NumPy arrays) survive long enough for the result
    # walker to dispatch them to the right codec.
    for name in value.__class__.model_fields:
        if not _is_safe_path_segment(name):
            raise ValueError(
                f"Unsupported result path at {_path_display((*path, name))}:\n"
                "pydantic field name cannot be used as an artifact path segment."
            )
        fields_out[name] = _dump_value(
            getattr(value, name),
            path=(*path, name),
            bundle_dir=bundle_dir,
            codecs=codecs,
        )
    return {
        WRAPPER_KEY: {
            "kind": "pydantic",
            "type": type_name,
            "fields": fields_out,
        }
    }


def _dump_external(
    value: object,
    *,
    codec: type[ResultCodec],
    path: LogicalPath,
    bundle_dir: Path,
) -> dict[str, JsonValue]:
    artifact_rel = _path_artifact_dir(path)
    artifact_dir = bundle_dir / artifact_rel
    meta = codec.dump(value, artifact_dir=artifact_dir, path=path)
    return {
        WRAPPER_KEY: {
            "kind": "external",
            "codec": codec.codec_id,
            "path": artifact_rel.as_posix(),
            "meta": meta,
        }
    }


# ---------------------------------------------------------------------------
# Load walker
# ---------------------------------------------------------------------------


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> object:
    if node is None or isinstance(node, (bool, int, float, str)):
        return node

    if isinstance(node, list):
        return [
            _load_value(child, bundle_dir=bundle_dir, codecs=codecs) for child in node
        ]

    if isinstance(node, dict):
        if WRAPPER_KEY in node:
            return _load_wrapper(
                node[WRAPPER_KEY], bundle_dir=bundle_dir, codecs=codecs
            )
        return {
            key: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
            for key, child in node.items()
        }

    raise ValueError(f"unsupported manifest node: {type(node).__name__}")


def _load_wrapper(
    body: JsonValue,
    *,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> object:
    if not isinstance(body, dict):
        raise ValueError(
            f"malformed Furu wrapper: expected object, got {type(body).__name__}"
        )

    kind = body.get("kind")
    if kind == "external":
        return _load_external(body, bundle_dir=bundle_dir, codecs=codecs)
    if kind == "dataclass":
        return _load_dataclass(body, bundle_dir=bundle_dir, codecs=codecs)
    if kind == "pydantic":
        return _load_pydantic(body, bundle_dir=bundle_dir, codecs=codecs)
    raise ValueError(f"malformed Furu wrapper: unknown kind {kind!r}")


def _load_external(
    body: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> object:
    codec_id = body.get("codec")
    if not isinstance(codec_id, str):
        raise ValueError("malformed external wrapper: missing codec id")

    rel_path = body.get("path")
    if not isinstance(rel_path, str):
        raise ValueError("malformed external wrapper: missing path")

    artifact_rel = Path(rel_path)
    if artifact_rel.is_absolute():
        raise ValueError(f"external wrapper path must be relative: {rel_path}")

    artifact_dir = (bundle_dir / artifact_rel).resolve()
    artifacts_root = (bundle_dir / ARTIFACTS_DIR_NAME).resolve()
    try:
        artifact_dir.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"external wrapper path escapes bundle artifacts dir: {rel_path}"
        ) from exc

    if not artifact_dir.exists():
        raise ValueError(f"external wrapper artifact directory missing: {artifact_dir}")

    return _codec_by_id(codec_id, codecs).load(
        artifact_dir=artifact_dir, meta=body.get("meta")
    )


def _load_dataclass(
    body: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> object:
    type_name = body.get("type")
    if not isinstance(type_name, str):
        raise ValueError("malformed dataclass wrapper: missing type")
    fields_node = body.get("fields")
    if not isinstance(fields_node, dict):
        raise ValueError("malformed dataclass wrapper: missing fields")

    cls = _import_type(type_name)
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        raise ValueError(
            f"manifest declared dataclass type {type_name!r}, but it is not a dataclass"
        )

    loaded_fields = {
        name: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
        for name, child in fields_node.items()
    }

    obj = object.__new__(cls)
    for name, value in loaded_fields.items():
        # ``object.__setattr__`` works for both regular and frozen dataclasses
        # without rerunning ``__post_init__``.
        object.__setattr__(obj, name, value)
    return obj


def _load_pydantic(
    body: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    codecs: list[type[ResultCodec]],
) -> object:
    type_name = body.get("type")
    if not isinstance(type_name, str):
        raise ValueError("malformed pydantic wrapper: missing type")
    fields_node = body.get("fields")
    if not isinstance(fields_node, dict):
        raise ValueError("malformed pydantic wrapper: missing fields")

    cls = _import_type(type_name)
    if not (isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)):
        raise ValueError(
            f"manifest declared pydantic type {type_name!r}, but it is not a BaseModel subclass"
        )

    loaded_fields = {
        name: _load_value(child, bundle_dir=bundle_dir, codecs=codecs)
        for name, child in fields_node.items()
    }
    return cls.model_construct(**loaded_fields)


def _import_type(qualified_name: str) -> Any:
    if "." not in qualified_name:
        raise ValueError(f"cannot import unqualified type name: {qualified_name!r}")
    module_name, _, attr_name = qualified_name.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"cannot import {qualified_name!r}: attribute not found in module"
        ) from exc


# ---------------------------------------------------------------------------
# Bundle API
# ---------------------------------------------------------------------------


def save_result_bundle(value: object, bundle_dir: Path) -> None:
    """Persist ``value`` to a fresh bundle at ``bundle_dir``.

    The bundle directory must not yet exist. The manifest file is written
    last so that a partial directory never looks complete.
    """

    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)

    bundle_dir.mkdir(parents=True)
    (bundle_dir / ARTIFACTS_DIR_NAME).mkdir()

    codecs = default_codecs()

    manifest = _dump_value(
        value,
        path=(),
        bundle_dir=bundle_dir,
        codecs=codecs,
    )
    _write_manifest(bundle_dir / MANIFEST_FILE_NAME, manifest)


def load_result_bundle(bundle_dir: Path) -> object:
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    codecs = default_codecs()
    return _load_value(raw, bundle_dir=bundle_dir, codecs=codecs)


def result_bundle_is_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / MANIFEST_FILE_NAME).exists()


def _write_manifest(manifest_path: Path, manifest: JsonValue) -> None:
    """Write ``manifest`` to ``manifest_path`` with sync semantics.

    The encoding uses Python's default JSON behavior, which means
    ``NaN``/``Infinity``/``-Infinity`` are written verbatim and round-trip
    through ``json.loads``.
    """

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
