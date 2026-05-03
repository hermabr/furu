"""Stage 1 result persistence.

This module is the single home for Furu's new on-disk result format.

A persisted result is a *bundle directory* with this layout::

    bundle_dir/
        manifest.json
        artifacts/
            <mirrored logical path>/
                data.npy
                data.parquet

`manifest.json` directly contains the persisted root value. There is no
envelope. A result is considered complete when `manifest.json` exists.

Stage 1 supports JSON-native scalars/lists/dicts, dataclass instances,
Pydantic model instances, plus optional external codecs for NumPy arrays
and Polars DataFrames.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, cast

import pydantic

from furu.utils import JsonValue, fully_qualified_name


# The reserved key used inside ``manifest.json`` to mark wrapper objects.
WRAPPER_KEY: Final[str] = "$furu"

# Subdirectory inside a bundle that holds external artifacts.
ARTIFACTS_DIR_NAME: Final[str] = "artifacts"

# Filename containing the manifest tree.
MANIFEST_FILE_NAME: Final[str] = "manifest.json"

# Logical path string used at the bundle root.
_ROOT_DISPLAY: Final[str] = "$"

# Artifact subdirectory used when the bundle root is a single external value.
_ROOT_ARTIFACT_NAME: Final[str] = "root"


# ---------------------------------------------------------------------------
# Logical paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KeySegment:
    key: str


@dataclass(frozen=True)
class _IndexSegment:
    index: int
    width: int


@dataclass(frozen=True)
class _FieldSegment:
    name: str


_Segment = _KeySegment | _IndexSegment | _FieldSegment


@dataclass(frozen=True)
class LogicalPath:
    """A small immutable path describing a position inside a result tree.

    Used for both the human-readable display string in error messages and
    the on-disk artifact directory under ``bundle_dir/artifacts/``.
    """

    parts: tuple[_Segment, ...] = ()

    def key(self, key: str) -> "LogicalPath":
        return LogicalPath(self.parts + (_KeySegment(key),))

    def index(self, index: int, *, width: int) -> "LogicalPath":
        return LogicalPath(self.parts + (_IndexSegment(index, width),))

    def field(self, name: str) -> "LogicalPath":
        return LogicalPath(self.parts + (_FieldSegment(name),))

    def display(self) -> str:
        if not self.parts:
            return _ROOT_DISPLAY

        chunks: list[str] = [_ROOT_DISPLAY]
        for part in self.parts:
            match part:
                case _KeySegment(key):
                    if _is_safe_path_segment(key) and _is_simple_identifier(key):
                        chunks.append(f".{key}")
                    else:
                        chunks.append(f"[{json.dumps(key)}]")
                case _IndexSegment(index, width):
                    chunks.append(f"[{index:0{width}d}]")
                case _FieldSegment(name):
                    chunks.append(f".{name}")
        return "".join(chunks)

    def artifact_dir(self) -> Path:
        if not self.parts:
            return Path(ARTIFACTS_DIR_NAME) / _ROOT_ARTIFACT_NAME

        segments: list[str] = [ARTIFACTS_DIR_NAME]
        for part in self.parts:
            match part:
                case _KeySegment(key):
                    segments.append(key)
                case _IndexSegment(index, width):
                    segments.append(f"{index:0{width}d}")
                case _FieldSegment(name):
                    segments.append(name)
        return Path(*segments)


def _is_simple_identifier(text: str) -> bool:
    """Whether a key can be displayed as ``$.foo`` rather than ``$["foo"]``."""

    return bool(text) and text.isidentifier()


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
# Codec protocol and registry
# ---------------------------------------------------------------------------


class ResultCodec(ABC):
    """A built-in external result codec.

    A codec knows how to dump a value to a directory and load it back.
    Stage 1 has only built-in codecs; there is no public registration API.
    """

    codec_id: ClassVar[str]

    @abstractmethod
    def matches(self, value: object) -> bool:
        """Whether this codec should be used for ``value``."""

    @abstractmethod
    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        """Persist ``value`` under ``artifact_dir`` and return manifest meta."""

    @abstractmethod
    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        """Reconstruct the value from ``artifact_dir`` and ``meta``."""


class ResultRegistry:
    """Internal registry mapping values and codec IDs to ``ResultCodec`` instances."""

    def __init__(self) -> None:
        self._codecs: list[ResultCodec] = []
        self._by_id: dict[str, ResultCodec] = {}

    def register(self, codec: ResultCodec) -> None:
        if codec.codec_id in self._by_id:
            raise ValueError(f"codec already registered: {codec.codec_id}")
        self._codecs.append(codec)
        self._by_id[codec.codec_id] = codec

    def codec_for_value(self, value: object) -> ResultCodec | None:
        for codec in self._codecs:
            if codec.matches(value):
                return codec
        return None

    def codec_by_id(self, codec_id: str) -> ResultCodec:
        try:
            return self._by_id[codec_id]
        except KeyError:
            raise ValueError(f"unknown result codec: {codec_id}") from None


# ---------------------------------------------------------------------------
# Built-in codecs
# ---------------------------------------------------------------------------


class NumpyNpyCodec(ResultCodec):
    """Persist NumPy arrays as ``data.npy`` files."""

    codec_id: ClassVar[str] = "numpy.ndarray.npy"

    def __init__(self) -> None:
        # Importing here so the codec class can be defined unconditionally
        # but ``default_result_registry()`` skips registration when NumPy
        # is unavailable.
        import numpy as np  # noqa: F401  (import-time dependency check)

    def matches(self, value: object) -> bool:
        try:
            import numpy as np
        except ImportError:
            return False
        return isinstance(value, np.ndarray)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        import numpy as np

        assert isinstance(value, np.ndarray)
        if value.dtype.hasobject:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                "numpy object-dtype arrays are not supported by the default npy codec."
            )

        artifact_dir.mkdir(parents=True, exist_ok=False)
        np.save(artifact_dir / "data.npy", value, allow_pickle=False)

        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


class PolarsParquetCodec(ResultCodec):
    """Persist Polars DataFrames as ``data.parquet`` files."""

    codec_id: ClassVar[str] = "polars.dataframe.parquet"

    def __init__(self) -> None:
        import polars as pl  # noqa: F401

    def matches(self, value: object) -> bool:
        try:
            import polars as pl
        except ImportError:
            return False
        # ``LazyFrame`` is intentionally not supported in Stage 1.
        return isinstance(value, pl.DataFrame)

    def dump(
        self,
        value: object,
        *,
        artifact_dir: Path,
        path: LogicalPath,
    ) -> JsonValue:
        import polars as pl

        assert isinstance(value, pl.DataFrame)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        value.write_parquet(artifact_dir / "data.parquet")

        return {
            "rows": value.height,
            "columns": list(value.columns),
        }

    def load(self, *, artifact_dir: Path, meta: JsonValue) -> object:
        import polars as pl

        return pl.read_parquet(artifact_dir / "data.parquet")


def default_result_registry() -> ResultRegistry:
    """Build the default registry, skipping codecs whose libraries are missing."""

    registry = ResultRegistry()

    try:
        registry.register(NumpyNpyCodec())
    except ImportError:
        pass

    try:
        registry.register(PolarsParquetCodec())
    except ImportError:
        pass

    return registry


# ---------------------------------------------------------------------------
# Dump walker
# ---------------------------------------------------------------------------


def dump_result_tree(
    value: object,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> JsonValue:
    """Walk ``value`` and emit the manifest tree, writing artifacts as needed."""

    return _dump_value(
        value,
        path=LogicalPath(),
        bundle_dir=bundle_dir,
        registry=registry,
    )


def _dump_value(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
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
            registry=registry,
        )

    if isinstance(value, dict):
        return _dump_dict(
            cast(dict[object, object], value),
            path=path,
            bundle_dir=bundle_dir,
            registry=registry,
        )

    # Pydantic and dataclass instances are *structural* containers. Pydantic
    # is checked first because a model can technically also be a dataclass
    # subclass in some configurations.
    if isinstance(value, pydantic.BaseModel):
        return _dump_pydantic(
            value, path=path, bundle_dir=bundle_dir, registry=registry
        )

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dump_dataclass(
            value, path=path, bundle_dir=bundle_dir, registry=registry
        )

    codec = registry.codec_for_value(value)
    if codec is not None:
        return _dump_external(value, codec=codec, path=path, bundle_dir=bundle_dir)

    raise ValueError(
        f"Unsupported result value at {path.display()}:\n"
        f"values of type {type(value).__name__!r} are not supported by Furu Stage 1."
    )


def _dump_list(
    value: list[object],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> list[JsonValue]:
    width = max(len(str(len(value))), 1)
    return [
        _dump_value(
            item,
            path=path.index(i, width=width),
            bundle_dir=bundle_dir,
            registry=registry,
        )
        for i, item in enumerate(value)
    ]


def _dump_dict(
    value: dict[object, object],
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                f"dict result keys must be strings in Stage 1; got {type(key).__name__} key {key!r}."
            )
        if key == WRAPPER_KEY:
            raise ValueError(
                f"Unsupported result value at {path.display()}:\n"
                f"dict keys named {WRAPPER_KEY!r} are reserved by Furu result persistence."
            )
        child_path = path.key(key)
        if not _is_safe_path_segment(key):
            raise ValueError(
                f"Unsupported result path at {child_path.display()}:\n"
                "dict key cannot be used as an artifact path segment."
            )
        out[key] = _dump_value(
            child,
            path=child_path,
            bundle_dir=bundle_dir,
            registry=registry,
        )
    return out


def _dump_dataclass(
    value: object,
    *,
    path: LogicalPath,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> dict[str, JsonValue]:
    type_name = fully_qualified_name(type(value))
    fields_out: dict[str, JsonValue] = {}
    for field in dataclasses.fields(cast(Any, value)):
        if not _is_safe_path_segment(field.name):
            raise ValueError(
                f"Unsupported result path at {path.field(field.name).display()}:\n"
                "dataclass field name cannot be used as an artifact path segment."
            )
        fields_out[field.name] = _dump_value(
            getattr(value, field.name),
            path=path.field(field.name),
            bundle_dir=bundle_dir,
            registry=registry,
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
    registry: ResultRegistry,
) -> dict[str, JsonValue]:
    type_name = fully_qualified_name(type(value))
    fields_out: dict[str, JsonValue] = {}
    # Walk actual field values via ``getattr``, not ``model_dump()``, so that
    # non-JSON leaves (NumPy arrays, Polars frames) survive long enough for
    # the result walker to dispatch them to the right codec.
    for name in value.__class__.model_fields:
        if not _is_safe_path_segment(name):
            raise ValueError(
                f"Unsupported result path at {path.field(name).display()}:\n"
                "pydantic field name cannot be used as an artifact path segment."
            )
        fields_out[name] = _dump_value(
            getattr(value, name),
            path=path.field(name),
            bundle_dir=bundle_dir,
            registry=registry,
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
    codec: ResultCodec,
    path: LogicalPath,
    bundle_dir: Path,
) -> dict[str, JsonValue]:
    artifact_rel = path.artifact_dir()
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


def load_result_tree(
    node: JsonValue,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    """Reconstruct a Python value from a manifest JSON tree."""

    return _load_value(node, bundle_dir=bundle_dir, registry=registry)


def _load_value(
    node: JsonValue,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if node is None or isinstance(node, (bool, int, float, str)):
        return node

    if isinstance(node, list):
        return [
            _load_value(child, bundle_dir=bundle_dir, registry=registry)
            for child in node
        ]

    if isinstance(node, dict):
        if WRAPPER_KEY in node:
            return _load_wrapper(
                node[WRAPPER_KEY], bundle_dir=bundle_dir, registry=registry
            )
        return {
            key: _load_value(child, bundle_dir=bundle_dir, registry=registry)
            for key, child in node.items()
        }

    raise ValueError(f"unsupported manifest node: {type(node).__name__}")


def _load_wrapper(
    body: JsonValue,
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
) -> object:
    if not isinstance(body, dict):
        raise ValueError(
            f"malformed Furu wrapper: expected object, got {type(body).__name__}"
        )

    kind = body.get("kind")
    if kind == "external":
        return _load_external(body, bundle_dir=bundle_dir, registry=registry)
    if kind == "dataclass":
        return _load_dataclass(body, bundle_dir=bundle_dir, registry=registry)
    if kind == "pydantic":
        return _load_pydantic(body, bundle_dir=bundle_dir, registry=registry)
    raise ValueError(f"malformed Furu wrapper: unknown kind {kind!r}")


def _load_external(
    body: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
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
    bundle_root = bundle_dir.resolve()
    artifacts_root = (bundle_dir / ARTIFACTS_DIR_NAME).resolve()
    try:
        artifact_dir.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"external wrapper path escapes bundle artifacts dir: {rel_path}"
        ) from exc
    # Defense in depth - make sure we never read above the bundle either.
    artifact_dir.relative_to(bundle_root)

    if not artifact_dir.exists():
        raise ValueError(f"external wrapper artifact directory missing: {artifact_dir}")

    codec = registry.codec_by_id(codec_id)
    return codec.load(artifact_dir=artifact_dir, meta=body.get("meta"))


def _load_dataclass(
    body: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    registry: ResultRegistry,
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
        name: _load_value(child, bundle_dir=bundle_dir, registry=registry)
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
    registry: ResultRegistry,
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
        name: _load_value(child, bundle_dir=bundle_dir, registry=registry)
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

    registry = default_result_registry()

    try:
        manifest = dump_result_tree(
            value,
            bundle_dir=bundle_dir,
            registry=registry,
        )
        _write_manifest(bundle_dir / MANIFEST_FILE_NAME, manifest)
    except BaseException:
        # If anything goes wrong we leave no half-written manifest behind.
        # The caller is responsible for removing the partial directory.
        raise


def load_result_bundle(bundle_dir: Path) -> object:
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = default_result_registry()
    return load_result_tree(raw, bundle_dir=bundle_dir, registry=registry)


def result_bundle_is_complete(bundle_dir: Path) -> bool:
    return (bundle_dir / MANIFEST_FILE_NAME).exists()


def _write_manifest(manifest_path: Path, manifest: JsonValue) -> None:
    """Write ``manifest`` to ``manifest_path`` with sync semantics.

    The encoding uses Python's default JSON behavior, which means
    ``NaN``/``Infinity``/``-Infinity`` are written verbatim and round-trip
    through ``json.loads``.
    """

    text = json.dumps(manifest, indent=2, sort_keys=True)
    fd = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
