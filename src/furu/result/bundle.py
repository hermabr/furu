from __future__ import annotations

import contextlib
import dataclasses
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    NamedTuple,
    assert_never,
    cast,
    get_args,
    get_origin,
)

import pydantic

from furu._declared_types import cached_type_hints, child_declared_type
from furu.constants import FIELDSMARKER, KINDMARKER, TYPEMARKER
from furu.logging import get_logger
from furu.result.codec import Codec, CodecMeta
from furu.storage._layout import scratch_dir_in
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name

WRAPPER_KEY: Final = "$furu"
# artifacts/<position>: what a codec merged for every row with a value there.
ARTIFACTS_DIR_NAME: Final = "artifacts"
# rows/<row>/<position>: one value that was not merged.
ROWS_DIR_NAME: Final = "rows"
MANIFEST_FILE_NAME: Final = "manifest.json"
_ROOT_ARTIFACT_NAME: Final = "root"
_LARGE_FALLBACK_FOLDERS: Final = 100
type ValuePath = tuple[str, ...]
_WARNED_FALLBACKS: set[tuple[str, str]] = set()  # label and field, once per process
type WrapperKind = Literal[
    "artifact",
    "dataclass",
    "datetime",
    "path",
    "data_path",
    "pydantic",
    "tuple",
    "set",
    "frozenset",
]


class _Cell(NamedTuple):
    """One codec value, and the manifest entry that will say where it went."""

    row: int
    codec: type[Codec]
    value: object
    body: dict[str, JsonValue]


@dataclasses.dataclass
class _Writer:
    """Sets aside the codec values of every row, by their position in the result."""

    result_codecs: tuple[type[Codec], ...]
    data_dir: Path
    row: int = 0
    columns: dict[ValuePath, list[_Cell]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _Reader:
    """Reads the codec values of the rows asked for, each directory once."""

    bundle_dir: Path
    data_dir: Path
    # The manifest entries of the codec values, by the directory they are in,
    # and the values read so far, by the id() of their entry.
    bodies: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    values: dict[int, object] = dataclasses.field(default_factory=dict)


def _value_path_display(value_path: ValuePath) -> str:
    if not value_path:
        return "<root>"
    return "/".join(value_path)


def _validate_result_path_segment(
    value: object,
    *,
    parent_value_path: ValuePath,
) -> str:
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 -- malformed payload, not a bad argument
            f"Unsupported result value at {_value_path_display(parent_value_path)}:\n"
            + f"must be strings; got {type(value).__name__} key {value!r}."
        )
    if value == WRAPPER_KEY:
        raise ValueError(
            f"Unsupported result value at {_value_path_display(parent_value_path)}:\n"
            + f"named {WRAPPER_KEY!r} are reserved by furu result persistence."
        )
    if (
        value == ""
        or value == "."
        or value == ".."
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(
            f"Unsupported result path at {_value_path_display((*parent_value_path, value))}:\n"
            + "cannot be used as an artifact path segment."
        )
    return value


def _wrap(kind: WrapperKind, **body: JsonValue) -> JsonValue:
    return {WRAPPER_KEY: {KINDMARKER: kind, **body}}


def _dump_value(
    value: object, declared_type: object, value_path: ValuePath, writer: _Writer
) -> JsonValue:
    annotated_codec: type[Codec] | None = None
    if get_origin(declared_type) is Annotated:
        for item in get_args(declared_type)[1:]:
            if isinstance(item, type) and issubclass(item, Codec):
                annotated_codec = item
                break

    if annotated_codec is not None:
        return _dump_artifact(value, annotated_codec, value_path, writer)

    match value:
        case None | bool() | int() | float() | str():
            return value
        case datetime():
            if codec := CodecMeta.find_codec(value, writer.result_codecs):
                return _dump_artifact(value, codec, value_path, writer)
            encoded = value.isoformat()
            loaded = datetime.fromisoformat(encoded)
            if (
                type(value) is not datetime
                or loaded != value
                or loaded.fold != value.fold
                or type(loaded.tzinfo) is not type(value.tzinfo)
                or loaded.tzinfo != value.tzinfo
                or loaded.tzname() != value.tzname()
            ):
                raise ValueError(
                    f"Datetime at {_value_path_display(value_path)} cannot be "
                    "faithfully stored as ISO format; use an explicit custom codec."
                )
            return _wrap("datetime", value=encoded)
        case list():
            return _dump_items(value, declared_type, value_path, writer)
        case tuple():
            return _wrap(
                "tuple", items=_dump_items(value, declared_type, value_path, writer)
            )
        case set() | frozenset():
            kind = "frozenset" if isinstance(value, frozenset) else "set"
            for item in value:
                if type(item).__repr__ is object.__repr__:
                    raise ValueError(
                        f"Unsupported result value at {_value_path_display(value_path)}:\n"
                        f"set members of type {type(item).__name__!r} have no "
                        "value-based repr, so their order cannot be made "
                        "deterministic; use a list or implement __repr__."
                    )
            items = sorted(
                value,
                key=lambda item: (
                    type(item).__module__,
                    type(item).__qualname__,
                    repr(item),
                ),
            )
            return _wrap(
                kind, items=_dump_items(items, declared_type, value_path, writer)
            )
        case dict():
            out: dict[str, JsonValue] = {}
            for raw_key, child in value.items():
                key = _validate_result_path_segment(
                    raw_key,
                    parent_value_path=value_path,
                )
                out[key] = _dump_value(
                    child,
                    child_declared_type(declared_type, raw_key),
                    (*value_path, key),
                    writer,
                )
            return out
        case Path():
            data_dir = writer.data_dir
            scratch_dir = scratch_dir_in(data_dir.parent)
            resolved = value.resolve()
            if resolved.is_relative_to(scratch_dir.resolve()):
                raise ValueError(
                    f"Result path at {_value_path_display(value_path)} must not "
                    f"point into the scratch dir {scratch_dir}, which is deleted "
                    f"once the result is stored: {value}"
                )
            if resolved.is_relative_to(data_dir.resolve()):
                # The data dir moves when the attempt is published.
                return _wrap(
                    "data_path",
                    value=resolved.relative_to(data_dir.resolve()).as_posix(),
                )
            return _wrap("path", value=str(value))
        case pydantic.BaseModel():
            names = type(value).model_fields
            return _dump_fields("pydantic", value, names, value_path, writer)
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            names = [field.name for field in dataclasses.fields(value)]
            return _dump_fields("dataclass", value, names, value_path, writer)
        case _:
            if codec := CodecMeta.find_codec(value, writer.result_codecs):
                return _dump_artifact(value, codec, value_path, writer)

    raise ValueError(
        f"Unsupported result value at {_value_path_display(value_path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by furu. Add a custom codec"
    )


def _dump_fields(
    kind: WrapperKind,
    value: object,
    names: Iterable[str],
    value_path: ValuePath,
    writer: _Writer,
) -> JsonValue:
    field_types = cached_type_hints(type(value))
    fields: dict[str, JsonValue] = {
        name: _dump_value(
            getattr(value, name),
            field_types.get(name, Any),
            (*value_path, name),
            writer,
        )
        for name in names
    }
    return _wrap(
        kind, **{TYPEMARKER: fully_qualified_name(type(value)), FIELDSMARKER: fields}
    )


def _dump_items(
    items: Sequence[object],
    declared_type: object,
    value_path: ValuePath,
    writer: _Writer,
) -> list[JsonValue]:
    width = len(str(len(items)))
    return [
        _dump_value(
            item,
            child_declared_type(declared_type, i),
            (*value_path, f"{i:0{width}d}"),
            writer,
        )
        for i, item in enumerate(items)
    ]


def _dump_artifact(
    value: object, codec: type[Codec], value_path: ValuePath, writer: _Writer
) -> JsonValue:
    body: dict[str, JsonValue] = {KINDMARKER: "artifact", "codec": codec._codec_id()}
    writer.columns.setdefault(value_path, []).append(
        _Cell(writer.row, codec, value, body)
    )
    return {WRAPPER_KEY: body}


def _save_columns(
    columns: dict[ValuePath, list[_Cell]],
    bundle_dir: Path,
    *,
    data_dir: Path,
    label: str,
) -> None:
    """Save each position's values and fill in where each one went."""
    positions = [value_path or (_ROOT_ARTIFACT_NAME,) for value_path in columns]
    # A value where another row holds a container, or the whole result of one
    # row where another has a field named "root", would share its directory.
    uses = Counter(
        position[:i] for position in positions for i in range(1, len(position) + 1)
    )
    clashing = {position for position in positions if uses[position] > 1}
    row_folders: Counter[str] = Counter()  # by field
    for (value_path, cells), position in zip(columns.items(), positions):
        codec = cells[0].codec
        shared = Path(ARTIFACTS_DIR_NAME, *position)
        metadatas = None
        if all(cell.codec is codec for cell in cells) and not any(
            position[:i] in clashing for i in range(1, len(position) + 1)
        ):
            (bundle_dir / shared).mkdir(parents=True)
            metadatas = codec.save_many(
                [cell.value for cell in cells], bundle_dir / shared
            )
            if metadatas is None:
                # Up to the first directory that another position uses.
                with contextlib.suppress(OSError):
                    for path in (shared, *shared.parents[: len(position)]):
                        (bundle_dir / path).rmdir()
        paths = [shared] * len(cells)
        if metadatas is None:
            paths = [Path(ROWS_DIR_NAME, str(cell.row), *position) for cell in cells]
            for path in paths:
                (bundle_dir / path).mkdir(parents=True)
            metadatas = [
                cell.codec.save(cell.value, bundle_dir / path)
                for cell, path in zip(cells, paths)
            ]
            row_folders[_value_path_display(value_path[:1])] += len(cells)
        for cell, path, metadata in zip(cells, paths, metadatas, strict=True):
            cell.body["path"] = path.as_posix()
            cell.body["metadata"] = _encode_codec_metadata_value(
                metadata, data_dir=data_dir, value_path=value_path
            )
    if row_folders.total() >= _LARGE_FALLBACK_FOLDERS:
        for field, count in row_folders.items():
            if (label, field) not in _WARNED_FALLBACKS:
                _WARNED_FALLBACKS.add((label, field))
                get_logger().warning(
                    "%s: %d values in %s were not merged into one file and "
                    "got a directory each",
                    label,
                    count,
                    field,
                )


def _encode_codec_metadata_value(
    value: object,
    *,
    data_dir: Path,
    value_path: ValuePath,
) -> JsonValue:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Path():
            resolved = value.resolve()
            data_dir_resolved = data_dir.resolve()
            if not resolved.is_relative_to(data_dir_resolved):
                raise ValueError(
                    f"Codec metadata path at {_value_path_display(value_path)} "
                    f"must live inside the data dir {data_dir}: {value}"
                )
            return _wrap(
                "path", value=resolved.relative_to(data_dir_resolved).as_posix()
            )
        case list() | tuple():
            return [
                _encode_codec_metadata_value(
                    item, data_dir=data_dir, value_path=value_path
                )
                for item in value
            ]
        case Mapping():
            out: dict[str, JsonValue] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Codec metadata keys at {_value_path_display(value_path)} "
                        f"must be strings; got {type(key).__name__} key {key!r}"
                    )
                if key == WRAPPER_KEY:
                    raise ValueError(
                        f"Codec metadata key {WRAPPER_KEY!r} at "
                        f"{_value_path_display(value_path)} is reserved by furu"
                    )
                out[key] = _encode_codec_metadata_value(
                    child, data_dir=data_dir, value_path=value_path
                )
            return out
        case _:
            raise TypeError(
                f"Unsupported codec metadata value at {_value_path_display(value_path)}: "
                f"values of type {type(value).__name__!r} cannot be stored; use JSON "
                "scalars, lists, mappings, or Path"
            )


def _decode_codec_metadata_value(node: JsonValue, *, data_dir: Path) -> object:
    match node:
        case dict() if WRAPPER_KEY in node:
            return data_dir / cast(dict[str, Any], node[WRAPPER_KEY])["value"]
        case dict():
            return {
                key: _decode_codec_metadata_value(child, data_dir=data_dir)
                for key, child in node.items()
            }
        case list():
            return [
                _decode_codec_metadata_value(child, data_dir=data_dir) for child in node
            ]
        case _:
            return node


def _load_value(node: JsonValue, reader: _Reader) -> object:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            return [_load_value(child, reader) for child in node]
        case dict() if WRAPPER_KEY in node:
            return _load_wrapper(cast(dict[str, Any], node[WRAPPER_KEY]), reader)
        case dict():
            return {key: _load_value(child, reader) for key, child in node.items()}
        case _:
            assert_never(node)


def _load_fields(
    body: dict[str, Any], reader: _Reader
) -> tuple[Any, dict[str, object]]:
    cls = resolve_fully_qualified_name(body[TYPEMARKER])
    return cls, {
        name: _load_value(child, reader) for name, child in body[FIELDSMARKER].items()
    }


def _read_artifacts(
    bodies: Sequence[dict[str, Any]], *, bundle_dir: Path, data_dir: Path
) -> Sequence[object]:
    """Read codec values stored in one directory, in the order given."""
    codec: type[Codec] = resolve_fully_qualified_name(bodies[0]["codec"])
    path = Path(bodies[0]["path"])
    metadatas = [
        cast(
            dict[str, object],
            _decode_codec_metadata_value(body["metadata"], data_dir=data_dir),
        )
        for body in bodies
    ]
    if path.parts[0] == ROWS_DIR_NAME:
        return [codec.load(metadata, bundle_dir / path) for metadata in metadatas]
    return codec.load_many(metadatas, bundle_dir / path)


def _load_wrapper(body: dict[str, Any], reader: _Reader) -> object:
    kind: WrapperKind = body[KINDMARKER]
    match kind:
        case "artifact":
            if id(body) not in reader.values:
                bodies = reader.bodies[body["path"]]
                values = _read_artifacts(
                    bodies, bundle_dir=reader.bundle_dir, data_dir=reader.data_dir
                )
                reader.values.update(zip(map(id, bodies), values, strict=True))
            return reader.values[id(body)]
        case "dataclass":
            cls, fields = _load_fields(body, reader)
            for field in dataclasses.fields(cls):
                if not field.init:
                    fields.pop(field.name, None)
            return cls(**fields)
        case "datetime":
            return datetime.fromisoformat(body["value"])
        case "path":
            return Path(body["value"])
        case "data_path":
            return reader.data_dir / body["value"]
        case "tuple" | "set" | "frozenset":
            items = [_load_value(child, reader) for child in body["items"]]
            return {"tuple": tuple, "set": set, "frozenset": frozenset}[kind](items)
        case "pydantic":
            cls, fields = _load_fields(body, reader)
            return cls.model_validate(fields)
        case _:
            assert_never(kind)


def _collect_artifacts(
    node: JsonValue, bodies: dict[str, list[dict[str, Any]]]
) -> None:
    """Group the codec values under ``node`` by the directory they are stored in."""
    if isinstance(node, list):
        for child in node:
            _collect_artifacts(child, bodies)
    elif isinstance(node, dict):
        body = node.get(WRAPPER_KEY)
        if isinstance(body, dict) and body.get(KINDMARKER) == "artifact":
            bodies.setdefault(cast(str, body["path"]), []).append(body)
        else:
            for child in node.values():
                _collect_artifacts(child, bodies)


def _save_result_bundle(
    values: Sequence[object],
    bundle_dir: Path,
    *,
    result_codecs: tuple[type[Codec], ...],
    data_dir: Path,
    label: str,
) -> None:
    """Store one result per row. ``label`` names the results in a warning."""
    bundle_dir.mkdir(parents=True, exist_ok=False)

    writer = _Writer(result_codecs, data_dir)
    manifest: list[JsonValue] = []
    for row, value in enumerate(values):
        writer.row = row
        manifest.append(_dump_value(value, Any, (), writer))
    _save_columns(writer.columns, bundle_dir, data_dir=data_dir, label=label)
    # One row per line, so that a row can be found by hand.
    (bundle_dir / MANIFEST_FILE_NAME).write_text(
        "[\n" + ",\n".join(map(json.dumps, manifest)) + "\n]\n",
        encoding="utf-8",
    )


def load_result_bundle(
    bundle_dir: Path,
    rows: Sequence[int],
    *,
    data_dir: Path,
) -> list[object]:
    """Load the rows asked for, in the order asked."""
    manifest = json.loads((bundle_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    reader = _Reader(bundle_dir, data_dir)
    for row in rows:
        _collect_artifacts(manifest[row], reader.bodies)
    return [_load_value(manifest[row], reader) for row in rows]
