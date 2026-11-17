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

from furu._declared_types import cached_type_hints, child_declared_type, strip_annotated
from furu.constants import FIELDSMARKER, KINDMARKER, TYPEMARKER
from furu.logging import get_logger
from furu.result.codec import Codec, CodecMeta
from furu.result.ref import Ref
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

    match value:
        case Ref():
            if annotated_codec is not None and annotated_codec is not value._codec:
                raise TypeError(
                    "Conflicting codecs: the Ref carries a codec from furu.ref(...), "
                    "but the field also has a different Annotated codec."
                )
            return _dump_artifact(value.load(), value._codec, value_path, writer)
        case _ if annotated_codec is not None:
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
            fields_out: dict[str, JsonValue] = {}
            field_types = cached_type_hints(value.__class__)
            for raw_name in value.__class__.model_fields:
                name = _validate_result_path_segment(
                    raw_name, parent_value_path=value_path
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    field_types.get(name, Any),
                    (*value_path, name),
                    writer,
                )
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "pydantic",
                    TYPEMARKER: fully_qualified_name(type(value)),
                    FIELDSMARKER: fields_out,
                }
            }
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields_out: dict[str, JsonValue] = {}
            field_types = cached_type_hints(type(value))
            for field in dataclasses.fields(cast(Any, value)):
                name = _validate_result_path_segment(
                    field.name, parent_value_path=value_path
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    field_types.get(field.name, Any),
                    (*value_path, name),
                    writer,
                )
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "dataclass",
                    TYPEMARKER: fully_qualified_name(type(value)),
                    FIELDSMARKER: fields_out,
                }
            }
        case _:
            if codec := CodecMeta.find_codec(value, writer.result_codecs):
                return _dump_artifact(value, codec, value_path, writer)

    raise ValueError(
        f"Unsupported result value at {_value_path_display(value_path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by furu. Add a custom codec"
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
            elif len(metadatas) != len(cells):
                raise TypeError(
                    f"{codec.__name__}.save_many() at "
                    f"{_value_path_display(value_path)} returned "
                    f"{len(metadatas)} metadata for {len(cells)} values"
                )
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
        for cell, path, metadata in zip(cells, paths, metadatas):
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    f"{cell.codec.__name__} at {_value_path_display(value_path)} "
                    f"must return metadata mappings; got {type(metadata).__name__!r}"
                )
            cell.body["path"] = path.as_posix()
            cell.body["metadata"] = _encode_codec_metadata_value(
                dict(metadata), data_dir=data_dir, value_path=value_path
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
        case list():
            return [
                _encode_codec_metadata_value(
                    item, data_dir=data_dir, value_path=value_path
                )
                for item in value
            ]
        case tuple():
            return _wrap(
                "tuple",
                items=[
                    _encode_codec_metadata_value(
                        item, data_dir=data_dir, value_path=value_path
                    )
                    for item in value
                ],
            )
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
            body = cast(dict[str, Any], node[WRAPPER_KEY])
            match body[KINDMARKER]:
                case "path":
                    rel_path = Path(body["value"])
                    if rel_path.is_absolute():
                        raise ValueError(
                            f"codec metadata path must be relative: {rel_path}"
                        )
                    resolved_path = (data_dir.resolve() / rel_path).resolve()
                    if not resolved_path.is_relative_to(data_dir.resolve()):
                        raise ValueError(
                            f"codec metadata path escapes data dir: {rel_path}"
                        )
                    return resolved_path
                case "tuple":
                    return tuple(
                        _decode_codec_metadata_value(child, data_dir=data_dir)
                        for child in body["items"]
                    )
                case _:
                    raise ValueError(
                        f"unknown codec metadata wrapper kind: {body[KINDMARKER]!r}"
                    )
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


def _load_value(
    node: JsonValue, declared_type: object, reader: _Reader, value_path: ValuePath
) -> object:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            return list(_load_items(node, declared_type, reader, value_path))
        case dict() if WRAPPER_KEY in node:
            return _load_wrapper(
                cast(dict[str, Any], node[WRAPPER_KEY]),
                declared_type,
                reader,
                value_path,
            )
        case dict():
            return {
                key: _load_value(
                    child,
                    child_declared_type(declared_type, key),
                    reader,
                    (*value_path, key),
                )
                for key, child in node.items()
            }
        case _:
            assert_never(node)


def _load_items(
    items: list[JsonValue],
    declared_type: object,
    reader: _Reader,
    value_path: ValuePath,
) -> Iterable[object]:
    width = len(str(len(items)))
    return (
        _load_value(
            child,
            child_declared_type(declared_type, i),
            reader,
            (*value_path, f"{i:0{width}d}"),
        )
        for i, child in enumerate(items)
    )


def _load_validated_fields(
    *,
    kind: str,
    cls: type[Any],
    expected: set[str],
    raw_fields: dict[str, JsonValue],
    reader: _Reader,
    value_path: ValuePath,
) -> dict[str, object]:
    actual = set(raw_fields)
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        field_types = cached_type_hints(cls)
        return {
            name: _load_value(
                child, field_types.get(name, Any), reader, (*value_path, name)
            )
            for name, child in raw_fields.items()
        }

    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(sorted(missing)))
    if extra:
        details.append("extra fields: " + ", ".join(sorted(extra)))

    raise ValueError(
        f"Cannot load {kind} {fully_qualified_name(cls)} at {_value_path_display(value_path)}: "
        + "; ".join(details)
    )


def _codec_of(body: dict[str, Any]) -> type[Codec]:
    codec = resolve_fully_qualified_name(body["codec"])
    if not isinstance(codec, type) or not issubclass(codec, Codec):
        raise TypeError(f"{body['codec']} is not a furu.Codec")
    return codec


def _read_artifacts(
    bodies: Sequence[dict[str, Any]], *, bundle_dir: Path, data_dir: Path
) -> Sequence[object]:
    """Read codec values stored in one directory, in the order given."""
    codec = _codec_of(bodies[0])
    path = Path(bodies[0]["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"artifact path escapes the result bundle: {path}")
    metadatas = [
        cast(
            dict[str, object],
            _decode_codec_metadata_value(body["metadata"], data_dir=data_dir),
        )
        for body in bodies
    ]
    if path.parts[0] == ROWS_DIR_NAME:
        return [codec.load(metadata, bundle_dir / path) for metadata in metadatas]
    values = codec.load_many(metadatas, bundle_dir / path)
    if len(values) != len(bodies):
        raise TypeError(
            f"{codec.__name__}.load_many() returned {len(values)} values "
            f"for {len(bodies)} metadata"
        )
    return values


def _load_wrapper(
    body: dict[str, Any], declared_type: object, reader: _Reader, value_path: ValuePath
) -> object:
    kind: WrapperKind = body[KINDMARKER]
    match kind:
        case "artifact":
            bundle_dir, data_dir = reader.bundle_dir, reader.data_dir
            if (declared := strip_annotated(declared_type)) is Ref or get_origin(
                declared
            ) is Ref:
                # Not the reader: it holds every value this load has read.
                return Ref._from_stored(
                    codec=_codec_of(body),
                    read=lambda: _read_artifacts(
                        [body], bundle_dir=bundle_dir, data_dir=data_dir
                    )[0],
                )
            if id(body) not in reader.values:
                bodies = reader.bodies[body["path"]]
                values = _read_artifacts(
                    bodies, bundle_dir=bundle_dir, data_dir=data_dir
                )
                reader.values.update(zip(map(id, bodies), values))
            return reader.values[id(body)]
        case "dataclass":
            cls = resolve_fully_qualified_name(body[TYPEMARKER])
            if not dataclasses.is_dataclass(cls):
                raise ValueError(
                    f"Cannot load dataclass at {_value_path_display(value_path)}: "
                    f"{fully_qualified_name(cls)} is not a dataclass"
                )
            dataclass_fields = dataclasses.fields(cls)
            init_fields = {field.name for field in dataclass_fields if field.init}
            loaded_fields = _load_validated_fields(
                kind="dataclass",
                cls=cls,
                expected={field.name for field in dataclass_fields},
                raw_fields=body[FIELDSMARKER],
                reader=reader,
                value_path=value_path,
            )
            try:
                return cls(
                    **{
                        name: value
                        for name, value in loaded_fields.items()
                        if name in init_fields
                    }
                )
            except Exception as exc:
                raise ValueError(
                    f"Cannot load dataclass {fully_qualified_name(cls)} "
                    f"at {_value_path_display(value_path)}: {exc}"
                ) from exc
        case "datetime":
            return datetime.fromisoformat(body["value"])
        case "path":
            return Path(body["value"])
        case "data_path":
            return reader.data_dir / body["value"]
        case "tuple":
            return tuple(_load_items(body["items"], declared_type, reader, value_path))
        case "set":
            return set(_load_items(body["items"], declared_type, reader, value_path))
        case "frozenset":
            return frozenset(
                _load_items(body["items"], declared_type, reader, value_path)
            )
        case "pydantic":
            cls = resolve_fully_qualified_name(body[TYPEMARKER])
            if not issubclass(cls, pydantic.BaseModel):
                raise ValueError(  # noqa: TRY004 -- malformed payload, not a bad argument
                    f"Cannot load pydantic model at {_value_path_display(value_path)}: "
                    f"{fully_qualified_name(cls)} is not a pydantic model"
                )
            loaded_fields = _load_validated_fields(
                kind="pydantic model",
                cls=cls,
                expected=set(cls.model_fields),
                raw_fields=body[FIELDSMARKER],
                reader=reader,
                value_path=value_path,
            )
            try:
                return cls.model_validate(loaded_fields)
            except pydantic.ValidationError as exc:
                raise ValueError(
                    f"Cannot load pydantic model {fully_qualified_name(cls)} "
                    f"at {_value_path_display(value_path)}: {exc}"
                ) from exc
        case _:
            raise ValueError(f"unknown wrapper kind: {kind!r}")


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
    declared_type: object,
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
        manifest.append(_dump_value(value, declared_type, (), writer))
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
    declared_type: object,
) -> list[object]:
    """Load the rows asked for, in the order asked."""
    manifest = json.loads((bundle_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    reader = _Reader(bundle_dir, data_dir)
    for row in rows:
        _collect_artifacts(manifest[row], reader.bodies)
    return [_load_value(manifest[row], declared_type, reader, ()) for row in rows]
