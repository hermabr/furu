from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    assert_never,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic

from furu._declared_types import child_declared_type, strip_annotated
from furu.constants import FIELDSMARKER, KINDMARKER, TYPEMARKER
from furu.result.codec import Codec, CodecMeta
from furu.result.ref import Ref
from furu.storage._layout import scratch_dir_in
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name

WRAPPER_KEY: Final = "$furu"
ARTIFACTS_DIR_NAME: Final = "artifacts"
MANIFEST_FILE_NAME: Final = "manifest.json"
_ROOT_ARTIFACT_NAME: Final = "root"
type ValuePath = tuple[str, ...]
type WrapperKind = Literal[
    "artifact",
    "dataclass",
    "datetime",
    "path",
    "pydantic",
    "tuple",
    "set",
    "frozenset",
]


@dataclasses.dataclass
class _RefBinding:
    ref: Ref[Any]
    metadata: Mapping[str, object]
    artifact_relative_path: Path


@dataclasses.dataclass
class _DumpState:
    data_dir: Path
    should_reload_value_after_save: bool = False
    ref_bindings: list[_RefBinding] = dataclasses.field(default_factory=list)


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


def _dump_value(
    value: object,
    *,
    declared_type: object,
    value_path: ValuePath,
    bundle_dir: Path,
    result_codecs: tuple[type[Codec], ...],
    dump_state: _DumpState,
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
            return _dump_artifact(
                value.load(),
                codec=value._codec,
                value_path=value_path,
                bundle_dir=bundle_dir,
                dump_state=dump_state,
                ref=value,
            )
        case _ if annotated_codec is not None:
            return _dump_artifact(
                value,
                codec=annotated_codec,
                value_path=value_path,
                bundle_dir=bundle_dir,
                dump_state=dump_state,
            )

    match value:
        case None | bool() | int() | float() | str():
            return value
        case datetime():
            if codec := CodecMeta.find_codec(value, result_codecs):
                return _dump_artifact(
                    value,
                    codec=codec,
                    value_path=value_path,
                    bundle_dir=bundle_dir,
                    dump_state=dump_state,
                )
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
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "datetime",
                    "value": encoded,
                }
            }
        case list():
            width = len(str(len(value)))
            return [
                _dump_value(
                    item,
                    declared_type=child_declared_type(declared_type, i),
                    value_path=(*value_path, f"{i:0{width}d}"),
                    bundle_dir=bundle_dir,
                    result_codecs=result_codecs,
                    dump_state=dump_state,
                )
                for i, item in enumerate(value)
            ]
        case tuple():
            width = len(str(len(value)))
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "tuple",
                    "items": [
                        _dump_value(
                            item,
                            declared_type=child_declared_type(declared_type, i),
                            value_path=(*value_path, f"{i:0{width}d}"),
                            bundle_dir=bundle_dir,
                            result_codecs=result_codecs,
                            dump_state=dump_state,
                        )
                        for i, item in enumerate(value)
                    ],
                }
            }
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
            width = len(str(len(items)))
            return {
                WRAPPER_KEY: {
                    KINDMARKER: kind,
                    "items": [
                        _dump_value(
                            item,
                            declared_type=child_declared_type(declared_type, i),
                            value_path=(*value_path, f"{i:0{width}d}"),
                            bundle_dir=bundle_dir,
                            result_codecs=result_codecs,
                            dump_state=dump_state,
                        )
                        for i, item in enumerate(items)
                    ],
                }
            }
        case dict():
            out: dict[str, JsonValue] = {}
            for raw_key, child in value.items():
                key = _validate_result_path_segment(
                    raw_key,
                    parent_value_path=value_path,
                )
                out[key] = _dump_value(
                    child,
                    declared_type=child_declared_type(declared_type, raw_key),
                    value_path=(*value_path, key),
                    bundle_dir=bundle_dir,
                    result_codecs=result_codecs,
                    dump_state=dump_state,
                )
            return out
        case Path():
            scratch_dir = scratch_dir_in(dump_state.data_dir.parent)
            if value.resolve().is_relative_to(scratch_dir.resolve()):
                raise ValueError(
                    f"Result path at {_value_path_display(value_path)} must not "
                    f"point into the scratch dir {scratch_dir}, which is deleted "
                    f"once the result is stored: {value}"
                )
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "path",
                    "value": str(value),
                }
            }
        case pydantic.BaseModel():
            fields_out: dict[str, JsonValue] = {}
            field_types = get_type_hints(value.__class__, include_extras=True)
            for raw_name in value.__class__.model_fields:
                name = _validate_result_path_segment(
                    raw_name, parent_value_path=value_path
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    declared_type=field_types.get(name, Any),
                    value_path=(*value_path, name),
                    bundle_dir=bundle_dir,
                    result_codecs=result_codecs,
                    dump_state=dump_state,
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
            field_types = get_type_hints(type(value), include_extras=True)
            for field in dataclasses.fields(cast(Any, value)):
                name = _validate_result_path_segment(
                    field.name, parent_value_path=value_path
                )
                fields_out[name] = _dump_value(
                    getattr(value, name),
                    declared_type=field_types.get(field.name, Any),
                    value_path=(*value_path, name),
                    bundle_dir=bundle_dir,
                    result_codecs=result_codecs,
                    dump_state=dump_state,
                )
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "dataclass",
                    TYPEMARKER: fully_qualified_name(type(value)),
                    FIELDSMARKER: fields_out,
                }
            }
        case _:
            if codec := CodecMeta.find_codec(value, result_codecs):
                return _dump_artifact(
                    value,
                    codec=codec,
                    value_path=value_path,
                    bundle_dir=bundle_dir,
                    dump_state=dump_state,
                )

    raise ValueError(
        f"Unsupported result value at {_value_path_display(value_path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by furu. Add a custom codec"
    )


def _dump_artifact(
    value: object,
    *,
    codec: type[Codec],
    value_path: ValuePath,
    bundle_dir: Path,
    dump_state: _DumpState,
    ref: Ref[Any] | None = None,
) -> JsonValue:
    artifact_rel = Path(ARTIFACTS_DIR_NAME, *(value_path or (_ROOT_ARTIFACT_NAME,)))
    artifact_dir = bundle_dir / artifact_rel
    artifact_dir.mkdir(parents=True, exist_ok=False)

    codec_metadata = codec.save(value, artifact_dir)
    if not isinstance(codec_metadata, Mapping):
        raise TypeError(
            f"Codec save() at {_value_path_display(value_path)} must return a "
            f"metadata mapping; got {type(codec_metadata).__name__!r}"
        )
    encoded_metadata = _encode_codec_metadata_value(
        dict(codec_metadata), data_dir=dump_state.data_dir, value_path=value_path
    )
    assert isinstance(encoded_metadata, dict)

    if ref is not None:
        dump_state.ref_bindings.append(
            _RefBinding(
                ref=ref,
                metadata=cast(
                    dict[str, object],
                    _decode_codec_metadata_value(
                        encoded_metadata, data_dir=dump_state.data_dir
                    ),
                ),
                artifact_relative_path=artifact_rel,
            )
        )
    elif codec.reload_value_after_save:
        dump_state.should_reload_value_after_save = True

    return {
        WRAPPER_KEY: {
            KINDMARKER: "artifact",
            "codec": codec._codec_id(),
            "path": artifact_rel.as_posix(),
            "metadata": encoded_metadata,
        }
    }


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
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "path",
                    "value": resolved.relative_to(data_dir_resolved).as_posix(),
                }
            }
        case list():
            return [
                _encode_codec_metadata_value(
                    item, data_dir=data_dir, value_path=value_path
                )
                for item in value
            ]
        case tuple():
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "tuple",
                    "items": [
                        _encode_codec_metadata_value(
                            item, data_dir=data_dir, value_path=value_path
                        )
                        for item in value
                    ],
                }
            }
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
    node: JsonValue,
    *,
    declared_type: object,
    bundle_dir: Path,
    data_dir: Path,
    value_path: ValuePath,
) -> object:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            width = len(str(len(node)))
            return [
                _load_value(
                    child,
                    declared_type=child_declared_type(declared_type, i),
                    bundle_dir=bundle_dir,
                    data_dir=data_dir,
                    value_path=(*value_path, f"{i:0{width}d}"),
                )
                for i, child in enumerate(node)
            ]
        case dict() if WRAPPER_KEY in node:
            return _load_wrapper(
                cast(dict[str, Any], node[WRAPPER_KEY]),
                declared_type=declared_type,
                bundle_dir=bundle_dir,
                data_dir=data_dir,
                value_path=value_path,
            )
        case dict():
            return {
                key: _load_value(
                    child,
                    declared_type=child_declared_type(declared_type, key),
                    bundle_dir=bundle_dir,
                    data_dir=data_dir,
                    value_path=(*value_path, key),
                )
                for key, child in node.items()
            }
        case _:
            assert_never(node)


def _structured_class(body: dict[str, Any], *, value_path: ValuePath) -> type[Any]:
    cls = resolve_fully_qualified_name(body[TYPEMARKER])
    if body[KINDMARKER] == "dataclass" and not dataclasses.is_dataclass(cls):
        raise ValueError(
            f"Cannot load dataclass at {_value_path_display(value_path)}: "
            f"{fully_qualified_name(cls)} is not a dataclass"
        )
    if body[KINDMARKER] == "pydantic" and not issubclass(cls, pydantic.BaseModel):
        raise ValueError(
            f"Cannot load pydantic model at {_value_path_display(value_path)}: "
            f"{fully_qualified_name(cls)} is not a pydantic model"
        )
    return cast(type[Any], cls)


def _load_fields(
    cls: type[Any],
    raw_fields: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    data_dir: Path,
    value_path: ValuePath,
) -> dict[str, object]:
    field_types = get_type_hints(cls, include_extras=True)
    return {
        name: _load_value(
            child,
            declared_type=field_types.get(name, Any),
            bundle_dir=bundle_dir,
            data_dir=data_dir,
            value_path=(*value_path, name),
        )
        for name, child in raw_fields.items()
    }


def _build_structured(
    cls: type[Any], fields: Mapping[str, object], *, value_path: ValuePath
) -> object:
    """Construct a dataclass or pydantic ``cls`` from decoded ``fields``,
    rejecting a field set that differs from the class."""
    is_model = issubclass(cls, pydantic.BaseModel)
    kind = "pydantic model" if is_model else "dataclass"
    expected = (
        set(cls.model_fields)
        if is_model
        else {field.name for field in dataclasses.fields(cls)}
    )
    actual = set(fields)
    details: list[str] = []
    if missing := expected - actual:
        details.append("missing fields: " + ", ".join(sorted(missing)))
    if extra := actual - expected:
        details.append("extra fields: " + ", ".join(sorted(extra)))
    if details:
        raise ValueError(
            f"Cannot load {kind} {fully_qualified_name(cls)} at "
            f"{_value_path_display(value_path)}: " + "; ".join(details)
        )
    try:
        if is_model:
            return cls.model_validate(dict(fields))
        init_fields = {field.name for field in dataclasses.fields(cls) if field.init}
        return cls(**{name: fields[name] for name in init_fields})
    except Exception as exc:
        raise ValueError(
            f"Cannot load {kind} {fully_qualified_name(cls)} "
            f"at {_value_path_display(value_path)}: {exc}"
        ) from exc


def _load_wrapper(
    body: dict[str, Any],
    *,
    declared_type: object,
    bundle_dir: Path,
    data_dir: Path,
    value_path: ValuePath,
) -> object:
    kind: WrapperKind = body[KINDMARKER]
    match kind:
        case "artifact":
            artifact_rel = Path(body["path"])
            if artifact_rel.is_absolute():
                raise ValueError(
                    f"artifact wrapper path must be relative: {artifact_rel}"
                )

            artifact_dir = (bundle_dir / artifact_rel).resolve()
            artifacts_root = (bundle_dir / ARTIFACTS_DIR_NAME).resolve()
            if not artifact_dir.is_relative_to(artifacts_root):
                raise ValueError(
                    f"artifact wrapper path escapes bundle artifacts dir: {artifact_rel}"
                )

            if not artifact_dir.exists():
                raise ValueError(
                    f"artifact wrapper artifact directory missing: {artifact_dir}"
                )

            codec_id = body["codec"]
            codec = resolve_fully_qualified_name(codec_id)
            if not isinstance(codec, type) or not issubclass(codec, Codec):
                raise TypeError(f"{codec_id} is not a furu.Codec")

            if not isinstance(
                metadata := _decode_codec_metadata_value(
                    body["metadata"], data_dir=data_dir
                ),
                dict,
            ):
                raise ValueError(  # noqa: TRY004 -- malformed payload, not a bad argument
                    f"Codec metadata at {_value_path_display(value_path)} must be a mapping"
                )
            metadata = cast(dict[str, object], metadata)
            if (declared := strip_annotated(declared_type)) is Ref or get_origin(
                declared
            ) is Ref:
                return Ref._from_stored(
                    codec=codec,
                    metadata=metadata,
                    artifact_directory=artifact_dir,
                )
            return codec.load(metadata, artifact_dir)
        case "dataclass" | "pydantic":
            cls = _structured_class(body, value_path=value_path)
            return _build_structured(
                cls,
                _load_fields(
                    cls,
                    body[FIELDSMARKER],
                    bundle_dir=bundle_dir,
                    data_dir=data_dir,
                    value_path=value_path,
                ),
                value_path=value_path,
            )
        case "datetime":
            return datetime.fromisoformat(body["value"])
        case "path":
            return Path(body["value"])
        case "tuple":
            return tuple(
                _load_value(
                    child,
                    declared_type=child_declared_type(declared_type, i),
                    bundle_dir=bundle_dir,
                    data_dir=data_dir,
                    value_path=(*value_path, str(i)),
                )
                for i, child in enumerate(body["items"])
            )
        case "set":
            return {
                _load_value(
                    child,
                    declared_type=child_declared_type(declared_type, i),
                    bundle_dir=bundle_dir,
                    data_dir=data_dir,
                    value_path=(*value_path, str(i)),
                )
                for i, child in enumerate(body["items"])
            }
        case "frozenset":
            return frozenset(
                _load_value(
                    child,
                    declared_type=child_declared_type(declared_type, i),
                    bundle_dir=bundle_dir,
                    data_dir=data_dir,
                    value_path=(*value_path, str(i)),
                )
                for i, child in enumerate(body["items"])
            )
        case _:
            raise ValueError(f"unknown wrapper kind: {kind!r}")


def _save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object,
    result_codecs: tuple[type[Codec], ...],
    data_dir: Path,
) -> _DumpState:
    bundle_dir.mkdir(parents=True, exist_ok=False)

    dump_state = _DumpState(data_dir=data_dir)
    manifest = _dump_value(
        value,
        declared_type=declared_type,
        value_path=(),
        bundle_dir=bundle_dir,
        result_codecs=result_codecs,
        dump_state=dump_state,
    )
    (bundle_dir / MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return dump_state


def _rewrite_target_class(raw: JsonValue, declared_type: object) -> type[Any] | None:
    """The class a top-level dataclass/pydantic result is rebuilt as after
    rewrites: the declared result type when it is one, else the stored name."""
    if not (isinstance(raw, dict) and WRAPPER_KEY in raw):
        return None
    body = cast(dict[str, Any], raw[WRAPPER_KEY])
    if body[KINDMARKER] not in ("dataclass", "pydantic"):
        return None
    declared = strip_annotated(declared_type)
    if isinstance(declared, type) and (
        dataclasses.is_dataclass(declared) or issubclass(declared, pydantic.BaseModel)
    ):
        return declared
    return _structured_class(body, value_path=())


def load_result_bundle(
    bundle_dir: Path,
    *,
    data_dir: Path,
    declared_type: object,
    rewrites: Sequence[Callable[[Any], Any]] = (),
) -> object:
    """Decode the stored result, replaying ``rewrites`` (from migration steps)
    over the decoded value in order.

    A top-level dataclass or pydantic result is handed to the rewrites as a
    dict of its decoded fields and rebuilt from the dict they return, using the
    declared result class rather than the stored name. That is what lets a
    rewrite add a field the stored wrapper lacks, or follow a moved class.
    """
    raw = json.loads((bundle_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    cls = _rewrite_target_class(raw, declared_type) if rewrites else None
    if cls is None:
        value = _load_value(
            raw,
            declared_type=declared_type,
            bundle_dir=bundle_dir,
            data_dir=data_dir,
            value_path=(),
        )
    else:
        value = _load_fields(
            cls,
            cast(dict[str, Any], raw)[WRAPPER_KEY][FIELDSMARKER],
            bundle_dir=bundle_dir,
            data_dir=data_dir,
            value_path=(),
        )
    for rewrite in rewrites:
        value = rewrite(value)
        if cls is not None and not isinstance(value, Mapping):
            raise TypeError(
                f"result_rewrite {getattr(rewrite, '__qualname__', rewrite)!r} must "
                f"return a mapping of {fully_qualified_name(cls)} fields; got "
                f"{type(value).__name__}"
            )
    if cls is None:
        return value
    return _build_structured(cls, cast(Mapping[str, object], value), value_path=())
