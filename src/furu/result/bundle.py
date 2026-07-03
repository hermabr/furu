from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    TypeAlias,
    assert_never,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic

from furu._declared_types import child_declared_type, strip_annotated
from furu.constants import DATAPATHMARKER, FIELDSMARKER, KINDMARKER, TYPEMARKER
from furu.result.codec import Codec, CodecMeta
from furu.result.ref import Ref
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name

WRAPPER_KEY: Final = "$furu"
ARTIFACTS_DIR_NAME: Final = "artifacts"
MANIFEST_FILE_NAME: Final = "manifest.json"
METADATA_KEY: Final = "metadata"
_ROOT_ARTIFACT_NAME: Final = "root"
ValuePath: TypeAlias = tuple[str, ...]
WrapperKind: TypeAlias = Literal[
    "artifact", "dataclass", "path", "pydantic", "tuple", "set", "frozenset"
]


@dataclasses.dataclass
class _RefRebind:
    ref: Ref[Any]
    codec: type[Codec]
    artifact_rel: Path
    encoded_metadata: JsonValue


@dataclasses.dataclass
class SaveResult:
    should_reload_value_after_save: bool
    ref_rebinds: list[_RefRebind]


@dataclasses.dataclass
class _DumpState:
    data_dir: Path
    should_reload_value_after_save: bool = False
    ref_rebinds: list[_RefRebind] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _ArtifactDump:
    wrapper: JsonValue
    artifact_rel: Path
    encoded_metadata: JsonValue


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
        raise ValueError(
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


def _annotated_codec(declared_type: object) -> type[Codec] | None:
    if get_origin(declared_type) is Annotated:
        for item in get_args(declared_type)[1:]:
            if isinstance(item, type) and issubclass(item, Codec):
                return item
    return None


def _ref_inner_declared_type(declared_type: object) -> object:
    stripped = strip_annotated(declared_type)
    if get_origin(stripped) is Ref:
        args = get_args(stripped)
        if args:
            return args[0]
    return Any


def _declares_ref(declared_type: object) -> bool:
    return get_origin(strip_annotated(declared_type)) is Ref


def _dump_value(
    value: object,
    *,
    declared_type: object = Any,
    value_path: ValuePath,
    bundle_dir: Path,
    result_codecs: tuple[type[Codec], ...],
    dump_state: _DumpState,
) -> JsonValue:
    if isinstance(value, Ref):
        inner_value = value.load()
        codec = _resolve_ref_codec(
            pin=value._codec_pin,
            annotated_codec=(
                _annotated_codec(declared_type)
                or _annotated_codec(_ref_inner_declared_type(declared_type))
            ),
            inner_value=inner_value,
            result_codecs=result_codecs,
            value_path=value_path,
        )
        dumped = _dump_artifact(
            inner_value,
            codec=codec,
            value_path=value_path,
            bundle_dir=bundle_dir,
            dump_state=dump_state,
        )
        dump_state.ref_rebinds.append(
            _RefRebind(
                ref=value,
                codec=codec,
                artifact_rel=dumped.artifact_rel,
                encoded_metadata=dumped.encoded_metadata,
            )
        )
        return dumped.wrapper

    if annotated_codec := _annotated_codec(declared_type):
        return _dump_eager_artifact(
            value,
            codec=annotated_codec,
            value_path=value_path,
            bundle_dir=bundle_dir,
            dump_state=dump_state,
        )

    match value:
        case None | bool() | int() | float() | str():
            return value
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
                return _dump_eager_artifact(
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


def _resolve_ref_codec(
    *,
    pin: type[Codec] | None,
    annotated_codec: type[Codec] | None,
    inner_value: object,
    result_codecs: tuple[type[Codec], ...],
    value_path: ValuePath,
) -> type[Codec]:
    if pin is not None and annotated_codec is not None and pin is not annotated_codec:
        raise TypeError(
            "Conflicting codecs: value was wrapped with furu.ref(..., codec=...), "
            "but the field also has an Annotated codec."
        )
    codec = pin or annotated_codec or CodecMeta.find_codec(inner_value, result_codecs)
    if codec is None:
        raise ValueError(
            f"Unsupported Ref value at {_value_path_display(value_path)}:\n"
            f"values of type {type(inner_value).__name__!r} resolve no codec. "
            "Pass an explicit codec to furu.ref(...)."
        )
    return codec


def _dump_eager_artifact(
    value: object,
    *,
    codec: type[Codec],
    value_path: ValuePath,
    bundle_dir: Path,
    dump_state: _DumpState,
) -> JsonValue:
    dumped = _dump_artifact(
        value,
        codec=codec,
        value_path=value_path,
        bundle_dir=bundle_dir,
        dump_state=dump_state,
    )
    if codec.reload_value_after_save:
        dump_state.should_reload_value_after_save = True
    return dumped.wrapper


def _dump_artifact(
    value: object,
    *,
    codec: type[Codec],
    value_path: ValuePath,
    bundle_dir: Path,
    dump_state: _DumpState,
) -> _ArtifactDump:
    artifact_rel = Path(ARTIFACTS_DIR_NAME, *(value_path or (_ROOT_ARTIFACT_NAME,)))
    artifact_dir = bundle_dir / artifact_rel
    artifact_dir.mkdir(parents=True, exist_ok=False)

    raw_metadata = codec().save(value, artifact_dir)
    if not isinstance(raw_metadata, Mapping):
        raise TypeError(
            f"{codec.__name__}.save() must return a mapping of metadata; "
            f"got {type(raw_metadata).__name__}"
        )
    encoded_metadata = _encode_codec_metadata(
        raw_metadata,
        data_dir=dump_state.data_dir,
        value_path=value_path,
    )

    body: dict[str, JsonValue] = {
        KINDMARKER: "artifact",
        "codec": codec._codec_id(),
        "path": artifact_rel.as_posix(),
    }
    if encoded_metadata:
        body[METADATA_KEY] = encoded_metadata
    return _ArtifactDump(
        wrapper={WRAPPER_KEY: body},
        artifact_rel=artifact_rel,
        encoded_metadata=encoded_metadata,
    )


def _encode_codec_metadata(
    node: object,
    *,
    data_dir: Path,
    value_path: ValuePath,
) -> JsonValue:
    match node:
        case Path():
            resolved = node.resolve()
            data_root = data_dir.resolve()
            if not resolved.is_relative_to(data_root):
                raise ValueError(
                    "codec metadata path escapes data dir at "
                    f"{_value_path_display(value_path)}: {node}"
                )
            return {DATAPATHMARKER: resolved.relative_to(data_root).as_posix()}
        case Mapping():
            return {
                str(key): _encode_codec_metadata(
                    child, data_dir=data_dir, value_path=value_path
                )
                for key, child in node.items()
            }
        case list() | tuple():
            return [
                _encode_codec_metadata(item, data_dir=data_dir, value_path=value_path)
                for item in node
            ]
        case None | bool() | int() | float() | str():
            return node
        case _:
            raise TypeError(
                "codec metadata may only contain JSON-native values and Paths; "
                f"got {type(node).__name__} at {_value_path_display(value_path)}"
            )


def _decode_codec_metadata(node: JsonValue, *, data_dir: Path) -> object:
    if isinstance(node, dict):
        rel = node.get(DATAPATHMARKER)
        if len(node) == 1 and isinstance(rel, str):
            rel_path = Path(rel)
            if rel_path.is_absolute():
                raise ValueError(f"codec metadata data path must be relative: {rel}")
            resolved = (data_dir.resolve() / rel_path).resolve()
            if not resolved.is_relative_to(data_dir.resolve()):
                raise ValueError(f"codec metadata data path escapes data dir: {rel}")
            return resolved
        return {
            key: _decode_codec_metadata(child, data_dir=data_dir)
            for key, child in node.items()
        }
    if isinstance(node, list):
        return [_decode_codec_metadata(item, data_dir=data_dir) for item in node]
    return node


def _load_value(
    node: JsonValue,
    *,
    declared_type: object = Any,
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


def _load_validated_fields(
    *,
    kind: str,
    cls: type[Any],
    expected: set[str],
    field_declared_types: dict[str, object],
    raw_fields: dict[str, JsonValue],
    bundle_dir: Path,
    data_dir: Path,
    value_path: ValuePath,
) -> dict[str, object]:
    actual = set(raw_fields)
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        return {
            name: _load_value(
                child,
                declared_type=field_declared_types.get(name, Any),
                bundle_dir=bundle_dir,
                data_dir=data_dir,
                value_path=(*value_path, name),
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
                    f"artifact wrapper directory missing: {artifact_dir}"
                )

            codec_id = body["codec"]
            codec = resolve_fully_qualified_name(codec_id)
            if not isinstance(codec, type) or not issubclass(codec, Codec):
                raise TypeError(f"{codec_id} is not a Codec")

            metadata = cast(
                Mapping[str, object],
                _decode_codec_metadata(body.get(METADATA_KEY, {}), data_dir=data_dir),
            )

            if _declares_ref(declared_type):
                return Ref._from_artifact(
                    codec=codec,
                    artifact_dir=artifact_dir,
                    metadata=metadata,
                )
            return codec().load(metadata, artifact_dir)
        case "dataclass":
            cls = resolve_fully_qualified_name(body[TYPEMARKER])
            if not dataclasses.is_dataclass(cls):
                raise ValueError(
                    f"Cannot load dataclass at {_value_path_display(value_path)}: "
                    f"{fully_qualified_name(cls)} is not a dataclass"
                )
            field_types = get_type_hints(cls, include_extras=True)
            dataclass_fields = dataclasses.fields(cls)
            init_fields = {field.name for field in dataclass_fields if field.init}
            loaded_fields = _load_validated_fields(
                kind="dataclass",
                cls=cls,
                expected={field.name for field in dataclass_fields},
                field_declared_types=field_types,
                raw_fields=body[FIELDSMARKER],
                bundle_dir=bundle_dir,
                data_dir=data_dir,
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
        case "pydantic":
            cls = resolve_fully_qualified_name(body[TYPEMARKER])
            if not issubclass(cls, pydantic.BaseModel):
                raise ValueError(
                    f"Cannot load pydantic model at {_value_path_display(value_path)}: "
                    f"{fully_qualified_name(cls)} is not a pydantic model"
                )
            field_types = get_type_hints(cls, include_extras=True)
            loaded_fields = _load_validated_fields(
                kind="pydantic model",
                cls=cls,
                expected=set(cls.model_fields),
                field_declared_types=field_types,
                raw_fields=body[FIELDSMARKER],
                bundle_dir=bundle_dir,
                data_dir=data_dir,
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


def _save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    result_codecs: tuple[type[Codec], ...],
    data_dir: Path,
) -> SaveResult:
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
    return SaveResult(
        should_reload_value_after_save=dump_state.should_reload_value_after_save,
        ref_rebinds=dump_state.ref_rebinds,
    )


def rebind_refs_to_published(
    save_result: SaveResult,
    *,
    result_dir: Path,
    data_dir: Path,
) -> None:
    """Point the in-memory result's value-backed refs at published storage."""
    for rebind in save_result.ref_rebinds:
        rebind.ref._rebind(
            codec=rebind.codec,
            artifact_dir=result_dir / rebind.artifact_rel,
            metadata=cast(
                Mapping[str, object],
                _decode_codec_metadata(rebind.encoded_metadata, data_dir=data_dir),
            ),
        )


def load_result_bundle(
    bundle_dir: Path,
    *,
    data_dir: Path,
    declared_type: object = Any,
) -> object:
    manifest_path = bundle_dir / MANIFEST_FILE_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _load_value(
        raw,
        declared_type=declared_type,
        bundle_dir=bundle_dir,
        data_dir=data_dir,
        value_path=(),
    )
