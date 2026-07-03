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
from furu.constants import FIELDSMARKER, KINDMARKER, TYPEMARKER
from furu.result.codec import Codec, CodecMeta
from furu.result.ref import Ref
from furu.utils import JsonValue, fully_qualified_name, resolve_fully_qualified_name

WRAPPER_KEY: Final = "$furu"
ARTIFACTS_DIR_NAME: Final = "artifacts"
MANIFEST_FILE_NAME: Final = "manifest.json"
_ROOT_ARTIFACT_NAME: Final = "root"
ValuePath: TypeAlias = tuple[str, ...]
WrapperKind: TypeAlias = Literal[
    "artifact", "dataclass", "path", "pydantic", "tuple", "set", "frozenset"
]


@dataclasses.dataclass
class _DumpState:
    data_dir: Path
    should_reload_value_after_save: bool = False


def _value_path_display(value_path: ValuePath) -> str:
    if not value_path:
        return "<root>"
    return "/".join(value_path)


def _annotated_codec(declared_type: object) -> type[Codec] | None:
    if get_origin(declared_type) is not Annotated:
        return None
    for item in get_args(declared_type)[1:]:
        if isinstance(item, type) and issubclass(item, Codec):
            return item
    return None


def _is_ref_type(declared_type: object) -> bool:
    declared_type = strip_annotated(declared_type)
    return declared_type is Ref or get_origin(declared_type) is Ref


def _ref_value_type(declared_type: object) -> object:
    declared_type = strip_annotated(declared_type)
    return (
        get_args(declared_type)[0]
        if get_origin(declared_type) is Ref and get_args(declared_type)
        else Any
    )


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


def _metadata_path_to_json(path: Path, *, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"codec metadata path must live inside data dir: {path}"
        ) from exc


def _metadata_path_from_json(path: str, *, data_dir: Path) -> Path:
    rel_path = Path(path)
    if rel_path.is_absolute():
        raise ValueError(f"data-dir codec path must be relative: {rel_path}")
    resolved_path = (data_dir.resolve() / rel_path).resolve()
    if not resolved_path.is_relative_to(data_dir.resolve()):
        raise ValueError(f"data-dir codec path escapes data dir: {rel_path}")
    return resolved_path


def _dump_codec_metadata(value: object, *, data_dir: Path) -> JsonValue:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Path():
            return {
                WRAPPER_KEY: {
                    KINDMARKER: "data_path",
                    "value": _metadata_path_to_json(value, data_dir=data_dir),
                }
            }
        case list() | tuple():
            return [_dump_codec_metadata(item, data_dir=data_dir) for item in value]
        case dict():
            return {
                _validate_result_path_segment(
                    key, parent_value_path=("metadata",)
                ): _dump_codec_metadata(
                    child,
                    data_dir=data_dir,
                )
                for key, child in value.items()
            }
        case _:
            raise TypeError(
                f"Unsupported codec metadata value {value!r} of type {type(value).__name__!r}"
            )


def _load_codec_metadata(value: JsonValue, *, data_dir: Path) -> object:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case list():
            return [_load_codec_metadata(item, data_dir=data_dir) for item in value]
        case dict() if value.get(WRAPPER_KEY, {}).get(KINDMARKER) == "data_path":
            return _metadata_path_from_json(
                cast(str, value[WRAPPER_KEY]["value"]), data_dir=data_dir
            )
        case dict():
            return {
                key: _load_codec_metadata(child, data_dir=data_dir)
                for key, child in value.items()
            }
        case _:
            assert_never(value)


def _load_codec_metadata_mapping(
    value: JsonValue,
    *,
    data_dir: Path,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("codec metadata must be a JSON object")
    return cast(dict[str, object], _load_codec_metadata(value, data_dir=data_dir))


def _dump_value(
    value: object,
    *,
    declared_type: object = Any,
    value_path: ValuePath,
    bundle_dir: Path,
    result_codecs: tuple[type[Codec], ...],
    dump_state: _DumpState,
) -> JsonValue:
    storage_declared_type = (
        _ref_value_type(declared_type) if isinstance(value, Ref) else declared_type
    )
    annotated_codec = _annotated_codec(storage_declared_type)

    if isinstance(value, Ref):
        if annotated_codec is not None and value._codec is not annotated_codec:
            raise TypeError(
                "Conflicting codecs: value was wrapped with furu.ref(...), "
                "but the field also has an Annotated codec."
            )
        return _dump_artifact(
            value.load(),
            codec=value._codec,
            value_path=value_path,
            bundle_dir=bundle_dir,
            dump_state=dump_state,
        )

    if annotated_codec is not None:
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
) -> JsonValue:
    artifact_rel = Path(ARTIFACTS_DIR_NAME, *(value_path or (_ROOT_ARTIFACT_NAME,)))
    artifact_dir = bundle_dir / artifact_rel
    artifact_dir.mkdir(parents=True, exist_ok=False)

    metadata = _dump_codec_metadata(
        dict(codec().save(value, artifact_dir)),
        data_dir=dump_state.data_dir,
    )
    if codec.reload_value_after_save:
        dump_state.should_reload_value_after_save = True
    return {
        WRAPPER_KEY: {
            KINDMARKER: "artifact",
            "codec": codec._codec_id(),
            "path": artifact_rel.as_posix(),
            "metadata": metadata,
        }
    }


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
    raw_fields: dict[str, JsonValue],
    field_types: Mapping[str, object],
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
                declared_type=field_types.get(name, Any),
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


def _artifact_from_wrapper(
    body: dict[str, Any],
    *,
    bundle_dir: Path,
    data_dir: Path,
) -> tuple[type[Codec], Path, dict[str, object]]:
    artifact_rel = Path(body["path"])
    if artifact_rel.is_absolute():
        raise ValueError(f"artifact wrapper path must be relative: {artifact_rel}")

    artifact_dir = (bundle_dir / artifact_rel).resolve()
    artifacts_root = (bundle_dir / ARTIFACTS_DIR_NAME).resolve()
    if not artifact_dir.is_relative_to(artifacts_root):
        raise ValueError(
            f"artifact wrapper path escapes bundle artifacts dir: {artifact_rel}"
        )

    if not artifact_dir.exists():
        raise ValueError(f"artifact wrapper artifact directory missing: {artifact_dir}")

    codec_id = body["codec"]
    codec = resolve_fully_qualified_name(codec_id)
    if not isinstance(codec, type) or not issubclass(codec, Codec):
        raise TypeError(f"{codec_id} is not a Codec")
    return (
        codec,
        artifact_dir,
        _load_codec_metadata_mapping(body.get("metadata", {}), data_dir=data_dir),
    )


def _load_wrapper(
    body: dict[str, Any],
    *,
    declared_type: object = Any,
    bundle_dir: Path,
    data_dir: Path,
    value_path: ValuePath,
) -> object:
    kind: WrapperKind = body[KINDMARKER]
    match kind:
        case "artifact":
            codec, artifact_dir, metadata = _artifact_from_wrapper(
                body,
                bundle_dir=bundle_dir,
                data_dir=data_dir,
            )
            if _is_ref_type(declared_type):
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
            dataclass_fields = dataclasses.fields(cls)
            init_fields = {field.name for field in dataclass_fields if field.init}
            loaded_fields = _load_validated_fields(
                kind="dataclass",
                cls=cls,
                expected={field.name for field in dataclass_fields},
                raw_fields=body[FIELDSMARKER],
                field_types=get_type_hints(cls, include_extras=True),
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
            loaded_fields = _load_validated_fields(
                kind="pydantic model",
                cls=cls,
                expected=set(cls.model_fields),
                raw_fields=body[FIELDSMARKER],
                field_types=get_type_hints(cls, include_extras=True),
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


def _rebind_saved_result(
    value: object,
    *,
    bundle_dir: Path,
    data_dir: Path,
    declared_type: object = Any,
) -> object:
    manifest = json.loads((bundle_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    return _rebind_saved_value(
        value,
        manifest,
        declared_type=declared_type,
        bundle_dir=bundle_dir,
        data_dir=data_dir,
    )


def _rebind_saved_value(
    value: object,
    node: JsonValue,
    *,
    declared_type: object,
    bundle_dir: Path,
    data_dir: Path,
) -> object:
    if isinstance(node, dict) and WRAPPER_KEY in node:
        body = cast(dict[str, Any], node[WRAPPER_KEY])
        kind = body[KINDMARKER]
        if kind == "artifact":
            codec, artifact_dir, metadata = _artifact_from_wrapper(
                body,
                bundle_dir=bundle_dir,
                data_dir=data_dir,
            )
            if isinstance(value, Ref):
                value._rebind(artifact_dir=artifact_dir, metadata=metadata)
                return value
            if codec.reload_value_after_save:
                return codec().load(metadata, artifact_dir)
            return value
        if kind in {"dataclass", "pydantic"}:
            node = cast(JsonValue, body[FIELDSMARKER])
        elif kind == "tuple":
            node = cast(JsonValue, body["items"])

    if isinstance(value, list) and isinstance(node, list):
        items = cast(list[Any], value)
        for i, child_node in enumerate(node):
            items[i] = _rebind_saved_value(
                items[i],
                child_node,
                declared_type=child_declared_type(declared_type, i),
                bundle_dir=bundle_dir,
                data_dir=data_dir,
            )
        return value
    if isinstance(value, tuple) and isinstance(node, list):
        return tuple(
            _rebind_saved_value(
                item,
                child_node,
                declared_type=child_declared_type(declared_type, i),
                bundle_dir=bundle_dir,
                data_dir=data_dir,
            )
            for i, (item, child_node) in enumerate(zip(value, node, strict=True))
        )
    if isinstance(value, dict) and isinstance(node, dict):
        mapping = cast(dict[str, Any], value)
        for key, child_node in node.items():
            mapping[key] = _rebind_saved_value(
                mapping[key],
                child_node,
                declared_type=child_declared_type(declared_type, key),
                bundle_dir=bundle_dir,
                data_dir=data_dir,
            )
        return value
    if isinstance(value, pydantic.BaseModel) and isinstance(node, dict):
        changes = _rebind_field_changes(
            value, node, data_dir=data_dir, bundle_dir=bundle_dir
        )
        return value.model_copy(update=changes) if changes else value
    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and isinstance(node, dict)
    ):
        changes = _rebind_field_changes(
            value, node, data_dir=data_dir, bundle_dir=bundle_dir
        )
        return dataclasses.replace(value, **changes) if changes else value
    return value


def _rebind_field_changes(
    value: object,
    fields_node: dict[str, JsonValue],
    *,
    bundle_dir: Path,
    data_dir: Path,
) -> dict[str, object]:
    changes: dict[str, object] = {}
    field_types = get_type_hints(type(value), include_extras=True)
    for name, child_node in fields_node.items():
        current = getattr(value, name)
        rebound = _rebind_saved_value(
            current,
            child_node,
            declared_type=field_types.get(name, Any),
            bundle_dir=bundle_dir,
            data_dir=data_dir,
        )
        if rebound is not current:
            changes[name] = rebound
    return changes


def _save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    result_codecs: tuple[type[Codec], ...],
    data_dir: Path,
) -> bool:
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
    return dump_state.should_reload_value_after_save


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
