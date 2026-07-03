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
_CODEC_METADATA_PATH_KIND: Final = "data-path"
ValuePath: TypeAlias = tuple[str, ...]
WrapperKind: TypeAlias = Literal[
    "artifact", "dataclass", "path", "pydantic", "tuple", "set", "frozenset"
]


@dataclasses.dataclass(frozen=True)
class _RefRebind:
    ref: Ref[Any]
    codec: type[Codec]
    metadata: Mapping[str, object]
    artifact_rel: Path


@dataclasses.dataclass(frozen=True)
class SavedResultBundle:
    should_reload_value_after_save: bool
    _ref_rebinds: tuple[_RefRebind, ...]

    def __bool__(self) -> bool:
        return self.should_reload_value_after_save

    def rebind_refs(self, bundle_dir: Path) -> None:
        for rebind in self._ref_rebinds:
            rebind.ref._bind_to_artifact(
                codec=rebind.codec,
                metadata=rebind.metadata,
                artifact_dir=(bundle_dir / rebind.artifact_rel).resolve(),
            )


@dataclasses.dataclass
class _DumpState:
    data_dir: Path
    should_reload_value_after_save: bool = False
    ref_rebinds: list[_RefRebind] = dataclasses.field(default_factory=list)


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
    if get_origin(declared_type) is not Annotated:
        return None
    for item in get_args(declared_type)[1:]:
        if isinstance(item, type) and issubclass(item, Codec):
            return item
    return None


def _ref_inner_declared_type(declared_type: object) -> object | None:
    declared_type = strip_annotated(declared_type)
    if get_origin(declared_type) is not Ref:
        return None
    args = get_args(declared_type)
    if not args:
        return Any
    return args[0]


def _declared_codec_for_value(declared_type: object) -> type[Codec] | None:
    if ref_inner_type := _ref_inner_declared_type(declared_type):
        return _annotated_codec(ref_inner_type) or _annotated_codec(declared_type)
    return _annotated_codec(declared_type)


def _dump_value(
    value: object,
    *,
    declared_type: object = Any,
    value_path: ValuePath,
    bundle_dir: Path,
    result_codecs: tuple[type[Codec], ...],
    dump_state: _DumpState,
) -> JsonValue:
    ref_inner_type = _ref_inner_declared_type(declared_type)
    if ref_inner_type is not None:
        if not isinstance(value, Ref):
            raise TypeError(
                f"Unsupported result value at {_value_path_display(value_path)}:\n"
                f"fields declared as furu.Ref[...] must be populated with furu.ref(...); "
                f"got {type(value).__name__!r}."
            )
        return _dump_ref(
            value,
            declared_inner_type=ref_inner_type,
            value_path=value_path,
            bundle_dir=bundle_dir,
            dump_state=dump_state,
        )

    if isinstance(value, Ref):
        return _dump_ref(
            value,
            declared_inner_type=Any,
            value_path=value_path,
            bundle_dir=bundle_dir,
            dump_state=dump_state,
        )

    if annotated_codec := _declared_codec_for_value(declared_type):
        return _dump_artifact(
            value,
            codec=annotated_codec,
            value_path=value_path,
            bundle_dir=bundle_dir,
            dump_state=dump_state,
            ref_value=None,
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
                    getattr(value, field.name),
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
                    ref_value=None,
                )

    raise ValueError(
        f"Unsupported result value at {_value_path_display(value_path)}:\n"
        f"values of type {type(value).__name__!r} are not supported by furu. Add a custom codec"
    )


def _dump_ref(
    value: Ref[Any],
    *,
    declared_inner_type: object,
    value_path: ValuePath,
    bundle_dir: Path,
    dump_state: _DumpState,
) -> JsonValue:
    annotated_codec = _declared_codec_for_value(declared_inner_type)
    codec = (
        value._codec
        if value._codec_was_explicit or annotated_codec is None
        else annotated_codec
    )
    return _dump_artifact(
        value._value_for_save(),
        codec=codec,
        value_path=value_path,
        bundle_dir=bundle_dir,
        dump_state=dump_state,
        ref_value=value,
    )


def _dump_artifact(
    value: object,
    *,
    codec: type[Codec],
    value_path: ValuePath,
    bundle_dir: Path,
    dump_state: _DumpState,
    ref_value: Ref[Any] | None,
) -> JsonValue:
    artifact_rel = Path(ARTIFACTS_DIR_NAME, *(value_path or (_ROOT_ARTIFACT_NAME,)))
    artifact_dir = bundle_dir / artifact_rel
    artifact_dir.mkdir(parents=True, exist_ok=False)

    raw_metadata = codec().save(value, artifact_dir)
    if not isinstance(raw_metadata, Mapping):
        raise TypeError(
            f"{codec.__name__}.save() must return a metadata mapping, "
            f"got {type(raw_metadata).__name__}"
        )
    metadata = cast(
        dict[str, JsonValue],
        _dump_codec_metadata(
            raw_metadata,
            data_dir=dump_state.data_dir,
            value_path=value_path,
        ),
    )

    if ref_value is None:
        if codec.reload_value_after_save:
            dump_state.should_reload_value_after_save = True
    else:
        dump_state.ref_rebinds.append(
            _RefRebind(
                ref=ref_value,
                codec=codec,
                metadata=cast(
                    Mapping[str, object],
                    _load_codec_metadata(metadata, data_dir=dump_state.data_dir),
                ),
                artifact_rel=artifact_rel,
            )
        )

    return {
        WRAPPER_KEY: {
            KINDMARKER: "artifact",
            "codec": codec._codec_id(),
            "path": artifact_rel.as_posix(),
            "metadata": metadata,
        }
    }


def _dump_codec_metadata(
    value: object,
    *,
    data_dir: Path,
    value_path: ValuePath,
) -> JsonValue:
    match value:
        case None | bool() | int() | float() | str():
            return value
        case Path():
            data_dir_resolved = data_dir.resolve()
            value_resolved = value.resolve()
            try:
                rel_path = value_resolved.relative_to(data_dir_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Codec metadata path at {_value_path_display(value_path)} "
                    f"must live inside the data dir: {value}"
                ) from exc
            return {
                WRAPPER_KEY: {
                    KINDMARKER: _CODEC_METADATA_PATH_KIND,
                    "path": rel_path.as_posix(),
                }
            }
        case list() | tuple():
            return [
                _dump_codec_metadata(
                    item,
                    data_dir=data_dir,
                    value_path=(*value_path, str(i)),
                )
                for i, item in enumerate(value)
            ]
        case Mapping():
            out: dict[str, JsonValue] = {}
            for raw_key, child in value.items():
                key = _validate_result_path_segment(
                    raw_key,
                    parent_value_path=value_path,
                )
                out[key] = _dump_codec_metadata(
                    child,
                    data_dir=data_dir,
                    value_path=(*value_path, key),
                )
            return out
        case _:
            raise TypeError(
                f"Codec metadata at {_value_path_display(value_path)} contains "
                f"unsupported value {type(value).__name__!r}"
            )


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


def _load_validated_fields(
    *,
    kind: str,
    cls: type[Any],
    expected: set[str],
    field_types: Mapping[str, object],
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
            artifact_dir = _artifact_dir_from_body(body, bundle_dir=bundle_dir)
            codec_id = body["codec"]
            codec = resolve_fully_qualified_name(codec_id)
            if not isinstance(codec, type) or not issubclass(codec, Codec):
                raise TypeError(f"{codec_id} is not a Codec")

            metadata = cast(
                Mapping[str, object],
                _load_codec_metadata(body.get("metadata", {}), data_dir=data_dir),
            )
            if _ref_inner_declared_type(declared_type) is not None:
                return Ref._from_artifact(
                    codec=codec,
                    metadata=metadata,
                    artifact_dir=artifact_dir,
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
                field_types=get_type_hints(cls, include_extras=True),
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
            loaded_fields = _load_validated_fields(
                kind="pydantic model",
                cls=cls,
                expected=set(cls.model_fields),
                field_types=get_type_hints(cls, include_extras=True),
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


def _artifact_dir_from_body(body: Mapping[str, Any], *, bundle_dir: Path) -> Path:
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
    return artifact_dir


def _load_codec_metadata(value: object, *, data_dir: Path) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_load_codec_metadata(item, data_dir=data_dir) for item in value]
    if isinstance(value, dict):
        value_dict = cast(dict[str, object], value)
        if WRAPPER_KEY in value_dict:
            body = cast(dict[str, Any], value_dict[WRAPPER_KEY])
            if body.get(KINDMARKER) != _CODEC_METADATA_PATH_KIND:
                raise ValueError(f"unknown codec metadata wrapper: {body!r}")
            rel_path = Path(body["path"])
            if rel_path.is_absolute():
                raise ValueError(f"data-dir codec path must be relative: {rel_path}")

            data_dir_resolved = data_dir.resolve()
            resolved_path = (data_dir_resolved / rel_path).resolve()
            if not resolved_path.is_relative_to(data_dir_resolved):
                raise ValueError(f"data-dir codec path escapes data dir: {rel_path}")
            return resolved_path
        return {
            key: _load_codec_metadata(child, data_dir=data_dir)
            for key, child in value_dict.items()
        }
    raise TypeError(
        f"Codec metadata contains unsupported value {type(value).__name__!r}"
    )


def _save_result_bundle(
    value: object,
    bundle_dir: Path,
    *,
    declared_type: object = Any,
    result_codecs: tuple[type[Codec], ...],
    data_dir: Path,
) -> SavedResultBundle:
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
    return SavedResultBundle(
        should_reload_value_after_save=dump_state.should_reload_value_after_save,
        _ref_rebinds=tuple(dump_state.ref_rebinds),
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
