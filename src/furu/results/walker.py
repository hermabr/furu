from __future__ import annotations

import dataclasses
import importlib
import json
import math
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from furu.results.api import (
    ResultConfig,
    ResultRule,
    SaveWith,
    unwrap_value_directive,
)
from furu.results.errors import (
    ResultCodecError,
    ResultDeserializationError,
    ResultSerializationError,
)
from furu.results.lazy import LazyValue
from furu.results.nodes import (
    FURU_NODE_KEY,
    RESULT_FORMAT,
    is_wrapped_node,
    unwrap_node,
    wrap_node,
)
from furu.results.paths import (
    LogicalPath,
    artifact_path_for_logical_path,
    artifact_relpath_str,
    format_logical_path,
)
from furu.results.protocol import (
    DumpContext,
    LoadContext,
    supports_furu_result_protocol,
)
from furu.results.registry import ResultCodec
from furu.utils import JsonValue, _stable_json_dump, class_label, fully_qualified_name

_EXTERNAL_KIND = "external"
_OBJECT_KIND = "object"
_DATACLASS_KIND = "dataclass"
_PYDANTIC_KIND = "pydantic"
_TUPLE_KIND = "tuple"
_SET_KIND = "set"
_FROZENSET_KIND = "frozenset"
_MAPPING_KIND = "mapping"
_PATH_KIND = "path"
_JSON_TREE_CODEC_ID = "furu.json-tree.v1"


def save_result_bundle(value: Any, result_dir: Path, config: ResultConfig) -> None:
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    dumper = _BundleDumper(result_dir=result_dir, config=config)
    root = dumper.encode_root(value)
    manifest = {
        "format": RESULT_FORMAT,
        "root": root,
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def load_result_bundle(result_dir: Path, config: ResultConfig) -> Any:
    manifest_path = result_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultDeserializationError(
            f"Missing result manifest at {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ResultDeserializationError(
            f"Invalid JSON in result manifest at {manifest_path}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ResultDeserializationError(
            f"Expected result manifest object at {manifest_path}"
        )
    if manifest.get("format") != RESULT_FORMAT:
        raise ResultDeserializationError(
            f"Unsupported result format at {manifest_path}: {manifest.get('format')!r}"
        )
    if "root" not in manifest:
        raise ResultDeserializationError(f"Missing root node in {manifest_path}")

    loader = _BundleLoader(result_dir=result_dir, config=config)
    return loader.decode_root(cast(JsonValue, manifest["root"]))


def _dump_json_tree_with_new_encoder(value: Any, ctx: DumpContext) -> JsonValue:
    dumper = _BundleDumper(result_dir=ctx.bundle_dir, config=ctx.config)
    return dumper.encode_subtree(value, logical_path=ctx.logical_path)


def _load_json_tree_from_path(path: Path, ctx: LoadContext) -> Any:
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultDeserializationError(
            f"Missing JSON tree artifact at {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ResultDeserializationError(
            f"Invalid JSON tree artifact at {path}"
        ) from exc

    loader = _BundleLoader(result_dir=ctx.bundle_dir, config=ctx.config)
    return loader.decode_subtree(cast(JsonValue, tree), logical_path=ctx.logical_path)


class _Directive:
    __slots__ = ("codec", "lazy")

    def __init__(
        self,
        *,
        codec: str | ResultCodec | None = None,
        lazy: bool | None = None,
    ) -> None:
        self.codec = codec
        self.lazy = lazy

    def merged_with(self, other: _Directive | None) -> _Directive:
        if other is None:
            return _Directive(codec=self.codec, lazy=self.lazy)
        return _Directive(
            codec=self.codec if self.codec is not None else other.codec,
            lazy=self.lazy if self.lazy is not None else other.lazy,
        )


class _BundleDumper:
    def __init__(self, *, result_dir: Path, config: ResultConfig) -> None:
        self._result_dir = result_dir
        self._config = config
        self._active_ids: set[int] = set()

    def encode_root(self, value: Any) -> JsonValue:
        return self.encode_subtree(value, logical_path=())

    def encode_subtree(
        self,
        value: Any,
        *,
        logical_path: LogicalPath,
        field_hint: Any | None = None,
    ) -> JsonValue:
        directive, raw_value = self._unwrap_directives(value)
        path_directive = _resolve_path_rule(self._config.rules, logical_path)
        annotation_directive = _directive_from_hint(field_hint)
        effective = directive.merged_with(path_directive).merged_with(
            annotation_directive
        )

        if effective.codec is not None:
            return self._encode_external_with_codec(
                raw_value,
                logical_path=logical_path,
                codec_ref=effective.codec,
                lazy=bool(effective.lazy),
            )

        if supports_furu_result_protocol(raw_value):
            return self._encode_protocol_value(
                raw_value,
                logical_path=logical_path,
                lazy=bool(effective.lazy),
            )

        type_rule = _resolve_type_rule(self._config.rules, raw_value)
        lazy_requested = _first_not_none(effective.lazy, type_rule.lazy)
        codec_ref = _first_not_none(
            type_rule.codec,
            self._registry_codec_for_value(raw_value),
        )

        if codec_ref is not None:
            return self._encode_external_with_codec(
                raw_value,
                logical_path=logical_path,
                codec_ref=codec_ref,
                lazy=bool(lazy_requested),
            )

        return self._encode_structural(
            raw_value,
            logical_path=logical_path,
            field_hint=field_hint,
            lazy=bool(lazy_requested),
        )

    def _unwrap_directives(self, value: Any) -> tuple[_Directive, Any]:
        directive = _Directive()
        current = value
        while True:
            wrapped = unwrap_value_directive(current)
            if wrapped is not None:
                directive = _Directive(
                    codec=directive.codec
                    if directive.codec is not None
                    else wrapped.codec,
                    lazy=directive.lazy if directive.lazy is not None else wrapped.lazy,
                )
                current = wrapped.value
                continue
            if isinstance(current, LazyValue):
                directive = _Directive(
                    codec=directive.codec
                    if directive.codec is not None
                    else current._furu_requested_codec,
                    lazy=directive.lazy if directive.lazy is not None else True,
                )
                current = current._furu_unwrap_for_dump()
                continue
            return directive, current

    def _encode_external_with_codec(
        self,
        value: Any,
        *,
        logical_path: LogicalPath,
        codec_ref: str | ResultCodec,
        lazy: bool,
    ) -> JsonValue:
        codec = self._resolve_codec(codec_ref)
        if codec.codec_id == _JSON_TREE_CODEC_ID:
            return self._encode_json_tree_external(
                value, logical_path=logical_path, lazy=lazy
            )

        artifact_dir = artifact_path_for_logical_path(self._result_dir, logical_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ctx = DumpContext(
            bundle_dir=self._result_dir,
            artifact_dir=artifact_dir,
            logical_path=logical_path,
            registry=self._config.registry,
            config=self._config,
        )
        try:
            meta = codec.dump(value, ctx)
        except Exception as exc:
            if isinstance(exc, ResultCodecError):
                raise
            raise ResultCodecError(
                f"Codec {codec.codec_id} failed to save {format_logical_path(logical_path)}: {exc}"
            ) from exc
        return wrap_node(
            {
                "kind": _EXTERNAL_KIND,
                "codec": codec.codec_id,
                "path": artifact_relpath_str(logical_path),
                "lazy": lazy,
                "meta": cast(JsonValue, meta),
            }
        )

    def _encode_json_tree_external(
        self,
        value: Any,
        *,
        logical_path: LogicalPath,
        lazy: bool,
    ) -> JsonValue:
        artifact_dir = artifact_path_for_logical_path(self._result_dir, logical_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        encoded = self._encode_json_tree_payload(value, logical_path=logical_path)
        (artifact_dir / "data.json").write_text(
            json.dumps(encoded, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return wrap_node(
            {
                "kind": _EXTERNAL_KIND,
                "codec": _JSON_TREE_CODEC_ID,
                "path": artifact_relpath_str(logical_path),
                "lazy": lazy,
                "meta": None,
            }
        )

    def _encode_json_tree_payload(
        self,
        value: Any,
        *,
        logical_path: LogicalPath,
    ) -> JsonValue:
        directive, raw_value = self._unwrap_directives(value)
        if directive.codec is not None:
            return self._encode_external_with_codec(
                raw_value,
                logical_path=logical_path,
                codec_ref=directive.codec,
                lazy=bool(directive.lazy),
            )
        if supports_furu_result_protocol(raw_value):
            return self._encode_protocol_value(
                raw_value,
                logical_path=logical_path,
                lazy=bool(directive.lazy),
            )
        type_rule = _resolve_type_rule(self._config.rules, raw_value)
        codec_ref = _first_not_none(
            type_rule.codec,
            self._registry_codec_for_value(raw_value),
        )
        lazy_requested = _first_not_none(directive.lazy, type_rule.lazy)
        if codec_ref is not None:
            return self._encode_external_with_codec(
                raw_value,
                logical_path=logical_path,
                codec_ref=codec_ref,
                lazy=bool(lazy_requested),
            )
        return self._encode_structural(
            raw_value,
            logical_path=logical_path,
            field_hint=None,
            lazy=False,
        )

    def _encode_protocol_value(
        self,
        value: Any,
        *,
        logical_path: LogicalPath,
        lazy: bool,
    ) -> JsonValue:
        artifact_dir = artifact_path_for_logical_path(self._result_dir, logical_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ctx = DumpContext(
            bundle_dir=self._result_dir,
            artifact_dir=artifact_dir,
            logical_path=logical_path,
            registry=self._config.registry,
            config=self._config,
        )
        meta = value.__furu_result_dump__(ctx)
        return wrap_node(
            {
                "kind": _OBJECT_KIND,
                "type": fully_qualified_name(type(value)),
                "path": artifact_relpath_str(logical_path),
                "lazy": lazy,
                "meta": meta,
            }
        )

    def _encode_structural(
        self,
        value: Any,
        *,
        logical_path: LogicalPath,
        field_hint: Any | None,
        lazy: bool,
    ) -> JsonValue:
        if lazy:
            return self._encode_json_tree_external(
                value, logical_path=logical_path, lazy=True
            )

        if value is None or isinstance(value, (bool, int, str)):
            return cast(JsonValue, value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ResultSerializationError(
                    f"Cannot save non-finite float at {format_logical_path(logical_path)}: {value!r} is not valid JSON."
                )
            return value
        if isinstance(value, Path):
            return wrap_node({"kind": _PATH_KIND, "value": str(value)})

        with self._cycle_guard(value, logical_path):
            if isinstance(value, list):
                return [
                    self.encode_subtree(item, logical_path=logical_path + (index,))
                    for index, item in enumerate(value)
                ]
            if isinstance(value, tuple):
                return wrap_node(
                    {
                        "kind": _TUPLE_KIND,
                        "items": [
                            self.encode_subtree(
                                item, logical_path=logical_path + (index,)
                            )
                            for index, item in enumerate(value)
                        ],
                    }
                )
            if isinstance(value, set):
                return wrap_node(
                    {
                        "kind": _SET_KIND,
                        "items": _sorted_items(
                            [
                                self.encode_subtree(
                                    item, logical_path=logical_path + (index,)
                                )
                                for index, item in enumerate(value)
                            ]
                        ),
                    }
                )
            if isinstance(value, frozenset):
                return wrap_node(
                    {
                        "kind": _FROZENSET_KIND,
                        "items": _sorted_items(
                            [
                                self.encode_subtree(
                                    item, logical_path=logical_path + (index,)
                                )
                                for index, item in enumerate(value)
                            ]
                        ),
                    }
                )
            if isinstance(value, dict):
                return self._encode_mapping(value, logical_path=logical_path)
            if dataclasses.is_dataclass(value):
                return self._encode_dataclass(value, logical_path=logical_path)
            if isinstance(value, BaseModel):
                return self._encode_pydantic_model(value, logical_path=logical_path)

        raise ResultSerializationError(
            "Cannot save value at "
            f"{format_logical_path(logical_path)} of type {class_label(type(value))}.\n"
            "Register a ResultCodec, use furu.save_with(...), or implement __furu_result_dump__."
        )

    def _encode_mapping(
        self,
        value: Mapping[Any, Any],
        *,
        logical_path: LogicalPath,
    ) -> JsonValue:
        if all(isinstance(key, str) and key != FURU_NODE_KEY for key in value):
            return {
                key: self.encode_subtree(item, logical_path=logical_path + (key,))
                for key, item in value.items()
            }
        items: list[JsonValue] = []
        for index, (key, item) in enumerate(value.items()):
            items.append(
                [
                    self.encode_subtree(
                        key, logical_path=logical_path + (index, "key")
                    ),
                    self.encode_subtree(
                        item, logical_path=logical_path + (index, "value")
                    ),
                ]
            )
        return wrap_node({"kind": _MAPPING_KIND, "items": items})

    def _encode_dataclass(self, value: Any, *, logical_path: LogicalPath) -> JsonValue:
        hints = get_type_hints(type(value), include_extras=True)
        fields: dict[str, JsonValue] = {}
        for field in dataclasses.fields(value):
            fields[field.name] = self.encode_subtree(
                getattr(value, field.name),
                logical_path=logical_path + (field.name,),
                field_hint=hints.get(field.name),
            )
        return wrap_node(
            {
                "kind": _DATACLASS_KIND,
                "type": fully_qualified_name(type(value)),
                "fields": fields,
            }
        )

    def _encode_pydantic_model(
        self, value: BaseModel, *, logical_path: LogicalPath
    ) -> JsonValue:
        hints = get_type_hints(type(value), include_extras=True)
        fields: dict[str, JsonValue] = {}
        for field_name in type(value).model_fields:
            fields[field_name] = self.encode_subtree(
                getattr(value, field_name),
                logical_path=logical_path + (field_name,),
                field_hint=hints.get(field_name),
            )
        return wrap_node(
            {
                "kind": _PYDANTIC_KIND,
                "type": fully_qualified_name(type(value)),
                "fields": fields,
            }
        )

    def _registry_codec_for_value(self, value: Any) -> str | None:
        return self._config.registry.find_codec_id_for_value(value)

    def _resolve_codec(self, codec_ref: str | ResultCodec) -> ResultCodec:
        if isinstance(codec_ref, str):
            return self._config.registry.get_codec(codec_ref)
        return codec_ref

    @contextmanager
    def _cycle_guard(self, value: Any, logical_path: LogicalPath):
        value_id = id(value)
        if value_id in self._active_ids:
            raise ResultSerializationError(
                f"Cannot save cyclic result structure at {format_logical_path(logical_path)}. Furu result persistence does not preserve object identity."
            )
        self._active_ids.add(value_id)
        try:
            yield
        finally:
            self._active_ids.remove(value_id)


class _BundleLoader:
    def __init__(self, *, result_dir: Path, config: ResultConfig) -> None:
        self._result_dir = result_dir
        self._config = config

    def decode_root(self, value: JsonValue) -> Any:
        return self.decode_subtree(value, logical_path=())

    def decode_subtree(self, value: JsonValue, *, logical_path: LogicalPath) -> Any:
        if is_wrapped_node(value):
            node = unwrap_node(value)
            kind = node.get("kind")
            if kind == _EXTERNAL_KIND:
                return self._decode_external(node, logical_path=logical_path)
            if kind == _OBJECT_KIND:
                return self._decode_protocol_object(node, logical_path=logical_path)
            if kind == _DATACLASS_KIND:
                return self._decode_dataclass(node, logical_path=logical_path)
            if kind == _PYDANTIC_KIND:
                return self._decode_pydantic(node, logical_path=logical_path)
            if kind == _TUPLE_KIND:
                return tuple(
                    self.decode_subtree(item, logical_path=logical_path + (index,))
                    for index, item in enumerate(_expect_list(node, "items"))
                )
            if kind == _SET_KIND:
                return set(
                    self.decode_subtree(item, logical_path=logical_path + (index,))
                    for index, item in enumerate(_expect_list(node, "items"))
                )
            if kind == _FROZENSET_KIND:
                return frozenset(
                    self.decode_subtree(item, logical_path=logical_path + (index,))
                    for index, item in enumerate(_expect_list(node, "items"))
                )
            if kind == _MAPPING_KIND:
                out: dict[Any, Any] = {}
                for index, pair in enumerate(_expect_list(node, "items")):
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ResultDeserializationError(
                            f"Invalid mapping entry at {format_logical_path(logical_path)}"
                        )
                    key = self.decode_subtree(
                        pair[0], logical_path=logical_path + (index, "key")
                    )
                    item = self.decode_subtree(
                        pair[1], logical_path=logical_path + (index, "value")
                    )
                    out[key] = item
                return out
            if kind == _PATH_KIND:
                raw_value = node.get("value")
                if not isinstance(raw_value, str):
                    raise ResultDeserializationError(
                        f"Invalid path node at {format_logical_path(logical_path)}"
                    )
                return Path(raw_value)
            raise ResultDeserializationError(
                f"Unknown result node kind at {format_logical_path(logical_path)}: {kind!r}"
            )
        if isinstance(value, list):
            return [
                self.decode_subtree(item, logical_path=logical_path + (index,))
                for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            return {
                key: self.decode_subtree(item, logical_path=logical_path + (key,))
                for key, item in value.items()
            }
        return value

    def _decode_external(
        self, node: dict[str, JsonValue], *, logical_path: LogicalPath
    ) -> Any:
        codec_id = _expect_str(node, "codec")
        artifact_dir = self._result_dir / _expect_str(node, "path")
        lazy = _expect_bool(node, "lazy")
        meta = cast(JsonValue | None, node.get("meta"))

        def load_value() -> Any:
            ctx = LoadContext(
                bundle_dir=self._result_dir,
                artifact_dir=artifact_dir,
                logical_path=logical_path,
                registry=self._config.registry,
                config=self._config,
            )
            if codec_id == _JSON_TREE_CODEC_ID:
                return _load_json_tree_from_path(artifact_dir / "data.json", ctx)
            codec = self._config.registry.get_codec(codec_id)
            try:
                return codec.load(ctx, meta)
            except Exception as exc:
                if isinstance(exc, ResultCodecError):
                    raise
                raise ResultCodecError(
                    f"Codec {codec_id} failed to load {format_logical_path(logical_path)}: {exc}"
                ) from exc

        if lazy:
            return LazyValue.from_loader(load_value, meta=meta)
        return load_value()

    def _decode_protocol_object(
        self,
        node: dict[str, JsonValue],
        *,
        logical_path: LogicalPath,
    ) -> Any:
        type_name = _expect_str(node, "type")
        artifact_dir = self._result_dir / _expect_str(node, "path")
        lazy = _expect_bool(node, "lazy")
        meta = node.get("meta")
        cls = _import_type(type_name)

        def load_value() -> Any:
            ctx = LoadContext(
                bundle_dir=self._result_dir,
                artifact_dir=artifact_dir,
                logical_path=logical_path,
                registry=self._config.registry,
                config=self._config,
            )
            return cls.__furu_result_load__(ctx, meta)

        if lazy:
            return LazyValue.from_loader(load_value, meta=meta)
        return load_value()

    def _decode_dataclass(
        self,
        node: dict[str, JsonValue],
        *,
        logical_path: LogicalPath,
    ) -> Any:
        cls = _import_type(_expect_str(node, "type"))
        raw_fields = _expect_dict(node, "fields")
        instance = object.__new__(cls)
        for field_name, field_value in raw_fields.items():
            object.__setattr__(
                instance,
                field_name,
                self.decode_subtree(
                    field_value,
                    logical_path=logical_path + (field_name,),
                ),
            )
        return instance

    def _decode_pydantic(
        self,
        node: dict[str, JsonValue],
        *,
        logical_path: LogicalPath,
    ) -> Any:
        cls = _import_type(_expect_str(node, "type"))
        raw_fields = _expect_dict(node, "fields")
        fields = {
            field_name: self.decode_subtree(
                field_value,
                logical_path=logical_path + (field_name,),
            )
            for field_name, field_value in raw_fields.items()
        }
        return cls.model_construct(**fields)


def _directive_from_hint(hint: Any | None) -> _Directive:
    if hint is None:
        return _Directive()
    if get_origin(hint) is not Annotated:
        return _Directive()
    args = get_args(hint)
    for metadata in args[1:]:
        if isinstance(metadata, SaveWith):
            return _Directive(codec=metadata.codec)
    return _Directive()


def _resolve_path_rule(rules: tuple[ResultRule, ...], path: LogicalPath) -> _Directive:
    directive = _Directive()
    for rule in rules:
        if rule.match_kind != "path" or rule.path != path:
            continue
        if rule.codec is not None:
            directive.codec = rule.codec
        if rule.lazy is not None:
            directive.lazy = rule.lazy
    return directive


def _resolve_type_rule(rules: tuple[ResultRule, ...], value: Any) -> _Directive:
    directive = _Directive()
    for rule in rules:
        if rule.match_kind != "type" or rule.value_type is None:
            continue
        if isinstance(value, rule.value_type):
            if rule.codec is not None:
                directive.codec = rule.codec
            if rule.lazy is not None:
                directive.lazy = rule.lazy
    return directive


def _sorted_items(items: list[JsonValue]) -> list[JsonValue]:
    return sorted(items, key=lambda item: _stable_json_dump(item))


def _first_not_none[T](first: T | None, second: T | None) -> T | None:
    return first if first is not None else second


def _import_type(type_name: str) -> type[Any]:
    module_name, _, qualname = type_name.rpartition(".")
    if not module_name:
        raise ResultDeserializationError(f"Invalid type reference: {type_name}")
    module = importlib.import_module(module_name)
    cls = getattr(module, qualname)
    if not isinstance(cls, type):
        raise ResultDeserializationError(
            f"Type reference did not resolve to a class: {type_name}"
        )
    return cls


def _expect_str(node: dict[str, JsonValue], key: str) -> str:
    value = node.get(key)
    if not isinstance(value, str):
        raise ResultDeserializationError(
            f"Expected string field {key!r} in result node"
        )
    return value


def _expect_bool(node: dict[str, JsonValue], key: str) -> bool:
    value = node.get(key)
    if not isinstance(value, bool):
        raise ResultDeserializationError(
            f"Expected boolean field {key!r} in result node"
        )
    return value


def _expect_list(node: dict[str, JsonValue], key: str) -> list[JsonValue]:
    value = node.get(key)
    if not isinstance(value, list):
        raise ResultDeserializationError(f"Expected list field {key!r} in result node")
    return cast(list[JsonValue], value)


def _expect_dict(node: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = node.get(key)
    if not isinstance(value, dict):
        raise ResultDeserializationError(
            f"Expected object field {key!r} in result node"
        )
    return cast(dict[str, JsonValue], value)
