from __future__ import annotations

import dataclasses
import datetime as _dt
import inspect
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import chz
from chz.util import MISSING as CHZ_MISSING, MISSING_TYPE
from chz.validators import is_subtype_instance, type_repr

from .aliases import collect_aliases
from .config import FURU_CONFIG
from .schema import (
    schema_key_from_cls,
    schema_key_from_furu_obj,
    schema_key_from_metadata_raw,
)
from .serialization import FuruSerializer
from .serialization.serializer import JsonValue
from .storage import MetadataManager, MigrationManager, MigrationRecord, StateManager
from .storage.migration import RootKind
from .storage.state import _StateResultMigrated


class _FuruClass(Protocol):
    version_controlled: bool

    @classmethod
    def _namespace(cls) -> Path: ...


class _FuruObject(Protocol):
    furu_hash: str
    version_controlled: bool

    @classmethod
    def _namespace(cls) -> Path: ...


MigrationConflict = Literal["throw", "skip"]
SourceObj = TypeVar("SourceObj")
TargetObj = TypeVar("TargetObj", bound=_FuruObject)

# Runtime type expression from annotations (e.g. concrete classes, unions,
# deferred/forward-referenced strings, and other typing constructs).
RuntimeTypeAnnotation: TypeAlias = Any


@dataclass(frozen=True)
class _MigrationSkipToken:
    pass


TransformSkip: TypeAlias = _MigrationSkipToken
MigrationSkipped: TypeAlias = Literal["skipped"]

MIGRATION_SKIP: Final[TransformSkip] = _MigrationSkipToken()
MIGRATION_SKIPPED: Final[MigrationSkipped] = "skipped"


@dataclass(frozen=True)
class FuruRef:
    namespace: str
    furu_hash: str
    root: RootKind
    furu_dir: Path

    @overload
    def migrate(
        self,
        transform: Callable[[SourceObj], TargetObj],
        *,
        dry_run: bool = False,
        strict_types: bool = True,
        origin: str | None = None,
        note: str | None = None,
    ) -> TargetObj: ...

    @overload
    def migrate(
        self,
        transform: Callable[[SourceObj], TransformSkip],
        *,
        dry_run: bool = False,
        strict_types: bool = True,
        origin: str | None = None,
        note: str | None = None,
    ) -> MigrationSkipped: ...

    @overload
    def migrate(
        self,
        transform: Callable[[SourceObj], TargetObj | TransformSkip],
        *,
        dry_run: bool = False,
        strict_types: bool = True,
        origin: str | None = None,
        note: str | None = None,
    ) -> TargetObj | MigrationSkipped: ...

    def migrate(
        self,
        transform: Callable[[SourceObj], TargetObj | TransformSkip],
        *,
        dry_run: bool = False,
        strict_types: bool = True,
        origin: str | None = None,
        note: str | None = None,
    ) -> TargetObj | MigrationSkipped:
        source_obj = cast(SourceObj, self._load_ref_object())
        transformed = transform(source_obj)
        if isinstance(transformed, _MigrationSkipToken):
            original_ref = resolve_original_ref(self)
            _ensure_original_success(original_ref)
            return MIGRATION_SKIPPED

        validated_target = _assert_furu_object(transformed)
        if strict_types:
            validated_target = _coerce_furu_object_types(validated_target)
            _validate_furu_object_types(validated_target)
        self._migrate_to_target(
            target_obj=validated_target,
            dry_run=dry_run,
            origin=origin,
            note=note,
        )
        return cast(TargetObj, validated_target)

    def _load_ref_object(self) -> JsonValue:
        metadata = MetadataManager.read_metadata(self.furu_dir)
        return FuruSerializer.from_dict(metadata.furu_obj, strict=False)

    def _migrate_to_target(
        self,
        *,
        target_obj: _FuruObject,
        dry_run: bool,
        origin: str | None,
        note: str | None,
    ) -> None:
        original_ref = resolve_original_ref(self)
        _ensure_original_success(original_ref)

        alias_index = collect_aliases(include_inactive=True)
        alias_schema_cache: dict[Path, tuple[str, ...]] = {}

        target_hash = FuruSerializer.compute_hash(target_obj)
        target_ref = _target_ref(cast(_FuruClass, target_obj.__class__), target_hash)
        target_obj_dict = FuruSerializer.to_dict(target_obj)
        if not isinstance(target_obj_dict, dict):
            raise TypeError("migration: target object must serialize to a dict")
        target_schema_key = schema_key_from_furu_obj(target_obj_dict)

        alias_key = (original_ref.namespace, original_ref.furu_hash, original_ref.root)
        if _alias_schema_conflict(
            alias_index,
            alias_schema_cache,
            alias_key,
            target_schema_key,
        ):
            raise ValueError("migration: alias schema already exists for original")

        if dry_run:
            if target_ref.furu_dir.exists():
                raise ValueError("migration: target already exists")
            return

        _write_alias(
            target_obj=cast(JsonValue, target_obj),
            original_ref=original_ref,
            target_ref=target_ref,
            source_ref=self,
            skips=[],
            conflict="throw",
            origin=origin,
            note=note,
        )


@dataclass(frozen=True)
class MigrationSkip:
    source: FuruRef
    reason: str


@dataclass(frozen=True)
class MigrationReport:
    records: list[MigrationRecord]
    skips: list[MigrationSkip]


@dataclass(frozen=True)
class _Source:
    ref: FuruRef
    furu_obj: dict[str, JsonValue]
    migration: MigrationRecord | None


def migrate(
    cls: _FuruClass,
    *,
    from_schema: Iterable[str] | None = None,
    from_drop: Iterable[str] | None = None,
    from_add: Iterable[str] | None = None,
    from_hash: str | None = None,
    from_furu_obj: dict[str, JsonValue] | None = None,
    from_namespace: str | None = None,
    default_field: Iterable[str] | None = None,
    set_field: Mapping[str, JsonValue] | None = None,
    drop_field: Iterable[str] | None = None,
    rename_field: Mapping[str, str] | None = None,
    include_alias_sources: bool = False,
    conflict: MigrationConflict = "throw",
    origin: str | None = None,
    note: str | None = None,
) -> MigrationReport:
    """
    Migrate stored objects to the current schema via alias-only records.

    Notes:
        set_field only adds fields that are not already present. To replace a
        field, drop it first and then set it.
    """
    selector_count = sum(
        1
        for value in (
            from_schema is not None,
            from_hash is not None,
            from_furu_obj is not None,
            from_drop is not None or from_add is not None,
        )
        if value
    )
    if selector_count == 0:
        raise ValueError(
            "migration: provide one of from_schema, from_hash, from_furu_obj, or from_drop/from_add"
        )
    if selector_count > 1:
        raise ValueError("migration: source selection is ambiguous")

    namespace = from_namespace or _namespace_str(cls)

    if from_schema is not None:
        from_schema_key = _normalize_schema(from_schema)
        sources = _sources_from_schema(
            namespace,
            from_schema_key,
            include_alias_sources=include_alias_sources,
        )
    elif from_hash is not None:
        sources = [
            _source_from_hash(
                namespace,
                from_hash,
                include_alias_sources=include_alias_sources,
            )
        ]
    elif from_furu_obj is not None:
        sources = [
            _source_from_furu_obj(
                namespace,
                from_furu_obj,
                include_alias_sources=include_alias_sources,
            )
        ]
    else:
        from_schema_key = _schema_from_drop_add(cls, from_drop, from_add)
        sources = _sources_from_schema(
            namespace,
            from_schema_key,
            include_alias_sources=include_alias_sources,
        )

    if not sources:
        return MigrationReport(records=[], skips=[])

    alias_index = collect_aliases(include_inactive=True)
    alias_schema_cache: dict[Path, tuple[str, ...]] = {}
    seen_alias_schema: set[tuple[tuple[str, str, RootKind], tuple[str, ...]]] = set()

    rename_map = dict(rename_field) if rename_field is not None else {}
    drop_list = list(drop_field) if drop_field is not None else []
    default_list = list(default_field) if default_field is not None else []
    set_map = dict(set_field) if set_field is not None else {}

    records: list[MigrationRecord] = []
    skips: list[MigrationSkip] = []

    for source in sources:
        original_ref = resolve_original_ref(source.ref)
        _ensure_original_success(original_ref)

        target_fields = schema_key_from_cls(cast(type, cls))
        updated_fields = _apply_transforms(
            source.furu_obj,
            target_fields=target_fields,
            rename_field=rename_map,
            drop_field=drop_list,
            default_field=default_list,
            set_field=set_map,
            target_class=cls,
        )

        target_namespace = _namespace_str(cls)
        target_obj = dict(updated_fields)
        target_obj["__class__"] = target_namespace

        obj = FuruSerializer.from_dict(target_obj)
        target_hash = FuruSerializer.compute_hash(obj)
        target_ref = _target_ref(cls, target_hash)
        target_schema_key = schema_key_from_furu_obj(FuruSerializer.to_dict(obj))

        alias_key = (original_ref.namespace, original_ref.furu_hash, original_ref.root)
        if (alias_key, target_schema_key) in seen_alias_schema:
            reason = "migration: alias schema already created in this run"
            if conflict == "skip":
                skips.append(MigrationSkip(source=source.ref, reason=reason))
                continue
            raise ValueError(reason)

        if _alias_schema_conflict(
            alias_index,
            alias_schema_cache,
            alias_key,
            target_schema_key,
        ):
            reason = "migration: alias schema already exists for original"
            if conflict == "skip":
                skips.append(MigrationSkip(source=source.ref, reason=reason))
                continue
            raise ValueError(reason)

        record = _write_alias(
            target_obj=obj,
            original_ref=original_ref,
            target_ref=target_ref,
            source_ref=source.ref,
            skips=skips,
            conflict=conflict,
            origin=origin,
            note=note,
        )
        if record is None:
            continue
        records.append(record)
        seen_alias_schema.add((alias_key, target_schema_key))

    return MigrationReport(records=records, skips=skips)


def migrate_one(
    cls: _FuruClass,
    *,
    from_hash: str,
    from_namespace: str | None = None,
    default_field: Iterable[str] | None = None,
    set_field: Mapping[str, JsonValue] | None = None,
    drop_field: Iterable[str] | None = None,
    rename_field: Mapping[str, str] | None = None,
    include_alias_sources: bool = False,
    conflict: MigrationConflict = "throw",
    origin: str | None = None,
    note: str | None = None,
) -> MigrationRecord | None:
    report = migrate(
        cls,
        from_hash=from_hash,
        from_namespace=from_namespace,
        default_field=default_field,
        set_field=set_field,
        drop_field=drop_field,
        rename_field=rename_field,
        include_alias_sources=include_alias_sources,
        conflict=conflict,
        origin=origin,
        note=note,
    )
    if report.records:
        return report.records[0]
    return None


def current(
    cls: _FuruClass,
    *,
    namespace: str | None = None,
) -> list[FuruRef]:
    target_schema = schema_key_from_cls(cast(type, cls))
    return _refs_by_schema_compatibility(
        namespace or _namespace_str(cls),
        target_schema,
        cls=cls,
        match=True,
    )


def stale(
    cls: _FuruClass,
    *,
    namespace: str | None = None,
) -> list[FuruRef]:
    target_schema = schema_key_from_cls(cast(type, cls))
    return _refs_by_schema_compatibility(
        namespace or _namespace_str(cls),
        target_schema,
        cls=cls,
        match=False,
    )


def resolve_original_ref(ref: FuruRef) -> FuruRef:
    current = ref
    seen: set[tuple[str, str, RootKind]] = set()
    while True:
        record = MigrationManager.read_migration(current.furu_dir)
        if record is None or record.kind != "alias":
            return current
        key = (record.from_namespace, record.from_hash, record.from_root)
        if key in seen:
            raise ValueError("migration: alias loop detected")
        seen.add(key)
        directory = MigrationManager.resolve_dir(record, target="from")
        current = FuruRef(
            namespace=record.from_namespace,
            furu_hash=record.from_hash,
            root=record.from_root,
            furu_dir=directory,
        )


def _assert_furu_object(value: JsonValue) -> _FuruObject:
    if not chz.is_chz(value):
        raise TypeError("migration: transform must return a Furu object")
    cls = value.__class__
    namespace_method = getattr(cls, "_namespace", None)
    if not callable(namespace_method):
        raise TypeError("migration: transform must return a Furu object")
    version_controlled = getattr(cls, "version_controlled", None)
    if not isinstance(version_controlled, bool):
        raise TypeError("migration: transform must return a Furu object")
    return cast(_FuruObject, value)


def _validate_furu_object_types(target_obj: _FuruObject) -> None:
    for field in chz.chz_fields(target_obj).values():
        value = getattr(target_obj, field.logical_name)
        expected = field.x_type
        if is_subtype_instance(value, expected):
            continue
        raise TypeError(
            "migration: strict_types check failed for field "
            f"{field.logical_name!r}: expected {type_repr(expected)}, "
            f"got {type_repr(type(value))}"
        )


def _coerce_furu_object_types(target_obj: _FuruObject) -> _FuruObject:
    updates: dict[str, JsonValue] = {}
    for field in chz.chz_fields(target_obj).values():
        name = field.logical_name
        value = getattr(target_obj, name)
        coerced = _coerce_value_for_type(value, field.final_type)
        if coerced is value:
            continue
        updates[name] = coerced
    if not updates:
        return target_obj
    return cast(_FuruObject, chz.replace(target_obj, **updates))


def _coerce_value_for_type(
    value: JsonValue,
    expected_type: RuntimeTypeAnnotation,
) -> JsonValue:
    """Coerce mappings toward a target type annotation when possible.

    `expected_type` may be a concrete runtime type, a `typing.Union`/`X | Y`,
    or another typing annotation provided by field metadata.
    """
    if is_subtype_instance(value, expected_type):
        return value

    union_origin = get_origin(expected_type)
    if union_origin in (types.UnionType, Union):
        return _coerce_union_value(
            value,
            union_types=get_args(expected_type),
            expected_type=expected_type,
        )

    if not isinstance(expected_type, type):
        return value
    if not isinstance(value, Mapping):
        return value

    if dataclasses.is_dataclass(expected_type):
        return _coerce_mapping_to_dataclass(value, expected_type)

    if chz.is_chz(expected_type):
        return _coerce_mapping_to_chz(value, expected_type)

    return value


def _coerce_union_value(
    value: JsonValue,
    *,
    union_types: tuple[RuntimeTypeAnnotation, ...],
    expected_type: RuntimeTypeAnnotation,
) -> JsonValue:
    if not isinstance(value, Mapping):
        return value

    class_marker_value = value.get(FuruSerializer.CLASS_MARKER)
    class_marker: str | None
    if class_marker_value is None:
        class_marker = None
    elif isinstance(class_marker_value, str):
        class_marker = class_marker_value
    else:
        return value

    candidates: list[tuple[RuntimeTypeAnnotation, JsonValue]] = []
    for union_type in union_types:
        if class_marker is not None and not _union_type_matches_class_marker(
            union_type,
            class_marker,
        ):
            continue
        coerced = _coerce_value_for_type(value, union_type)
        if not is_subtype_instance(coerced, union_type):
            continue
        candidates.append((union_type, coerced))

    if class_marker is not None:
        if not candidates:
            raise TypeError(
                "migration: class marker does not match a valid union branch: "
                f"{class_marker!r} for {type_repr(expected_type)}"
            )
        if len(candidates) > 1:
            raise TypeError(
                "migration: class marker is ambiguous across union branches: "
                f"{class_marker!r} for {type_repr(expected_type)}"
            )
        return candidates[0][1]

    if not candidates:
        return value
    if len(candidates) == 1:
        return candidates[0][1]

    candidate_types = ", ".join(type_repr(union_type) for union_type, _ in candidates)
    raise TypeError(
        "migration: ambiguous union match for dict payload; "
        f"add '{FuruSerializer.CLASS_MARKER}' to disambiguate among: {candidate_types}"
    )


def _coerce_mapping_to_dataclass(
    value: Mapping[str, JsonValue],
    expected_type: type[Any],
) -> JsonValue:
    if not _mapping_class_marker_matches(value, expected_type):
        return cast(JsonValue, value)

    dataclass_fields = [
        field for field in dataclasses.fields(expected_type) if field.init
    ]
    dataclass_type_hints = dict(inspect.get_annotations(expected_type, eval_str=False))
    try:
        dataclass_type_hints.update(get_type_hints(expected_type))
    except (AttributeError, NameError, TypeError):
        pass

    provided_keys = _mapping_field_keys(value)
    expected_keys = {field.name for field in dataclass_fields}
    required_keys = {
        field.name
        for field in dataclass_fields
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    }

    if not _mapping_keys_are_supported(provided_keys, expected_keys):
        return cast(JsonValue, value)
    if not _mapping_has_required_keys(provided_keys, required_keys):
        return cast(JsonValue, value)

    init_kwargs: dict[str, JsonValue] = {}
    for field in dataclass_fields:
        if field.name not in provided_keys:
            continue
        field_type = dataclass_type_hints.get(field.name, field.type)
        field_value = value[field.name]
        coerced = _coerce_value_for_type(field_value, field_type)
        if not is_subtype_instance(coerced, field_type):
            return cast(JsonValue, value)
        init_kwargs[field.name] = coerced
    return expected_type(**init_kwargs)


def _coerce_mapping_to_chz(
    value: Mapping[str, JsonValue],
    expected_type: type[Any],
) -> JsonValue:
    if not _mapping_class_marker_matches(value, expected_type):
        return cast(JsonValue, value)

    fields = chz.chz_fields(expected_type)
    provided_keys = _mapping_field_keys(value)
    expected_keys = set(fields)
    required_keys = {
        name
        for name, field in fields.items()
        if field._default is CHZ_MISSING
        and isinstance(field._default_factory, MISSING_TYPE)
    }

    if not _mapping_keys_are_supported(provided_keys, expected_keys):
        return cast(JsonValue, value)
    if not _mapping_has_required_keys(provided_keys, required_keys):
        return cast(JsonValue, value)

    init_kwargs: dict[str, JsonValue] = {}
    for name, field in fields.items():
        if name not in provided_keys:
            continue
        field_value = value[name]
        coerced = _coerce_value_for_type(field_value, field.final_type)
        if not is_subtype_instance(coerced, field.final_type):
            return cast(JsonValue, value)
        init_kwargs[name] = coerced
    return expected_type(**init_kwargs)


def _mapping_class_marker_matches(
    value: Mapping[str, JsonValue],
    expected_type: type[Any],
) -> bool:
    marker_value = value.get(FuruSerializer.CLASS_MARKER)
    if marker_value is None:
        return True
    if not isinstance(marker_value, str):
        return False
    return marker_value == _class_marker_for_type(expected_type)


def _mapping_field_keys(value: Mapping[str, JsonValue]) -> set[str]:
    return {key for key in value if key != FuruSerializer.CLASS_MARKER}


def _mapping_keys_are_supported(keys: set[str], expected_keys: set[str]) -> bool:
    return keys <= expected_keys


def _mapping_has_required_keys(keys: set[str], required_keys: set[str]) -> bool:
    return required_keys <= keys


def _union_type_matches_class_marker(
    union_type: RuntimeTypeAnnotation,
    class_marker: str,
) -> bool:
    if not isinstance(union_type, type):
        return False
    return _class_marker_for_type(union_type) == class_marker


def _class_marker_for_type(expected_type: type[Any]) -> str:
    return f"{expected_type.__module__}.{expected_type.__qualname__}"


def _schema_from_drop_add(
    cls: _FuruClass,
    from_drop: Iterable[str] | None,
    from_add: Iterable[str] | None,
) -> tuple[str, ...]:
    current = set(schema_key_from_cls(cast(type, cls)))

    if from_drop is not None:
        drop_fields = _normalize_schema(from_drop)
        missing = set(drop_fields) - current
        if missing:
            raise ValueError(
                f"migration: from_drop fields not in current schema: {_format_fields(missing)}"
            )
        current -= set(drop_fields)

    if from_add is not None:
        add_fields = _normalize_schema(from_add)
        overlap = set(add_fields) & current
        if overlap:
            raise ValueError(
                f"migration: from_add fields already in current schema: {_format_fields(overlap)}"
            )
        current |= set(add_fields)

    return tuple(sorted(current))


def _normalize_schema(values: Iterable[str]) -> tuple[str, ...]:
    keys: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("migration: schema fields must be strings")
        if value.startswith("_"):
            raise ValueError("migration: schema fields cannot start with '_'")
        keys.add(value)
    return tuple(sorted(keys))


def _sources_from_schema(
    namespace: str,
    schema_key: tuple[str, ...],
    *,
    include_alias_sources: bool,
) -> list[_Source]:
    sources: list[_Source] = []
    for ref, metadata in _iter_namespace_metadata(
        namespace,
        include_alias_sources=include_alias_sources,
    ):
        if schema_key_from_metadata_raw(metadata) != schema_key:
            continue
        furu_obj = metadata.get("furu_obj")
        if not isinstance(furu_obj, dict):
            raise TypeError("migration: metadata furu_obj must be a dict")
        migration = MigrationManager.read_migration(ref.furu_dir)
        sources.append(_Source(ref=ref, furu_obj=furu_obj, migration=migration))
    return sources


def _source_from_hash(
    namespace: str,
    furu_hash: str,
    *,
    include_alias_sources: bool,
) -> _Source:
    ref = _find_ref_by_hash(namespace, furu_hash)
    metadata = MetadataManager.read_metadata_raw(ref.furu_dir)
    if metadata is None:
        raise FileNotFoundError(f"migration: metadata not found for {ref.furu_dir}")
    furu_obj = metadata.get("furu_obj")
    if not isinstance(furu_obj, dict):
        raise TypeError("migration: metadata furu_obj must be a dict")
    migration = MigrationManager.read_migration(ref.furu_dir)
    if not _alias_source_allowed(
        migration,
        include_alias_sources=include_alias_sources,
    ):
        raise ValueError(
            "migration: source is an alias; set include_alias_sources=True"
        )
    return _Source(ref=ref, furu_obj=furu_obj, migration=migration)


def _source_from_furu_obj(
    namespace: str,
    furu_obj: dict[str, JsonValue],
    *,
    include_alias_sources: bool,
) -> _Source:
    furu_hash = FuruSerializer.compute_hash(furu_obj)
    return _source_from_hash(
        namespace,
        furu_hash,
        include_alias_sources=include_alias_sources,
    )


def _iter_namespace_metadata(
    namespace: str,
    *,
    include_alias_sources: bool,
) -> Iterable[tuple[FuruRef, dict[str, JsonValue]]]:
    namespace_path = Path(*namespace.split("."))
    for version_controlled in (False, True):
        root = FURU_CONFIG.get_root(version_controlled=version_controlled)
        class_dir = root / namespace_path
        if not class_dir.exists():
            continue
        for entry in class_dir.iterdir():
            if not entry.is_dir():
                continue
            metadata = MetadataManager.read_metadata_raw(entry)
            if metadata is None:
                continue
            migration = MigrationManager.read_migration(entry)
            if not _alias_source_allowed(
                migration,
                include_alias_sources=include_alias_sources,
            ):
                continue
            root_kind: RootKind = "git" if version_controlled else "data"
            ref = FuruRef(
                namespace=namespace,
                furu_hash=entry.name,
                root=root_kind,
                furu_dir=entry,
            )
            yield ref, metadata


def _alias_source_allowed(
    migration: MigrationRecord | None,
    *,
    include_alias_sources: bool,
) -> bool:
    if migration is None:
        return True
    if migration.kind != "alias":
        return True
    return include_alias_sources


def _find_ref_by_hash(namespace: str, furu_hash: str) -> FuruRef:
    namespace_path = Path(*namespace.split("."))
    matches: list[FuruRef] = []
    for version_controlled in (False, True):
        root = FURU_CONFIG.get_root(version_controlled=version_controlled)
        directory = root / namespace_path / furu_hash
        if not directory.is_dir():
            continue
        root_kind: RootKind = "git" if version_controlled else "data"
        matches.append(
            FuruRef(
                namespace=namespace,
                furu_hash=furu_hash,
                root=root_kind,
                furu_dir=directory,
            )
        )
    if not matches:
        raise FileNotFoundError(
            f"migration: source not found for namespace={namespace} hash={furu_hash}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"migration: multiple sources found for namespace={namespace} hash={furu_hash}"
        )
    return matches[0]


def _apply_transforms(
    source_obj: dict[str, JsonValue],
    *,
    target_fields: tuple[str, ...],
    rename_field: Mapping[str, str],
    drop_field: Iterable[str],
    default_field: Iterable[str],
    set_field: Mapping[str, JsonValue],
    target_class: _FuruClass,
) -> dict[str, JsonValue]:
    fields = {k: v for k, v in source_obj.items() if k != "__class__"}
    target_field_set = set(target_fields)

    for old, new in rename_field.items():
        if old not in fields:
            raise ValueError(
                f"migration: rename_field missing source field: {_format_fields([old])}"
            )
        if new in fields:
            raise ValueError(
                f"migration: rename_field target already exists: {_format_fields([new])}"
            )
        if new not in target_field_set:
            raise ValueError(
                f"migration: rename_field target not in schema: {_format_fields([new])}"
            )
        fields[new] = fields.pop(old)

    for name in drop_field:
        if name not in fields:
            raise ValueError(
                f"migration: drop_field missing source field: {_format_fields([name])}"
            )
        fields.pop(name)

    for name in default_field:
        if name not in target_field_set:
            raise ValueError(
                f"migration: default_field not in schema: {_format_fields([name])}"
            )
        if name in fields:
            continue
        fields[name] = _serialize_value(_default_value_for_field(target_class, name))

    for name, value in set_field.items():
        if name not in target_field_set:
            raise ValueError(
                f"migration: set_field not in schema: {_format_fields([name])}"
            )
        if name in fields:
            raise ValueError(
                f"migration: set_field already set: {_format_fields([name])}"
            )
        fields[name] = _serialize_value(value)

    return fields


def _default_value_for_field(target_class: _FuruClass, name: str) -> JsonValue:
    fields = chz.chz_fields(target_class)
    fields_by_logical = {field.logical_name: field for field in fields.values()}
    field = fields_by_logical.get(name)
    if field is None:
        raise ValueError(
            f"migration: default_field missing defaults for fields: {_format_fields([name])}"
        )
    if field._default is not CHZ_MISSING:
        return field._default
    if not isinstance(field._default_factory, MISSING_TYPE):
        return field._default_factory()
    raise ValueError(
        f"migration: default_field missing defaults for fields: {_format_fields([name])}"
    )


def _serialize_value(value: JsonValue) -> JsonValue:
    result = FuruSerializer.to_dict(value)
    if result is None:
        return result
    if isinstance(result, (str, int, float, bool, list, dict)):
        return result
    raise TypeError(f"migration: unsupported value type {type(result)}")


def _write_alias(
    *,
    target_obj: JsonValue,
    original_ref: FuruRef,
    target_ref: FuruRef,
    source_ref: FuruRef,
    skips: list[MigrationSkip],
    conflict: MigrationConflict,
    origin: str | None,
    note: str | None,
) -> MigrationRecord | None:
    try:
        target_ref.furu_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        reason = "migration: target already exists"
        if conflict == "skip":
            skips.append(MigrationSkip(source=source_ref, reason=reason))
            return None
        raise ValueError(reason) from None
    StateManager.ensure_internal_dir(target_ref.furu_dir)
    _write_migrated_state(target_ref.furu_dir)

    metadata = MetadataManager.create_metadata(
        target_obj, target_ref.furu_dir, ignore_diff=True
    )
    MetadataManager.write_metadata(metadata, target_ref.furu_dir)

    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    record = MigrationRecord(
        kind="alias",
        policy="alias",
        from_namespace=original_ref.namespace,
        from_hash=original_ref.furu_hash,
        from_root=original_ref.root,
        to_namespace=target_ref.namespace,
        to_hash=target_ref.furu_hash,
        to_root=target_ref.root,
        migrated_at=now,
        overwritten_at=None,
        default_values=None,
        origin=origin,
        note=note,
    )
    MigrationManager.write_migration(record, target_ref.furu_dir)
    return record


def _write_migrated_state(directory: Path) -> None:
    def mutate(state) -> None:
        state.result = _StateResultMigrated(status="migrated")
        state.attempt = None

    StateManager.update_state(directory, mutate)


def _ensure_original_success(original_ref: FuruRef) -> None:
    if not StateManager.success_marker_exists(original_ref.furu_dir):
        raise ValueError("migration: original artifact is not successful")


def _alias_schema_conflict(
    alias_index: dict[tuple[str, str, RootKind], list[MigrationRecord]],
    alias_schema_cache: dict[Path, tuple[str, ...]],
    alias_key: tuple[str, str, RootKind],
    target_schema_key: tuple[str, ...],
) -> bool:
    records = alias_index.get(alias_key, [])
    for record in records:
        alias_dir = MigrationManager.resolve_dir(record, target="to")
        if alias_dir not in alias_schema_cache:
            metadata = MetadataManager.read_metadata_raw(alias_dir)
            if metadata is None:
                raise FileNotFoundError(
                    f"migration: metadata not found for alias {alias_dir}"
                )
            alias_schema_cache[alias_dir] = schema_key_from_metadata_raw(metadata)
        if alias_schema_cache[alias_dir] == target_schema_key:
            return True
    return False


def _ref_matches_current_code(
    ref: FuruRef,
    metadata: dict[str, JsonValue],
    *,
    cls: _FuruClass,
    target_schema: tuple[str, ...],
) -> bool:
    # 1) shallow schema_key must match
    if schema_key_from_metadata_raw(metadata) != target_schema:
        return False

    furu_obj = metadata.get("furu_obj")
    if not isinstance(furu_obj, dict):
        return False

    # 2) must deserialize
    try:
        obj = FuruSerializer.from_dict(furu_obj)
    except Exception:
        return False

    # must hydrate to the expected class (namespace mismatch => stale)
    if not isinstance(obj, cast(type, cls)):
        return False

    # 3) must re-hash to the directory hash
    try:
        current_hash = FuruSerializer.compute_hash(obj)
    except Exception:
        return False

    return current_hash == ref.furu_hash


def _refs_by_schema_compatibility(
    namespace: str,
    schema_key: tuple[str, ...],
    *,
    match: bool,
    cls: _FuruClass,
) -> list[FuruRef]:
    refs: list[FuruRef] = []
    for ref, metadata in _iter_namespace_metadata(
        namespace,
        include_alias_sources=True,
    ):
        is_current = _ref_matches_current_code(
            ref,
            metadata,
            cls=cls,
            target_schema=schema_key,
        )
        if is_current == match:
            refs.append(ref)
    refs.sort(key=lambda item: item.furu_hash)
    return refs


def _namespace_str(cls: _FuruClass) -> str:
    return ".".join(cls._namespace().parts)


def _target_ref(cls: _FuruClass, furu_hash: str) -> FuruRef:
    root = FURU_CONFIG.get_root(version_controlled=cls.version_controlled)
    namespace = _namespace_str(cls)
    directory = root / Path(*namespace.split(".")) / furu_hash
    root_kind: RootKind = "git" if cls.version_controlled else "data"
    return FuruRef(
        namespace=namespace,
        furu_hash=furu_hash,
        root=root_kind,
        furu_dir=directory,
    )


def _format_fields(fields: Iterable[str]) -> str:
    return ", ".join(sorted(fields))
