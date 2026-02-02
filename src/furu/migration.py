from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import chz
from chz.util import MISSING as CHZ_MISSING, MISSING_TYPE

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
    def _namespace(cls) -> Path: """
Get the filesystem path that serves as the namespace directory for the given class.

Returns:
    Path: The directory path used to store artifacts for the class's namespace.
"""
...


MigrationConflict = Literal["throw", "skip"]


@dataclass(frozen=True)
class FuruRef:
    namespace: str
    furu_hash: str
    root: RootKind
    directory: Path


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
    Migrate one or more stored artifacts into the schema represented by `cls`, creating alias migration records for each successful migration.
    
    Parameters:
        cls (_FuruClass): Target class whose schema key will be used for migrated objects.
        from_schema (Iterable[str] | None): Select sources whose schema key matches this collection of field names.
        from_drop (Iterable[str] | None): Fields to remove from the source schema when selecting sources (used with `from_add`).
        from_add (Iterable[str] | None): Fields to add to the source schema when selecting sources (used with `from_drop`).
        from_hash (str | None): Select a single source by its furu hash.
        from_furu_obj (dict[str, JsonValue] | None): Select a single source defined by the given serialized object (its hash is computed).
        from_namespace (str | None): Override the namespace to search for sources; defaults to the namespace of `cls`.
        default_field (Iterable[str] | None): Fields to ensure are present on the migrated object by applying default values when missing.
        set_field (Mapping[str, JsonValue] | None): Explicit field values to set on the migrated object; errors if a target field is already present.
        drop_field (Iterable[str] | None): Fields to remove from each source during migration.
        rename_field (Mapping[str, str] | None): Map of source field name -> target field name to rename fields during migration.
        include_alias_sources (bool): If True, allow alias entries to be considered as migration sources.
        conflict (MigrationConflict): Policy when encountering existing targets or alias/schema conflicts; `"throw"` raises, `"skip"` records a skip.
        origin (str | None): Optional origin identifier to record on created migration entries.
        note (str | None): Optional free-text note to record on created migration entries.
    
    Returns:
        MigrationReport: Contains `records` for successfully created MigrationRecord entries and `skips` for sources that were skipped with reasons.
    
    Raises:
        ValueError: If source selection is missing or ambiguous, or if a conflict is encountered and `conflict` is `"throw"`.
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
            namespace, from_schema_key, include_alias_sources
        )
    elif from_hash is not None:
        sources = [_source_from_hash(namespace, from_hash, include_alias_sources)]
    elif from_furu_obj is not None:
        sources = [
            _source_from_furu_obj(namespace, from_furu_obj, include_alias_sources)
        ]
    else:
        from_schema_key = _schema_from_drop_add(cls, from_drop, from_add)
        sources = _sources_from_schema(
            namespace, from_schema_key, include_alias_sources
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

        if target_ref.directory.exists():
            reason = "migration: target already exists"
            if conflict == "skip":
                skips.append(MigrationSkip(source=source.ref, reason=reason))
                continue
            raise ValueError(reason)

        record = _write_alias(
            target_obj=obj,
            original_ref=original_ref,
            target_ref=target_ref,
            origin=origin,
            note=note,
        )
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
    """
    Migrate a single Furu object, selected by its hash, to the schema of the given class and return the resulting migration record if any.
    
    Parameters:
        cls (_FuruClass): Target class whose schema and namespace determine the migration target.
        from_hash (str): Hash of the source Furu object to migrate.
        from_namespace (str | None): Override namespace to locate the source; if None the namespace is derived from cls.
        default_field (Iterable[str] | None): Field names for which to apply default values when missing in the source.
        set_field (Mapping[str, JsonValue] | None): Field values to set on the migrated object; keys are field names and values must be JSON-serializable primitives/structures.
        drop_field (Iterable[str] | None): Field names to remove from the source before migration.
        rename_field (Mapping[str, str] | None): Mapping of source field name -> target field name to rename fields during migration.
        include_alias_sources (bool): If True, allow alias records to be considered as valid migration sources.
        conflict (MigrationConflict): Behavior when encountering alias/schema conflicts; "throw" to raise, "skip" to skip conflicted sources.
        origin (str | None): Optional origin identifier to record with the migration.
        note (str | None): Optional human-readable note to record with the migration.
    
    Returns:
        MigrationRecord | None: The first MigrationRecord produced for the migrated object, or `None` if no migration record was created.
    """
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
    """
    Return FuruRefs whose stored schema matches the current schema of the given Furu class.
    
    Parameters:
        namespace (str | None): Optional namespace to search; when omitted the class's namespace is used.
    
    Returns:
        list[FuruRef]: Targets whose metadata schema key equals the class's current schema key.
    """
    target_schema = schema_key_from_cls(cast(type, cls))
    return _refs_by_schema(namespace or _namespace_str(cls), target_schema, match=True)


def stale(
    cls: _FuruClass,
    *,
    namespace: str | None = None,
) -> list[FuruRef]:
    """
    Return FuruRefs in a namespace whose metadata schema key does not match the class's current schema.
    
    Parameters:
        namespace (str | None): Namespace to search; if None, use the class's namespace.
    
    Returns:
        list[FuruRef]: FuruRef objects for targets whose schema key is different from the class's current schema.
    """
    target_schema = schema_key_from_cls(cast(type, cls))
    return _refs_by_schema(namespace or _namespace_str(cls), target_schema, match=False)


def resolve_original_ref(ref: FuruRef) -> FuruRef:
    """
    Resolve an alias chain for a FuruRef to its ultimate non-alias reference.
    
    Parameters:
        ref (FuruRef): Starting reference which may point to an alias.
    
    Returns:
        FuruRef: The resolved reference that is not an alias.
    
    Raises:
        ValueError: If an alias loop is detected while following alias chains.
    """
    current = ref
    seen: set[tuple[str, str, RootKind]] = set()
    while True:
        record = MigrationManager.read_migration(current.directory)
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
            directory=directory,
        )


def _schema_from_drop_add(
    cls: _FuruClass,
    from_drop: Iterable[str] | None,
    from_add: Iterable[str] | None,
) -> tuple[str, ...]:
    """
    Compute the resulting schema key after applying drop and add operations to the class's current schema.
    
    Parameters:
        cls: The target class whose current schema will be modified.
        from_drop: Iterable of field names to remove from the current schema; if None, no fields are removed.
        from_add: Iterable of field names to add to the current schema; if None, no fields are added.
    
    Returns:
        tuple[str, ...]: Sorted tuple of resulting field names forming the new schema key.
    
    Raises:
        ValueError: If any field in `from_drop` does not exist in the current schema, or if any field in `from_add` already exists in the current schema.
    """
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
    """
    Validate and normalize a collection of schema field names.
    
    Parameters:
        values (Iterable[str]): Iterable of candidate field names.
    
    Returns:
        tuple[str, ...]: Sorted tuple of unique field names.
    
    Raises:
        TypeError: If any value in `values` is not a string.
        ValueError: If any field name starts with an underscore.
    """
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
    include_alias_sources: bool,
) -> list[_Source]:
    """
    Collect all sources in a namespace whose stored schema key equals the given schema_key.
    
    Iterates namespace metadata (optionally including alias entries) and builds a list of _Source objects for entries whose metadata schema key matches the provided tuple.
    
    Parameters:
        namespace (str): Namespace to search.
        schema_key (tuple[str, ...]): Target schema key to match.
        include_alias_sources (bool): If True, include alias records when scanning metadata.
    
    Returns:
        list[_Source]: _Source entries for each matching metadata record.
    
    Raises:
        TypeError: If a matching metadata entry's "furu_obj" is not a dict.
    """
    sources: list[_Source] = []
    for ref, metadata in _iter_namespace_metadata(namespace, include_alias_sources):
        if schema_key_from_metadata_raw(metadata) != schema_key:
            continue
        furu_obj = metadata.get("furu_obj")
        if not isinstance(furu_obj, dict):
            raise TypeError("migration: metadata furu_obj must be a dict")
        migration = MigrationManager.read_migration(ref.directory)
        sources.append(_Source(ref=ref, furu_obj=furu_obj, migration=migration))
    return sources


def _source_from_hash(
    namespace: str,
    furu_hash: str,
    include_alias_sources: bool,
) -> _Source:
    """
    Load a migration source identified by namespace and furu hash.
    
    Reads the stored metadata for the given namespace and hash, validates that the stored `furu_obj` is a mapping, and returns an _Source containing the reference, the object, and any migration record. If the stored item is an alias and `include_alias_sources` is False, the function raises an error.
    
    Parameters:
        namespace (str): Namespace to search.
        furu_hash (str): Hash of the furu artifact to locate.
        include_alias_sources (bool): If True, accept alias records as valid sources; if False, raise when the found record is an alias.
    
    Returns:
        _Source: A source record with `ref`, `furu_obj` (dict), and optional `migration`.
    
    Raises:
        FileNotFoundError: If metadata for the found reference is missing.
        TypeError: If the metadata's `furu_obj` is not a dict.
        ValueError: If the found migration is an alias while `include_alias_sources` is False.
    """
    ref = _find_ref_by_hash(namespace, furu_hash)
    metadata = MetadataManager.read_metadata_raw(ref.directory)
    if metadata is None:
        raise FileNotFoundError(f"migration: metadata not found for {ref.directory}")
    furu_obj = metadata.get("furu_obj")
    if not isinstance(furu_obj, dict):
        raise TypeError("migration: metadata furu_obj must be a dict")
    migration = MigrationManager.read_migration(ref.directory)
    if (
        migration is not None
        and migration.kind == "alias"
        and not include_alias_sources
    ):
        raise ValueError(
            "migration: source is an alias; set include_alias_sources=True"
        )
    return _Source(ref=ref, furu_obj=furu_obj, migration=migration)


def _source_from_furu_obj(
    namespace: str,
    furu_obj: dict[str, JsonValue],
    include_alias_sources: bool,
) -> _Source:
    """
    Create a migration source entry for a given in-memory Furu object by computing its content hash and locating the corresponding stored artifact.
    
    Parameters:
        namespace (str): Namespace in which to look up the stored artifact.
        furu_obj (dict[str, JsonValue]): The serialized object to compute the hash from.
        include_alias_sources (bool): If True, consider alias migration records as valid lookup targets; if False, ignore alias records.
    
    Returns:
        _Source: A _Source containing the located FuruRef, the original `furu_obj`, and any associated MigrationRecord (if present).
    """
    furu_hash = FuruSerializer.compute_hash(furu_obj)
    return _source_from_hash(namespace, furu_hash, include_alias_sources)


def _iter_namespace_metadata(
    namespace: str, include_alias_sources: bool
) -> Iterable[tuple[FuruRef, dict[str, JsonValue]]]:
    """
    Iterate metadata entries for a namespace across both non-versioned and versioned roots.
    
    Yields pairs of (FuruRef, metadata) for each metadata directory found under the namespace. When an entry corresponds to an alias migration and `include_alias_sources` is False, that entry is skipped.
    
    Parameters:
        namespace (str): Dot-separated class namespace (e.g., "package.module.Class") to search.
        include_alias_sources (bool): If True, include entries whose migration record is an alias; if False, skip alias sources.
    
    Returns:
        Iterable[tuple[FuruRef, dict[str, JsonValue]]]: An iterator of tuples where the first element is the reference to the metadata directory and the second element is the raw metadata mapping.
    """
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
            if (
                migration is not None
                and migration.kind == "alias"
                and not include_alias_sources
            ):
                continue
            root_kind: RootKind = "git" if version_controlled else "data"
            ref = FuruRef(
                namespace=namespace,
                furu_hash=entry.name,
                root=root_kind,
                directory=entry,
            )
            yield ref, metadata


def _find_ref_by_hash(namespace: str, furu_hash: str) -> FuruRef:
    """
    Locate the FuruRef for a given namespace and furu object hash across configured roots.
    
    Parameters:
        namespace (str): Dot-separated namespace (e.g., "pkg.module") to search.
        furu_hash (str): The furu object hash to locate.
    
    Returns:
        FuruRef: The single matching reference; its `root` indicates whether it was found in the version-controlled ("git") or non-version-controlled ("data") root.
    
    Raises:
        FileNotFoundError: If no matching reference is found.
        ValueError: If more than one matching reference is found.
    """
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
                directory=directory,
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
    """
    Apply a set of migration transformations to a source object's fields and produce the transformed field mapping.
    
    Parameters:
        source_obj (dict[str, JsonValue]): Source serialized object (metadata), typically containing fields and a "__class__" entry which will be ignored.
        target_fields (tuple[str, ...]): Ordered field names that define the target schema; used to validate rename/default/set/drop operations.
        rename_field (Mapping[str, str]): Mapping of source field name -> target field name to rename; each source must exist, each target must not already exist in the source, and each target must be present in target_fields.
        drop_field (Iterable[str]): Iterable of field names to remove from the source; each must exist in the source.
        default_field (Iterable[str]): Iterable of field names that must be present in the resulting object; if missing, a default value from target_class is inserted.
        set_field (Mapping[str, JsonValue]): Mapping of field name -> value to set on the resulting object; each name must be in target_fields and must not already be present in the source. Values are serialized before insertion.
        target_class (_FuruClass): The destination class used to resolve default values for fields.
    
    Returns:
        dict[str, JsonValue]: The transformed field mapping (excluding any "__class__" key).
    
    Raises:
        ValueError: If any transform is invalid (missing source field for rename/drop, rename target exists or not in schema, default/set field not in schema, or set targets already present).
        TypeError: If a value provided for `set_field` or a default cannot be serialized to a JSON-compatible form.
    """
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
    """
    Return the default serializable value for a named field on the target class.
    
    Retrieves the field definition from the class and returns the field's default value if one is defined; if no direct default is present, invokes and returns the field's default factory result. Raises a ValueError when the field has neither a default nor a default factory.
    
    Parameters:
        target_class (_FuruClass): Class whose field definition is queried.
        name (str): Name of the field to obtain a default for.
    
    Returns:
        JsonValue: The field's default value (or the result of its default factory).
    
    Raises:
        ValueError: If the field has neither a default nor a default factory.
    """
    fields = chz.chz_fields(target_class)
    field = fields[name]
    if field._default is not CHZ_MISSING:
        return field._default
    if not isinstance(field._default_factory, MISSING_TYPE):
        return field._default_factory()
    raise ValueError(
        f"migration: default_field missing defaults for fields: {_format_fields([name])}"
    )


def _serialize_value(value: JsonValue) -> JsonValue:
    """
    Convert a value into a JSON-serializable primitive form.
    
    Parameters:
        value (JsonValue): The value to serialize.
    
    Returns:
        JsonValue: The serialized primitive (`str`, `int`, `float`, `bool`, `list`, `dict`) or `None`.
    
    Raises:
        TypeError: If the value cannot be represented as one of the allowed primitive types.
    """
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
    origin: str | None,
    note: str | None,
) -> MigrationRecord:
    """
    Create the target directory for an alias, write migrated state and metadata, and record the alias migration.
    
    Parameters:
        target_obj (JsonValue): Serialized object to store in the target metadata.
        original_ref (FuruRef): Reference to the original source being aliased.
        target_ref (FuruRef): Reference for the new alias target; its directory will be created.
        origin (str | None): Optional origin identifier to include in the migration record.
        note (str | None): Optional free-form note to include in the migration record.
    
    Returns:
        MigrationRecord: A migration record of kind "alias" describing the created alias relationship.
    """
    target_ref.directory.mkdir(parents=True, exist_ok=False)
    StateManager.ensure_internal_dir(target_ref.directory)
    _write_migrated_state(target_ref.directory)

    metadata = MetadataManager.create_metadata(
        target_obj, target_ref.directory, ignore_diff=True
    )
    MetadataManager.write_metadata(metadata, target_ref.directory)

    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
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
    MigrationManager.write_migration(record, target_ref.directory)
    return record


def _write_migrated_state(directory: Path) -> None:
    """
    Mark the artifact at the given directory as migrated in the state store.
    
    Parameters:
        directory (Path): Path to the artifact's directory whose migration state will be updated.
    """
    def mutate(state) -> None:
        state.result = _StateResultMigrated(status="migrated")
        state.attempt = None

    StateManager.update_state(directory, mutate)


def _ensure_original_success(original_ref: FuruRef) -> None:
    """
    Ensure the original artifact referenced by `original_ref` has a success marker.
    
    Parameters:
        original_ref (FuruRef): Reference to the original artifact whose success state will be checked.
    
    Raises:
        ValueError: If the original artifact does not have a success marker.
    """
    if not StateManager.success_marker_exists(original_ref.directory):
        raise ValueError("migration: original artifact is not successful")


def _alias_schema_conflict(
    alias_index: dict[tuple[str, str, RootKind], list[MigrationRecord]],
    alias_schema_cache: dict[Path, tuple[str, ...]],
    alias_key: tuple[str, str, RootKind],
    target_schema_key: tuple[str, ...],
) -> bool:
    """
    Check whether any alias for a given alias key points to a different schema than the target schema.
    
    Parameters:
        alias_index (dict[tuple[str, str, RootKind], list[MigrationRecord]]):
            Mapping from alias keys (namespace, furu_hash, root) to migration records that created those aliases.
        alias_schema_cache (dict[Path, tuple[str, ...]]):
            Cache mapping alias directories to their schema key tuples; populated when metadata is read.
        alias_key (tuple[str, str, RootKind]):
            The alias key to look up in alias_index.
        target_schema_key (tuple[str, ...]):
            The schema key tuple that the target should have.
    
    Returns:
        bool: `True` if any alias referenced by `alias_key` has the same schema as `target_schema_key`, `False` otherwise.
    """
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


def _refs_by_schema(
    namespace: str,
    schema_key: tuple[str, ...],
    *,
    match: bool,
) -> list[FuruRef]:
    """
    Collects FuruRef entries in a namespace filtered by whether each entry's metadata schema key matches the given schema_key.
    
    Filters metadata for the provided namespace (including alias sources) and returns the matching references sorted by their furu_hash.
    
    Parameters:
        namespace (str): Namespace to search.
        schema_key (tuple[str, ...]): Target schema key as a tuple of field names.
        match (bool): If True, include refs whose metadata schema equals `schema_key`; if False, include refs whose schema does not equal `schema_key`.
    
    Returns:
        list[FuruRef]: FuruRef objects matching the filter, sorted by `furu_hash`.
    """
    refs: list[FuruRef] = []
    for ref, metadata in _iter_namespace_metadata(
        namespace, include_alias_sources=True
    ):
        current_key = schema_key_from_metadata_raw(metadata)
        if (current_key == schema_key) is match:
            refs.append(ref)
    refs.sort(key=lambda item: item.furu_hash)
    return refs


def _namespace_str(cls: _FuruClass) -> str:
    """
    Return a dot-separated namespace string for a Furu class.
    
    Parameters:
        cls (_FuruClass): Class providing a Path-like namespace via its `_namespace()` classmethod.
    
    Returns:
        namespace_str (str): The namespace joined with dots (e.g., "package.subpackage.object").
    """
    return ".".join(cls._namespace().parts)


def _target_ref(cls: _FuruClass, furu_hash: str) -> FuruRef:
    """
    Create a FuruRef for the given class namespace and furu object hash.
    
    Parameters:
        cls (_FuruClass): Class providing the namespace and version_controlled flag.
        furu_hash (str): Content hash identifying the furu object.
    
    Returns:
        FuruRef: Reference with the resolved namespace, the provided hash, the root kind ("git" if the class is version-controlled, otherwise "data"), and the filesystem directory where the object is stored.
    """
    root = FURU_CONFIG.get_root(version_controlled=cls.version_controlled)
    namespace = _namespace_str(cls)
    directory = root / Path(*namespace.split(".")) / furu_hash
    root_kind: RootKind = "git" if cls.version_controlled else "data"
    return FuruRef(
        namespace=namespace,
        furu_hash=furu_hash,
        root=root_kind,
        directory=directory,
    )


def _format_fields(fields: Iterable[str]) -> str:
    """
    Format an iterable of field names as a sorted, comma-separated string.
    
    Parameters:
        fields (Iterable[str]): Field names to format.
    
    Returns:
        A single string containing the provided field names sorted alphabetically and joined by ", ".
    """
    return ", ".join(sorted(fields))