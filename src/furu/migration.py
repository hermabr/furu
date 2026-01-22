from __future__ import annotations

import datetime as _dt
import importlib
import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast, overload

import chz
from chz.util import MISSING as CHZ_MISSING, MISSING_TYPE
from chz.validators import for_all_fields, typecheck

from .config import FURU_CONFIG
from .core.furu import Furu
from .runtime.logging import get_logger
from .serialization import FuruSerializer
from .serialization.serializer import JsonValue
from .storage import MetadataManager, MigrationManager, MigrationRecord, StateManager
from .storage.state import _StateAttemptRunning, _StateResultMigrated


Primitive: TypeAlias = str | int | float | bool | None
MigrationValue: TypeAlias = (
    Primitive | Furu | tuple["MigrationValue", ...] | dict[str, "MigrationValue"]
)


@dataclass(frozen=True)
class NamespacePair:
    from_namespace: str
    to_namespace: str


@dataclass(frozen=True)
class FuruRef:
    namespace: str
    furu_hash: str
    root: Literal["data", "git"]
    directory: Path


@dataclass(frozen=True)
class MigrationCandidate:
    from_ref: FuruRef
    to_ref: FuruRef
    to_namespace: str
    to_config: dict[str, JsonValue]
    defaults_applied: dict[str, MigrationValue]
    fields_dropped: list[str]
    missing_fields: list[str]
    extra_fields: list[str]

    def with_default_values(
        self, values: Mapping[str, MigrationValue]
    ) -> "MigrationCandidate":
        if not values:
            return self
        updated_defaults = dict(self.defaults_applied)
        updated_defaults.update(values)
        return _rebuild_candidate_with_defaults(self, dict(values), updated_defaults)


@dataclass(frozen=True)
class MigrationSkip:
    candidate: MigrationCandidate
    reason: str


def _rebuild_candidate_with_defaults(
    candidate: MigrationCandidate,
    new_defaults: dict[str, MigrationValue],
    defaults_applied: dict[str, MigrationValue],
) -> MigrationCandidate:
    target_class = _resolve_target_class(candidate.to_namespace)
    updated_config = _typed_config(dict(candidate.to_config))

    target_fields = _target_field_names(target_class)
    config_keys = set(updated_config.keys()) - {"__class__"}
    conflicts = set(new_defaults) & config_keys
    if conflicts:
        raise ValueError(
            "migration: default_values provided for existing fields: "
            f"{_format_fields(conflicts)}"
        )
    unknown = set(new_defaults) - set(target_fields)
    if unknown:
        raise ValueError(
            "migration: default_values contains fields not in target schema: "
            f"{_format_fields(unknown)}"
        )

    for field, value in new_defaults.items():
        updated_config[field] = _serialize_value(value)

    updated_config["__class__"] = candidate.to_namespace
    _typecheck_config(updated_config)

    config_keys = set(updated_config.keys()) - {"__class__"}
    missing_fields = sorted(set(target_fields) - config_keys)
    if missing_fields:
        raise ValueError(
            "migration: missing required fields for target class: "
            f"{_format_fields(missing_fields)}"
        )
    extra_fields = sorted(config_keys - set(target_fields))
    if extra_fields:
        raise ValueError(
            "migration: extra fields present; use drop_fields to remove: "
            f"{_format_fields(extra_fields)}"
        )

    to_hash = FuruSerializer.compute_hash(updated_config)
    to_ref = _build_target_ref(target_class, candidate.to_namespace, to_hash)
    return MigrationCandidate(
        from_ref=candidate.from_ref,
        to_ref=to_ref,
        to_namespace=candidate.to_namespace,
        to_config=updated_config,
        defaults_applied=defaults_applied,
        fields_dropped=candidate.fields_dropped,
        missing_fields=missing_fields,
        extra_fields=extra_fields,
    )


MigrationPolicy = Literal["alias", "move", "copy"]
MigrationConflict = Literal["throw", "skip", "overwrite"]


@overload
def find_migration_candidates(
    *,
    namespace: str,
    to_obj: type[Furu],
    default_values: Mapping[str, MigrationValue] | None = None,
    default_fields: Iterable[str] | None = None,
    drop_fields: Iterable[str] | None = None,
) -> list[MigrationCandidate]: ...


@overload
def find_migration_candidates(
    *,
    namespace: NamespacePair,
    to_obj: None = None,
    default_values: Mapping[str, MigrationValue] | None = None,
    default_fields: Iterable[str] | None = None,
    drop_fields: Iterable[str] | None = None,
) -> list[MigrationCandidate]: ...


def find_migration_candidates(
    *,
    namespace: str | NamespacePair,
    to_obj: type[Furu] | None = None,
    default_values: Mapping[str, MigrationValue] | None = None,
    default_fields: Iterable[str] | None = None,
    drop_fields: Iterable[str] | None = None,
) -> list[MigrationCandidate]:
    if namespace is None:
        raise ValueError("migration: namespace is required")
    if isinstance(namespace, NamespacePair):
        if to_obj is not None:
            raise ValueError("migration: to_obj cannot be used with NamespacePair")
        from_namespace = namespace.from_namespace
        to_namespace = namespace.to_namespace
        target_class = _resolve_target_class(to_namespace)
    elif isinstance(namespace, str):
        if not _is_furu_class(to_obj):
            raise ValueError(
                "migration: to_obj must be a class (use find_migration_candidates_initialized_target for instances)"
            )
        from_namespace = namespace
        target_class = to_obj
        if target_class is None:
            raise ValueError(
                "migration: to_obj must be a class (use find_migration_candidates_initialized_target for instances)"
            )
        to_namespace = _namespace_str(target_class)
    else:
        raise ValueError("migration: namespace must be str or NamespacePair")

    candidates: list[MigrationCandidate] = []
    for from_ref, config in _iter_source_configs(from_namespace):
        candidate = _build_candidate(
            from_ref,
            config,
            to_namespace=to_namespace,
            target_class=target_class,
            default_values=default_values,
            default_fields=default_fields,
            drop_fields=drop_fields,
            default_source=None,
        )
        candidates.append(candidate)
    return candidates


def find_migration_candidates_initialized_target(
    *,
    to_obj: Furu,
    from_namespace: str | None = None,
    default_fields: Iterable[str] | None = None,
    drop_fields: Iterable[str] | None = None,
) -> list[MigrationCandidate]:
    if isinstance(to_obj, type):
        raise ValueError(
            "migration: to_obj must be an instance (use find_migration_candidates for classes)"
        )
    if not isinstance(to_obj, Furu):
        raise ValueError(
            "migration: to_obj must be an instance (use find_migration_candidates for classes)"
        )

    target_class = to_obj.__class__
    to_namespace = _namespace_str(target_class)
    source_namespace = from_namespace or to_namespace

    target_config = FuruSerializer.to_dict(to_obj)
    if not isinstance(target_config, dict):
        raise TypeError("migration: to_obj must serialize to a dict")
    target_config = _typed_config(target_config)
    target_config["__class__"] = to_namespace
    _typecheck_config(target_config)

    candidates: list[MigrationCandidate] = []
    for from_ref, config in _iter_source_configs(source_namespace):
        candidate = _build_candidate(
            from_ref,
            config,
            to_namespace=to_namespace,
            target_class=target_class,
            default_values=None,
            default_fields=default_fields,
            drop_fields=drop_fields,
            default_source=to_obj,
        )
        aligned = _align_candidate_to_target(candidate, target_config)
        if aligned is not None:
            candidates.append(aligned)
    return candidates


@overload
def apply_migration(
    candidate: MigrationCandidate,
    *,
    policy: MigrationPolicy = "alias",
    cascade: bool = True,
    origin: str | None = None,
    note: str | None = None,
    conflict: Literal["throw", "overwrite"] = "throw",
) -> list[MigrationRecord]: ...


@overload
def apply_migration(
    candidate: MigrationCandidate,
    *,
    policy: MigrationPolicy = "alias",
    cascade: bool = True,
    origin: str | None = None,
    note: str | None = None,
    conflict: Literal["skip"],
) -> list[MigrationRecord | MigrationSkip]: ...


def apply_migration(
    candidate: MigrationCandidate,
    *,
    policy: MigrationPolicy = "alias",
    cascade: bool = True,
    origin: str | None = None,
    note: str | None = None,
    conflict: MigrationConflict = "throw",
) -> list[MigrationRecord | MigrationSkip]:
    if policy not in {"alias", "move", "copy"}:
        raise ValueError(f"Unsupported migration policy: {policy}")

    if not cascade:
        get_logger().warning(
            "migration: cascade disabled; dependents will not be migrated"
        )

    cascade_nodes = (
        _build_cascade_candidates(candidate)
        if cascade
        else [_CascadeNode(candidate=candidate, parent=None)]
    )
    parent_map = {node.key: node.parent for node in cascade_nodes}

    conflict_statuses: dict[_CandidateKey, str] = {}
    for node in cascade_nodes:
        status = _target_status(node.candidate)
        if status is not None:
            conflict_statuses[node.key] = status

    if conflict == "throw" and conflict_statuses:
        status = next(iter(conflict_statuses.values()))
        raise ValueError(
            f"migration: target exists with status {status}; pass conflict='overwrite' or conflict='skip'"
        )

    skip_keys: set[_CandidateKey] = set()
    if conflict == "skip" and conflict_statuses:
        skip_keys = _expand_skip_keys(conflict_statuses.keys(), parent_map)
        for key in conflict_statuses:
            status = conflict_statuses[key]
            get_logger().warning(
                "migration: skipping candidate due to target status %s",
                status,
            )

    results: list[MigrationRecord | MigrationSkip] = []
    for node in cascade_nodes:
        if node.key in skip_keys:
            reason = "migration: skipping candidate due to skipped dependency"
            if node.key in conflict_statuses:
                status = conflict_statuses[node.key]
                reason = f"migration: skipping candidate due to target status {status}"
            results.append(MigrationSkip(candidate=node.candidate, reason=reason))
            continue
        record = _apply_single_migration(
            node.candidate,
            policy=policy,
            origin=origin,
            note=note,
            conflict=conflict,
            conflict_status=conflict_statuses.get(node.key),
        )
        results.append(record)
    return results


@dataclass(frozen=True)
class _CascadeNode:
    candidate: MigrationCandidate
    parent: _CandidateKey | None

    @property
    def key(self) -> "_CandidateKey":
        return _candidate_key(self.candidate)


_CandidateKey: TypeAlias = tuple[str, str, str]


def _candidate_key(candidate: MigrationCandidate) -> _CandidateKey:
    return (
        candidate.from_ref.namespace,
        candidate.from_ref.furu_hash,
        candidate.from_ref.root,
    )


def _build_cascade_candidates(root: MigrationCandidate) -> list[_CascadeNode]:
    nodes: list[_CascadeNode] = []
    queue: list[_CascadeNode] = [_CascadeNode(candidate=root, parent=None)]
    seen: set[_CandidateKey] = {_candidate_key(root)}

    while queue:
        node = queue.pop(0)
        nodes.append(node)
        for dependent in _find_dependents(node.candidate):
            key = _candidate_key(dependent)
            if key in seen:
                continue
            seen.add(key)
            queue.append(_CascadeNode(candidate=dependent, parent=node.key))
    return nodes


def _find_dependents(candidate: MigrationCandidate) -> list[MigrationCandidate]:
    metadata = MetadataManager.read_metadata_raw(candidate.from_ref.directory)
    if metadata is None:
        return []
    furu_obj = metadata.get("furu_obj")
    if not isinstance(furu_obj, dict):
        return []

    from_hash = candidate.from_ref.furu_hash
    dependents: list[MigrationCandidate] = []

    for ref, config in _iter_all_configs():
        updated_config, changed = _replace_dependency(
            config, from_hash, candidate.to_config
        )
        if not changed:
            continue
        dependent_namespace = _extract_namespace(updated_config)
        target_class = _resolve_target_class(dependent_namespace)
        dependent_candidate = _build_candidate(
            ref,
            updated_config,
            to_namespace=dependent_namespace,
            target_class=target_class,
            default_values=None,
            default_fields=None,
            drop_fields=None,
            default_source=None,
        )
        if (
            dependent_candidate.to_ref.furu_hash
            == dependent_candidate.from_ref.furu_hash
        ):
            continue
        dependents.append(dependent_candidate)
    return dependents


def _replace_dependency(
    value: JsonValue,
    from_hash: str,
    to_config: dict[str, JsonValue],
) -> tuple[JsonValue, bool]:
    if isinstance(value, dict):
        if "__class__" in value:
            if FuruSerializer.compute_hash(value) == from_hash:
                return dict(to_config), True
        changed = False
        updated: dict[str, JsonValue] = {}
        for key, item in value.items():
            new_value, was_changed = _replace_dependency(item, from_hash, to_config)
            if was_changed:
                changed = True
            updated[key] = new_value
        return updated, changed
    if isinstance(value, list):
        updated_list: list[JsonValue] = []
        changed = False
        for item in value:
            new_value, was_changed = _replace_dependency(item, from_hash, to_config)
            if was_changed:
                changed = True
            updated_list.append(new_value)
        return updated_list, changed
    return value, False


def _apply_single_migration(
    candidate: MigrationCandidate,
    *,
    policy: MigrationPolicy,
    origin: str | None,
    note: str | None,
    conflict: MigrationConflict,
    conflict_status: str | None,
) -> MigrationRecord:
    from_dir = candidate.from_ref.directory
    to_dir = candidate.to_ref.directory

    if conflict == "overwrite" and to_dir.exists():
        shutil.rmtree(to_dir)

    to_dir.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    if policy in {"move", "copy"}:
        _transfer_payload(from_dir, to_dir, policy)
        _copy_state(from_dir, to_dir, clear_source=policy == "move")
    else:
        _write_migrated_state(to_dir)

    to_obj = FuruSerializer.from_dict(candidate.to_config)
    metadata = MetadataManager.create_metadata(to_obj, to_dir, ignore_diff=True)
    MetadataManager.write_metadata(metadata, to_dir)

    default_values = _serialize_default_values(candidate.defaults_applied)
    record = MigrationRecord(
        kind=_kind_for_policy(policy),
        policy=policy,
        from_namespace=candidate.from_ref.namespace,
        from_hash=candidate.from_ref.furu_hash,
        from_root=candidate.from_ref.root,
        to_namespace=candidate.to_ref.namespace,
        to_hash=candidate.to_ref.furu_hash,
        to_root=candidate.to_ref.root,
        migrated_at=now,
        overwritten_at=None,
        default_values=default_values,
        origin=origin,
        note=note,
    )
    MigrationManager.write_migration(record, to_dir)

    if policy != "copy":
        from_record = MigrationRecord(
            kind="migrated",
            policy=policy,
            from_namespace=candidate.from_ref.namespace,
            from_hash=candidate.from_ref.furu_hash,
            from_root=candidate.from_ref.root,
            to_namespace=candidate.to_ref.namespace,
            to_hash=candidate.to_ref.furu_hash,
            to_root=candidate.to_ref.root,
            migrated_at=now,
            overwritten_at=None,
            default_values=default_values,
            origin=origin,
            note=note,
        )
        MigrationManager.write_migration(from_record, from_dir)

    event: dict[str, str | int] = {
        "type": "migrated",
        "policy": policy,
        "from_namespace": candidate.from_ref.namespace,
        "from_hash": candidate.from_ref.furu_hash,
        "to_namespace": candidate.to_ref.namespace,
        "to_hash": candidate.to_ref.furu_hash,
    }
    if default_values is not None:
        event["default_values"] = json.dumps(default_values, sort_keys=True)
    StateManager.append_event(to_dir, event)
    StateManager.append_event(from_dir, event.copy())

    if conflict == "overwrite" and conflict_status is not None:
        overwrite_event = {
            "type": "migration_overwrite",
            "policy": policy,
            "from_namespace": candidate.from_ref.namespace,
            "from_hash": candidate.from_ref.furu_hash,
            "to_namespace": candidate.to_ref.namespace,
            "to_hash": candidate.to_ref.furu_hash,
            "reason": "force_overwrite",
        }
        StateManager.append_event(to_dir, overwrite_event)
        StateManager.append_event(from_dir, overwrite_event.copy())

    get_logger().info(
        "migration: %s -> %s (%s)",
        from_dir,
        to_dir,
        policy,
    )
    return record


def _transfer_payload(from_dir: Path, to_dir: Path, policy: MigrationPolicy) -> None:
    for item in from_dir.iterdir():
        if item.name == StateManager.INTERNAL_DIR:
            continue
        destination = to_dir / item.name
        if policy == "move":
            shutil.move(str(item), destination)
            continue
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _copy_state(from_dir: Path, to_dir: Path, *, clear_source: bool) -> None:
    src_internal = from_dir / StateManager.INTERNAL_DIR
    if not src_internal.exists():
        return
    dst_internal = to_dir / StateManager.INTERNAL_DIR
    dst_internal.mkdir(parents=True, exist_ok=True)
    state_path = StateManager.get_state_path(from_dir)
    if state_path.is_file():
        shutil.copy2(state_path, StateManager.get_state_path(to_dir))
    success_marker = StateManager.get_success_marker_path(from_dir)
    if success_marker.is_file():
        shutil.copy2(success_marker, StateManager.get_success_marker_path(to_dir))
    if clear_source:
        _write_migrated_state(from_dir)
        StateManager.get_success_marker_path(from_dir).unlink(missing_ok=True)


def _write_migrated_state(directory: Path) -> None:
    def mutate(state) -> None:
        state.result = _StateResultMigrated(status="migrated")
        state.attempt = None

    StateManager.update_state(directory, mutate)


def _kind_for_policy(policy: MigrationPolicy) -> Literal["alias", "moved", "copied"]:
    if policy == "alias":
        return "alias"
    if policy == "move":
        return "moved"
    if policy == "copy":
        return "copied"
    raise ValueError(f"Unsupported migration policy: {policy}")


def _iter_source_configs(
    namespace: str,
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
            furu_obj = metadata.get("furu_obj")
            if not isinstance(furu_obj, dict):
                continue
            root_kind: str = "git" if version_controlled else "data"
            ref = FuruRef(
                namespace=namespace,
                furu_hash=entry.name,
                root=root_kind,
                directory=entry,
            )
            yield ref, _typed_config(furu_obj)


def _iter_all_configs() -> Iterable[tuple[FuruRef, dict[str, JsonValue]]]:
    for version_controlled in (False, True):
        root = FURU_CONFIG.get_root(version_controlled=version_controlled)
        if not root.exists():
            continue
        for namespace_dir in root.rglob("*"):
            if not namespace_dir.is_dir():
                continue
            state_path = StateManager.get_state_path(namespace_dir)
            if not state_path.is_file():
                continue
            metadata = MetadataManager.read_metadata_raw(namespace_dir)
            if metadata is None:
                continue
            furu_obj = metadata.get("furu_obj")
            if not isinstance(furu_obj, dict):
                continue
            namespace = ".".join(namespace_dir.relative_to(root).parts[:-1])
            root_kind: str = "git" if version_controlled else "data"
            ref = FuruRef(
                namespace=namespace,
                furu_hash=namespace_dir.name,
                root=root_kind,
                directory=namespace_dir,
            )
            yield ref, _typed_config(furu_obj)


def _build_candidate(
    from_ref: FuruRef,
    source_config: dict[str, JsonValue],
    *,
    to_namespace: str,
    target_class: type[Furu],
    default_values: Mapping[str, MigrationValue] | None,
    default_fields: Iterable[str] | None,
    drop_fields: Iterable[str] | None,
    default_source: Furu | None,
) -> MigrationCandidate:
    config = dict(source_config)
    defaults_applied: dict[str, MigrationValue] = {}

    fields_dropped = _drop_fields(config, drop_fields)
    target_fields = _target_field_names(target_class)

    default_values_map = dict(default_values) if default_values is not None else {}
    default_fields_list = list(default_fields) if default_fields is not None else []

    overlap = set(default_values_map) & set(default_fields_list)
    if overlap:
        raise ValueError(
            f"migration: default_fields and default_values overlap: {_format_fields(overlap)}"
        )

    existing_fields = set(config.keys()) - {"__class__"}
    remaining_fields = existing_fields - set(fields_dropped)
    if default_values_map:
        conflicts = set(default_values_map) & remaining_fields
        if conflicts:
            raise ValueError(
                f"migration: default_values provided for existing fields: {_format_fields(conflicts)}"
            )
        unknown = set(default_values_map) - set(target_fields)
        if unknown:
            raise ValueError(
                f"migration: default_values contains fields not in target schema: {_format_fields(unknown)}"
            )

    if default_fields_list:
        conflicts = set(default_fields_list) & remaining_fields
        if conflicts:
            raise ValueError(
                f"migration: default_fields provided for existing fields: {_format_fields(conflicts)}"
            )
        unknown = set(default_fields_list) - set(target_fields)
        if unknown:
            raise ValueError(
                f"migration: default_fields contains fields not in target schema: {_format_fields(unknown)}"
            )

    if default_fields_list and default_source is None:
        missing_defaults = _missing_class_defaults(target_class, default_fields_list)
        if missing_defaults:
            raise ValueError(
                f"migration: default_fields missing defaults for fields: {_format_fields(missing_defaults)}"
            )

    for field, value in default_values_map.items():
        defaults_applied[field] = value
        config[field] = _serialize_value(value)

    for field in default_fields_list:
        value = _default_value_for_field(target_class, default_source, field)
        defaults_applied[field] = value
        config[field] = _serialize_value(value)

    config_keys = set(config.keys()) - {"__class__"}
    missing_fields = sorted(set(target_fields) - config_keys)
    if missing_fields:
        raise ValueError(
            f"migration: missing required fields for target class: {_format_fields(missing_fields)}"
        )
    extra_fields = sorted(config_keys - set(target_fields))
    if extra_fields:
        raise ValueError(
            f"migration: extra fields present; use drop_fields to remove: {_format_fields(extra_fields)}"
        )

    config["__class__"] = to_namespace
    to_config = _typed_config(config)
    _typecheck_config(to_config)

    to_hash = FuruSerializer.compute_hash(to_config)
    to_ref = _build_target_ref(target_class, to_namespace, to_hash)

    return MigrationCandidate(
        from_ref=from_ref,
        to_ref=to_ref,
        to_namespace=to_namespace,
        to_config=to_config,
        defaults_applied=defaults_applied,
        fields_dropped=fields_dropped,
        missing_fields=missing_fields,
        extra_fields=extra_fields,
    )


def _drop_fields(
    config: dict[str, JsonValue], drop_fields: Iterable[str] | None
) -> list[str]:
    if drop_fields is None:
        return []
    fields = list(drop_fields)
    unknown = [field for field in fields if field not in config]
    if unknown:
        raise ValueError(
            f"migration: drop_fields contains unknown fields: {_format_fields(unknown)}"
        )
    for field in fields:
        config.pop(field, None)
    return fields


def _target_field_names(target_class: type[Furu]) -> list[str]:
    return [field.logical_name for field in chz.chz_fields(target_class).values()]


def _missing_class_defaults(
    target_class: type[Furu],
    default_fields: list[str],
) -> list[str]:
    fields = chz.chz_fields(target_class)
    missing: list[str] = []
    for name in default_fields:
        field = fields[name]
        if field._default is not CHZ_MISSING:
            continue
        if not isinstance(field._default_factory, MISSING_TYPE):
            continue
        missing.append(name)
    return missing


def _default_value_for_field(
    target_class: type[Furu],
    default_source: Furu | None,
    field_name: str,
) -> MigrationValue:
    if default_source is not None:
        return getattr(default_source, field_name)
    fields = chz.chz_fields(target_class)
    field = fields[field_name]
    if field._default is not CHZ_MISSING:
        return field._default
    if not isinstance(field._default_factory, MISSING_TYPE):
        return field._default_factory()
    raise ValueError(
        f"migration: default_fields missing defaults for fields: {_format_fields([field_name])}"
    )


def _serialize_default_values(
    values: Mapping[str, MigrationValue],
) -> dict[str, JsonValue] | None:
    if not values:
        return None
    return {key: _serialize_value(value) for key, value in values.items()}


def _serialize_value(value: MigrationValue) -> JsonValue:
    result = FuruSerializer.to_dict(value)
    if result is None:
        return result
    if isinstance(result, (str, int, float, bool, list, dict)):
        return result
    raise TypeError(f"Unsupported migration value type: {type(result)}")


def _align_candidate_to_target(
    candidate: MigrationCandidate,
    target_config: dict[str, JsonValue],
) -> MigrationCandidate | None:
    if candidate.to_config != target_config:
        return None
    return candidate


def _typecheck_config(config: dict[str, JsonValue]) -> None:
    obj = FuruSerializer.from_dict(config)
    obj = _normalize_tuple_fields(obj)
    for_all_fields(typecheck)(obj)


def _normalize_tuple_fields(obj: Furu) -> Furu:
    changes: dict[str, object] = {}
    for field in chz.chz_fields(obj).values():
        field_type = field.final_type
        origin = getattr(field_type, "__origin__", None)
        if field_type is tuple or origin is tuple:
            value = getattr(obj, field.logical_name)
            if isinstance(value, list):
                changes[field.logical_name] = tuple(value)
    if not changes:
        return obj
    return chz.replace(obj, **changes)


def _build_target_ref(
    target_class: type[Furu],
    namespace: str,
    furu_hash: str,
) -> FuruRef:
    root_kind: str = "git" if target_class.version_controlled else "data"
    root = FURU_CONFIG.get_root(version_controlled=target_class.version_controlled)
    directory = root / Path(*namespace.split(".")) / furu_hash
    return FuruRef(
        namespace=namespace,
        furu_hash=furu_hash,
        root=root_kind,
        directory=directory,
    )


def _resolve_target_class(namespace: str) -> type[Furu]:
    module_path, _, class_name = namespace.rpartition(".")
    if not module_path:
        raise ValueError(f"migration: unable to resolve target class: {namespace}")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - import errors
        raise ValueError(
            f"migration: unable to resolve target class: {namespace}"
        ) from exc
    obj = getattr(module, class_name, None)
    if obj is None:
        raise ValueError(f"migration: unable to resolve target class: {namespace}")
    if not _is_furu_class(obj):
        raise ValueError(f"migration: unable to resolve target class: {namespace}")
    return cast(type[Furu], obj)


def _is_furu_class(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, Furu)


def _namespace_str(target_class: type[Furu]) -> str:
    return ".".join(target_class._namespace().parts)


def _extract_namespace(config: dict[str, JsonValue]) -> str:
    class_name = config.get("__class__")
    if isinstance(class_name, str):
        return class_name
    raise ValueError("migration: unable to resolve target class: <unknown>")


def _target_status(candidate: MigrationCandidate) -> str | None:
    to_obj = FuruSerializer.from_dict(candidate.to_config)
    state = to_obj.get_state(candidate.to_ref.directory)
    if isinstance(state.result, _StateResultMigrated):
        return None
    if state.result.status == "success":
        return "success"
    attempt = state.attempt
    if isinstance(attempt, _StateAttemptRunning):
        return "running"
    return None


def _expand_skip_keys(
    conflicts: Iterable[_CandidateKey],
    parent_map: dict[_CandidateKey, _CandidateKey | None],
) -> set[_CandidateKey]:
    skipped = set(conflicts)
    changed = True
    while changed:
        changed = False
        for key, parent in parent_map.items():
            if parent is None:
                continue
            if parent in skipped and key not in skipped:
                skipped.add(key)
                changed = True
    return skipped


def _format_fields(fields: Iterable[str]) -> str:
    return ", ".join(sorted(fields))


def _typed_config(config: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {str(key): value for key, value in config.items()}
