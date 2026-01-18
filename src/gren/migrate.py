from __future__ import annotations

import datetime as _dt
import json
import shutil
from typing import Mapping
from pathlib import Path
from typing import Literal

from .core.gren import Gren
from .runtime.logging import get_logger
from .serialization import DEFAULT_GREN_VERSION, GrenSerializer
from .storage import MetadataManager, MigrationManager, MigrationRecord, StateManager
from .storage.state import _StateResultMigrated, _StateResultSuccess


MigrationPolicy = Literal["alias", "move", "copy"]


def migrate(
    from_obj: Gren,
    to_obj: Gren,
    *,
    policy: MigrationPolicy = "alias",
    origin: str | None = None,
    note: str | None = None,
    default_values: Mapping[str, str | int | float | bool] | None = None,
) -> MigrationRecord:
    if policy not in {"alias", "move", "copy"}:
        raise ValueError(f"Unsupported migration policy: {policy}")

    to_dict = GrenSerializer.to_dict(to_obj)
    if not isinstance(to_dict, dict):
        raise TypeError("Migration target must serialize to a dict")
    gren_version = to_dict.get("gren_version")
    if not isinstance(gren_version, (float, int)):
        gren_version = DEFAULT_GREN_VERSION
    gren_version = float(gren_version)
    if gren_version != DEFAULT_GREN_VERSION:
        raise ValueError(
            f"Migration target must use current gren_version={DEFAULT_GREN_VERSION}"
        )

    from_dir = from_obj.gren_dir
    to_dir = to_obj.gren_dir

    from_state = StateManager.read_state(from_dir)
    if not isinstance(from_state.result, _StateResultSuccess):
        get_logger().warning(
            "migration source is not successful: %s (%s)",
            from_dir,
            from_state.result.status,
        )

    to_dir.mkdir(parents=True, exist_ok=True)

    from_root = _root_kind(from_obj)
    to_root = _root_kind(to_obj)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    from_namespace = _namespace_str(from_obj)
    to_namespace = _namespace_str(to_obj)

    from_dict = GrenSerializer.to_dict(from_obj)
    if not isinstance(from_dict, dict):
        raise TypeError("Migration source must serialize to a dict")
    from_version_value = from_dict.get("gren_version", DEFAULT_GREN_VERSION)
    if not isinstance(from_version_value, (float, int)):
        from_version_value = DEFAULT_GREN_VERSION

    record = MigrationRecord(
        kind="alias",
        policy=policy,
        from_namespace=from_namespace,
        from_hash=from_obj._gren_hash,
        from_version=float(from_version_value),
        from_root=from_root,
        to_namespace=to_namespace,
        to_hash=to_obj._gren_hash,
        to_version=gren_version,
        to_root=to_root,
        migrated_at=now,
        overwritten_at=None,
        default_values=dict(default_values) if default_values is not None else None,
        origin=origin,
        note=note,
    )

    if policy in {"move", "copy"}:
        _transfer_payload(from_dir, to_dir, policy)
        _copy_state(from_dir, to_dir)
    else:
        _write_migrated_state(to_dir)

    metadata = MetadataManager.create_metadata(to_obj, to_dir, ignore_diff=True)
    MetadataManager.write_metadata(metadata, to_dir)
    MigrationManager.write_migration(record, to_dir)

    event: dict[str, str | int] = {
        "type": "migrated",
        "policy": policy,
        "from": f"{from_namespace}:{from_obj._gren_hash}",
        "to": f"{to_namespace}:{to_obj._gren_hash}",
    }
    if default_values is not None:
        event["default_values"] = json.dumps(default_values, sort_keys=True)
    StateManager.append_event(to_dir, event)

    if policy != "copy":
        old_record = MigrationRecord(
            kind="migrated",
            policy=policy,
            from_namespace=from_namespace,
            from_hash=from_obj._gren_hash,
            from_version=float(from_version_value),
            from_root=from_root,
            to_namespace=to_namespace,
            to_hash=to_obj._gren_hash,
            to_version=gren_version,
            to_root=to_root,
            migrated_at=now,
            overwritten_at=None,
            default_values=dict(default_values) if default_values is not None else None,
            origin=origin,
            note=note,
        )
        MigrationManager.write_migration(old_record, from_dir)

    StateManager.append_event(from_dir, event.copy())

    get_logger().info(
        "migration: %s -> %s (%s)",
        from_dir,
        to_dir,
        policy,
    )

    if policy == "alias":
        # No-op: state retained in old dir.
        pass

    return record


def _namespace_str(obj: Gren) -> str:
    return ".".join(obj._namespace().parts)


def _root_kind(obj: Gren) -> Literal["data", "git"]:
    return "git" if obj.version_controlled else "data"


def _transfer_payload(from_dir: Path, to_dir: Path, policy: MigrationPolicy) -> None:
    for item in from_dir.iterdir():
        if item.name == StateManager.INTERNAL_DIR:
            continue
        destination = to_dir / item.name
        if policy == "move":
            shutil.move(str(item), destination)
        else:
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)


def _copy_state(from_dir: Path, to_dir: Path) -> None:
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


def _write_migrated_state(directory: Path) -> None:
    def mutate(state) -> None:
        state.result = _StateResultMigrated(status="migrated")
        state.attempt = None

    StateManager.update_state(directory, mutate)
