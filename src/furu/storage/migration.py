from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..config import FURU_CONFIG
from ..serialization.serializer import JsonValue


RootKind = Literal["data", "git"]
MigrationPolicy = Literal["alias", "move", "copy"]
MigrationKind = Literal["alias", "moved", "copied", "migrated"]


class MigrationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    kind: MigrationKind
    policy: MigrationPolicy
    from_namespace: str
    from_hash: str
    from_root: RootKind
    to_namespace: str
    to_hash: str
    to_root: RootKind
    migrated_at: str
    overwritten_at: str | None = None
    default_values: dict[str, JsonValue] | None = None
    origin: str | None = None
    note: str | None = None


class MigrationManager:
    INTERNAL_DIR = ".furu"
    MIGRATION_FILE = "migration.json"

    @classmethod
    def get_migration_path(cls, directory: Path) -> Path:
        return directory / cls.INTERNAL_DIR / cls.MIGRATION_FILE

    @classmethod
    def read_migration(cls, directory: Path) -> MigrationRecord | None:
        path = cls.get_migration_path(directory)
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return MigrationRecord.model_validate(data)

    @classmethod
    def write_migration(cls, record: MigrationRecord, directory: Path) -> None:
        path = cls.get_migration_path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record.model_dump(mode="json"), indent=2))
        tmp.replace(path)

    @classmethod
    def resolve_dir(
        cls, record: MigrationRecord, *, target: Literal["from", "to"]
    ) -> Path:
        if target == "from":
            namespace = record.from_namespace
            furu_hash = record.from_hash
            root_kind = record.from_root
        else:
            namespace = record.to_namespace
            furu_hash = record.to_hash
            root_kind = record.to_root
        root = FURU_CONFIG.get_root(version_controlled=root_kind == "git")
        return root / Path(*namespace.split(".")) / furu_hash

    @classmethod
    def root_kind_for_dir(cls, directory: Path) -> RootKind:
        for version_controlled in (False, True):
            root = FURU_CONFIG.get_root(version_controlled=version_controlled)
            if directory.is_relative_to(root):
                return "git" if version_controlled else "data"
        raise ValueError(f"Directory {directory} is not under a Furu root")
