from __future__ import annotations

from typing import Literal, Mapping

from .core.furu import Furu
from .migration import (
    MigrationValue,
    apply_migration,
    find_migration_candidates_initialized_target,
)
from .storage import MigrationRecord


MigrationPolicy = Literal["alias", "move", "copy"]


def migrate(
    from_obj: Furu,
    to_obj: Furu,
    *,
    policy: MigrationPolicy = "alias",
    origin: str | None = None,
    note: str | None = None,
    default_values: Mapping[str, MigrationValue] | None = None,
) -> MigrationRecord:
    from_namespace = ".".join(from_obj._namespace().parts)
    candidates = find_migration_candidates_initialized_target(
        to_obj=to_obj,
        from_namespace=from_namespace,
        default_fields=None,
        drop_fields=None,
    )
    if not candidates:
        raise ValueError("migration: no candidates found for initialized target")
    if len(candidates) != 1:
        raise ValueError("migration: expected exactly one candidate")
    candidate = candidates[0]
    if default_values:
        candidate = candidate.with_default_values(default_values)
    records = apply_migration(
        candidate,
        policy=policy,
        cascade=True,
        origin=origin,
        note=note,
        conflict="throw",
    )
    return records[0]
