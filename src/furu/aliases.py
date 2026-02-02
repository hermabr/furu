from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from .config import FURU_CONFIG
from .storage import MigrationManager, MigrationRecord
from .storage.migration import RootKind
from .storage.state import StateManager

AliasKey = tuple[str, str, RootKind]


def iter_roots() -> Iterator[Path]:
    """Iterate over all existing Furu storage roots."""
    for version_controlled in (False, True):
        root = FURU_CONFIG.get_root(version_controlled)
        if root.exists():
            yield root


def find_experiment_dirs(root: Path) -> list[Path]:
    """Find all directories containing .furu/state.json files."""
    experiments: list[Path] = []

    for furu_dir in root.rglob(StateManager.INTERNAL_DIR):
        if not furu_dir.is_dir():
            continue
        state_file = furu_dir / StateManager.STATE_FILE
        if state_file.is_file():
            experiments.append(furu_dir.parent)

    return experiments


def alias_key(migration: MigrationRecord) -> AliasKey:
    return (migration.from_namespace, migration.from_hash, migration.from_root)


def collect_aliases(
    *, include_inactive: bool = True
) -> dict[AliasKey, list[MigrationRecord]]:
    aliases: dict[AliasKey, list[MigrationRecord]] = defaultdict(list)
    for root in iter_roots():
        for experiment_dir in find_experiment_dirs(root):
            migration = MigrationManager.read_migration(experiment_dir)
            if migration is None or migration.kind != "alias":
                continue
            if not include_inactive and migration.overwritten_at is not None:
                continue
            aliases[alias_key(migration)].append(migration)
    return aliases
