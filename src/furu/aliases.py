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
    """
    Yield each existing Furu storage root path.
    
    Yields:
        Path: Each configured storage root path that exists on disk.
    """
    for version_controlled in (False, True):
        root = FURU_CONFIG.get_root(version_controlled)
        if root.exists():
            yield root


def find_experiment_dirs(root: Path) -> list[Path]:
    """
    Locate experiment directories under the given root that contain the internal state file.
    
    Returns:
        list[Path]: Paths to experiment directories whose StateManager.STATE_FILE exists inside a StateManager.INTERNAL_DIR.
    """
    experiments: list[Path] = []

    for furu_dir in root.rglob(StateManager.INTERNAL_DIR):
        if not furu_dir.is_dir():
            continue
        state_file = furu_dir / StateManager.STATE_FILE
        if state_file.is_file():
            experiments.append(furu_dir.parent)

    return experiments


def alias_key(migration: MigrationRecord) -> AliasKey:
    """
    Compute the AliasKey that identifies the source target of an alias migration.
    
    Parameters:
        migration (MigrationRecord): Migration record to derive the key from.
    
    Returns:
        AliasKey: Tuple of (from_namespace, from_hash, from_root) taken from the migration.
    """
    return (migration.from_namespace, migration.from_hash, migration.from_root)


def collect_aliases(
    *, include_inactive: bool = True
) -> dict[AliasKey, list[MigrationRecord]]:
    """
    Collect alias migration records across all discovered Furu storage roots and group them by their alias key.
    
    Only migrations whose `kind` is "alias" are included. When `include_inactive` is False, migrations with a non-None `overwritten_at` are excluded. Each dictionary key is an AliasKey (from_namespace, from_hash, from_root) and the value is a list of MigrationRecord objects for that key in discovery order.
    
    Parameters:
    	include_inactive (bool): If False, exclude alias migrations that have been overwritten; defaults to True.
    
    Returns:
    	aliases (dict[AliasKey, list[MigrationRecord]]): Mapping from alias key to a list of corresponding alias migrations.
    """
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