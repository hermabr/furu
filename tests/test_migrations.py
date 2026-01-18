import json

import json

import gren
from gren.storage import MigrationManager, StateManager
from gren.storage.state import _StateResultMigrated, _StateResultSuccess


class MigrationDummy(gren.Gren[int]):
    value: int = gren.chz.field(default=0)

    def _create(self) -> int:
        (self.gren_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.gren_dir / "value.txt").read_text())


class RenamedSource(gren.Gren[int]):
    value: int = gren.chz.field(default=0)

    def _create(self) -> int:
        (self.gren_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.gren_dir / "value.txt").read_text())


class RenamedTarget(gren.Gren[int]):
    value: int = gren.chz.field(default=0)

    def _create(self) -> int:
        (self.gren_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.gren_dir / "value.txt").read_text())


class AddedFieldV1(gren.Gren[int]):
    value: int = gren.chz.field(default=0)

    def _create(self) -> int:
        (self.gren_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.gren_dir / "value.txt").read_text())


class AddedFieldV2(gren.Gren[int]):
    value: int = gren.chz.field(default=0)
    extra: str = gren.chz.field(default="default")

    def _create(self) -> int:
        (self.gren_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.gren_dir / "value.txt").read_text())


def _events_for(directory) -> list[dict[str, str | int]]:
    path = StateManager.get_events_path(directory)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_migrate_move_transfers_payload(gren_tmp_root) -> None:
    from_obj = MigrationDummy(value=1)
    to_obj = MigrationDummy(value=2)

    assert from_obj.load_or_create() == 1

    gren.migrate(from_obj, to_obj, policy="move", origin="tests")

    from_dir = from_obj._base_gren_dir()
    to_dir = to_obj._base_gren_dir()

    assert not (from_dir / "value.txt").exists()
    assert (to_dir / "value.txt").read_text() == "1"
    assert to_obj.load_or_create() == 1

    to_record = MigrationManager.read_migration(to_dir)
    assert to_record is not None
    assert to_record.kind == "alias"
    assert to_record.policy == "move"

    from_record = MigrationManager.read_migration(from_dir)
    assert from_record is not None
    assert from_record.kind == "migrated"
    assert from_record.policy == "move"


def test_migrate_alias_force_recompute_detaches(gren_tmp_root, monkeypatch) -> None:
    from_obj = MigrationDummy(value=1)
    to_obj = MigrationDummy(value=2)

    assert from_obj.load_or_create() == 1

    gren.migrate(from_obj, to_obj, policy="alias", origin="tests")

    alias_dir = to_obj._base_gren_dir()
    state = StateManager.read_state(alias_dir)
    assert isinstance(state.result, _StateResultMigrated)

    qualname = f"{to_obj.__class__.__module__}.{to_obj.__class__.__qualname__}"
    monkeypatch.setattr(gren.GREN_CONFIG, "force_recompute", {qualname})

    assert to_obj.load_or_create() == 2

    state_after = StateManager.read_state(alias_dir)
    assert isinstance(state_after.result, _StateResultSuccess)
    assert (alias_dir / "value.txt").read_text() == "2"

    alias_record = MigrationManager.read_migration(alias_dir)
    assert alias_record is not None
    assert alias_record.overwritten_at is not None

    original_record = MigrationManager.read_migration(from_obj._base_gren_dir())
    assert original_record is not None
    assert original_record.overwritten_at is not None

    alias_events = [
        event
        for event in _events_for(alias_dir)
        if event.get("type") == "migration_overwrite"
    ]
    original_events = [
        event
        for event in _events_for(from_obj._base_gren_dir())
        if event.get("type") == "migration_overwrite"
    ]
    assert len(alias_events) == 1
    assert len(original_events) == 1
    assert alias_events[0].get("reason") == "force_recompute"
    assert original_events[0].get("reason") == "force_recompute"


def test_migrate_alias_for_rename(gren_tmp_root) -> None:
    from_obj = RenamedSource(value=10)
    to_obj = RenamedTarget(value=10)

    assert from_obj.load_or_create() == 10

    gren.migrate(from_obj, to_obj, policy="alias", origin="tests", note="rename")

    alias_state = StateManager.read_state(to_obj._base_gren_dir())
    assert isinstance(alias_state.result, _StateResultMigrated)

    alias_record = MigrationManager.read_migration(to_obj._base_gren_dir())
    assert alias_record is not None
    assert alias_record.kind == "alias"
    assert alias_record.from_namespace.endswith("RenamedSource")
    assert alias_record.to_namespace.endswith("RenamedTarget")

    assert to_obj.load_or_create() == 10


def test_migrate_alias_with_added_field_default(gren_tmp_root) -> None:
    from_obj = AddedFieldV1(value=5)
    to_obj = AddedFieldV2(value=5)

    assert from_obj.load_or_create() == 5

    gren.migrate(from_obj, to_obj, policy="alias", origin="tests", note="added-field")

    alias_record = MigrationManager.read_migration(to_obj._base_gren_dir())
    assert alias_record is not None
    assert alias_record.kind == "alias"

    alias_state = StateManager.read_state(to_obj._base_gren_dir())
    assert isinstance(alias_state.result, _StateResultMigrated)

    assert to_obj.load_or_create() == 5
