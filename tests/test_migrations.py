import json
import sys
import textwrap
from typing import cast

import pytest

import furu
from furu.storage import MigrationManager, StateManager
from furu.storage.state import _StateResultMigrated, _StateResultSuccess


class MigrationDummy(furu.Furu[int]):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class GitDummy(furu.Furu[int], version_controlled=True):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class RenamedSource(furu.Furu[int]):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class RenamedTarget(furu.Furu[int]):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class AddedFieldV1(furu.Furu[int]):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class AddedFieldV2(furu.Furu[int]):
    value: int = furu.chz.field(default=0)
    extra: str = furu.chz.field(default="default")

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


def _define_same_class(source: str) -> type:
    namespace: dict[str, object] = {"furu": furu, "__name__": __name__}
    exec(textwrap.dedent(source), namespace)
    cls = namespace.get("SameClass")
    if not isinstance(cls, type):
        raise AssertionError("SameClass definition failed")
    if not issubclass(cls, furu.Furu):
        raise AssertionError("SameClass must be a Furu")
    cls.__module__ = __name__
    cls.__qualname__ = "SameClass"
    module = sys.modules[__name__]
    setattr(module, "SameClass", cls)
    return cls


def _same_class_v1() -> type:
    return _define_same_class(
        """
        class SameClass(furu.Furu[int]):
            name: str = furu.chz.field(default="")

            def _create(self) -> int:
                (self.furu_dir / "value.txt").write_text(self.name)
                return len(self.name)

            def _load(self) -> int:
                return len((self.furu_dir / "value.txt").read_text())
        """
    )


def _same_class_v2_required() -> type:
    return _define_same_class(
        """
        class SameClass(furu.Furu[int]):
            name: str = furu.chz.field(default="")
            language: str = furu.chz.field()

            def _create(self) -> int:
                (self.furu_dir / "value.txt").write_text(f"{self.name}:{self.language}")
                return len(self.name)

            def _load(self) -> int:
                return len((self.furu_dir / "value.txt").read_text().split(\":\")[0])
        """
    )


def _same_class_v2_optional() -> type:
    return _define_same_class(
        """
        class SameClass(furu.Furu[int]):
            name: str = furu.chz.field(default="")
            language: str = furu.chz.field(default="spanish")

            def _create(self) -> int:
                (self.furu_dir / "value.txt").write_text(f"{self.name}:{self.language}")
                return len(self.name)

            def _load(self) -> int:
                return len((self.furu_dir / "value.txt").read_text().split(\":\")[0])
        """
    )


class DataBase(furu.Furu[int]):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        raise NotImplementedError("DataBase does not implement _create")

    def _load(self) -> int:
        raise NotImplementedError("DataBase does not implement _load")


class DataV1(DataBase):
    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class DataV2(DataBase):
    language: str = furu.chz.field(default="english")

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(f"{self.value}:{self.language}")
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text().split(":")[0])


class ExperimentV1(furu.Furu[int]):
    data: DataBase = furu.chz.field()

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.data.value))
        return self.data.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class ExperimentV2(furu.Furu[int]):
    data: DataBase = furu.chz.field()

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.data.value))
        return self.data.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


def _events_for(directory) -> list[dict[str, str | int]]:
    path = StateManager.get_events_path(directory)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_migrate_move_transfers_payload(furu_tmp_root) -> None:
    from_obj = RenamedSource(value=1)
    to_obj = RenamedTarget(value=1)

    assert from_obj.get() == 1

    furu.migrate(from_obj, to_obj, policy="move", origin="tests")

    from_dir = from_obj._base_furu_dir()
    to_dir = to_obj._base_furu_dir()

    assert not (from_dir / "value.txt").exists()
    assert (to_dir / "value.txt").read_text() == "1"
    assert to_obj.get() == 1

    to_record = MigrationManager.read_migration(to_dir)
    assert to_record is not None
    assert to_record.kind == "moved"
    assert to_record.policy == "move"

    from_record = MigrationManager.read_migration(from_dir)
    assert from_record is not None
    assert from_record.kind == "migrated"
    assert from_record.policy == "move"

    from_state = StateManager.read_state(from_dir)
    assert isinstance(from_state.result, _StateResultMigrated)


def test_migrate_alias_always_rerun_detaches(furu_tmp_root, monkeypatch) -> None:
    from_obj = RenamedSource(value=1)
    to_obj = RenamedTarget(value=1)

    assert from_obj.get() == 1

    furu.migrate(from_obj, to_obj, policy="alias", origin="tests")

    alias_dir = to_obj._base_furu_dir()
    state = StateManager.read_state(alias_dir)
    assert isinstance(state.result, _StateResultMigrated)

    qualname = f"{to_obj.__class__.__module__}.{to_obj.__class__.__qualname__}"
    monkeypatch.setattr(furu.FURU_CONFIG, "always_rerun", {qualname})

    assert to_obj.get() == 1

    state_after = StateManager.read_state(alias_dir)
    assert isinstance(state_after.result, _StateResultSuccess)
    assert (alias_dir / "value.txt").read_text() == "1"

    alias_record = MigrationManager.read_migration(alias_dir)
    assert alias_record is not None
    assert alias_record.overwritten_at is not None

    original_record = MigrationManager.read_migration(from_obj._base_furu_dir())
    assert original_record is not None
    assert original_record.overwritten_at is not None

    alias_events = [
        event
        for event in _events_for(alias_dir)
        if event.get("type") == "migration_overwrite"
    ]
    original_events = [
        event
        for event in _events_for(from_obj._base_furu_dir())
        if event.get("type") == "migration_overwrite"
    ]
    assert len(alias_events) == 1
    assert len(original_events) == 1
    assert alias_events[0].get("reason") == "always_rerun"
    assert original_events[0].get("reason") == "always_rerun"


def test_migrate_alias_of_alias_is_skipped(furu_tmp_root) -> None:
    from_obj = RenamedSource(value=2)
    to_obj = RenamedTarget(value=2)

    assert from_obj.get() == 2

    furu.migrate(from_obj, to_obj, policy="alias", origin="tests")

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.RenamedSource",
            to_namespace="test_migrations.RenamedTarget",
        ),
    )
    assert len(candidates) == 1

    results = furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="alias-of-alias",
        conflict="skip",
    )
    assert len(results) == 1
    assert isinstance(results[0], furu.MigrationSkip)


def test_migrate_alias_for_rename(furu_tmp_root) -> None:
    from_obj = RenamedSource(value=10)
    to_obj = RenamedTarget(value=10)

    assert from_obj.get() == 10

    furu.migrate(from_obj, to_obj, policy="alias", origin="tests", note="rename")

    alias_state = StateManager.read_state(to_obj._base_furu_dir())
    assert isinstance(alias_state.result, _StateResultMigrated)

    alias_record = MigrationManager.read_migration(to_obj._base_furu_dir())
    assert alias_record is not None
    assert alias_record.kind == "alias"
    assert alias_record.from_namespace.endswith("RenamedSource")
    assert alias_record.to_namespace.endswith("RenamedTarget")

    assert to_obj.get() == 10


def test_migrate_alias_with_added_field_default(furu_tmp_root) -> None:
    from_obj = AddedFieldV1(value=5)
    to_obj = AddedFieldV2(value=5)

    assert from_obj.get() == 5

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.AddedFieldV1",
            to_namespace="test_migrations.AddedFieldV2",
        ),
        default_values={"extra": "default"},
    )
    assert len(candidates) == 1

    to_config = candidates[0].to_config
    assert furu.FuruSerializer.compute_hash(to_config) == to_obj.furu_hash

    furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="added-field",
    )

    alias_record = MigrationManager.read_migration(to_obj._base_furu_dir())
    assert alias_record is not None
    assert alias_record.kind == "alias"
    assert alias_record.default_values == {"extra": "default"}

    alias_state = StateManager.read_state(to_obj._base_furu_dir())
    assert isinstance(alias_state.result, _StateResultMigrated)

    assert to_obj.get() == 5


def test_migrate_same_class_add_required_field(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    from_obj = same_class_v1(name="mnist")

    assert from_obj.get() == 5

    same_class_v2 = _same_class_v2_required()
    candidates = furu.find_migration_candidates(
        namespace="test_migrations.SameClass",
        to_obj=cast(type[furu.Furu], same_class_v2),
        drop_fields=["name"],
        default_values={"name": "mnist", "language": "english"},
    )
    assert len(candidates) == 1

    furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="add-language",
    )

    to_obj = same_class_v2(name="mnist", language="english")
    alias_record = MigrationManager.read_migration(to_obj._base_furu_dir())
    assert alias_record is not None
    assert alias_record.default_values == {"name": "mnist", "language": "english"}

    alias_state = StateManager.read_state(to_obj._base_furu_dir())
    assert isinstance(alias_state.result, _StateResultMigrated)

    assert to_obj.get() == 5


def test_migrate_same_class_drop_fields_and_defaults(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    from_obj = same_class_v1(name="dummy")

    assert from_obj.get() == 5

    same_class_v2 = _same_class_v2_optional()
    candidates = furu.find_migration_candidates(
        namespace="test_migrations.SameClass",
        to_obj=cast(type[furu.Furu], same_class_v2),
        drop_fields=["name"],
        default_values={"name": "dummy", "language": "english"},
    )
    assert len(candidates) == 1

    furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="drop-and-default",
    )

    to_obj = same_class_v2(name="dummy", language="english")
    alias_record = MigrationManager.read_migration(to_obj._base_furu_dir())
    assert alias_record is not None
    assert alias_record.default_values == {"name": "dummy", "language": "english"}

    alias_state = StateManager.read_state(to_obj._base_furu_dir())
    assert isinstance(alias_state.result, _StateResultMigrated)

    assert to_obj.get() == 5


def test_migrate_default_fields_conflict(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    from_obj = same_class_v1(name="dummy")

    assert from_obj.get() == 5

    same_class_v2 = _same_class_v2_optional()
    with pytest.raises(ValueError, match="default_fields and default_values overlap"):
        furu.find_migration_candidates(
            namespace="test_migrations.SameClass",
            to_obj=cast(type[furu.Furu], same_class_v2),
            default_fields=["language"],
            default_values={"language": "english"},
        )


def test_migrate_default_values_wrong_type(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    from_obj = same_class_v1(name="dummy")

    assert from_obj.get() == 5

    same_class_v2 = _same_class_v2_required()
    with pytest.raises(Exception):
        furu.find_migration_candidates(
            namespace="test_migrations.SameClass",
            to_obj=cast(type[furu.Furu], same_class_v2),
            default_values={"language": 123},
        )


def test_migrate_drop_fields_unknown(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    from_obj = same_class_v1(name="dummy")

    assert from_obj.get() == 5

    same_class_v2 = _same_class_v2_optional()
    with pytest.raises(ValueError, match="drop_fields contains unknown fields"):
        furu.find_migration_candidates(
            namespace="test_migrations.SameClass",
            to_obj=cast(type[furu.Furu], same_class_v2),
            drop_fields=["language"],
        )


def test_migrate_drop_fields_and_default_overlap(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    from_obj = same_class_v1(name="dummy")

    assert from_obj.get() == 5

    same_class_v2 = _same_class_v2_optional()
    with pytest.raises(ValueError, match="default_values provided for existing fields"):
        furu.find_migration_candidates(
            namespace="test_migrations.SameClass",
            to_obj=cast(type[furu.Furu], same_class_v2),
            default_values={"name": "dummy"},
        )


def test_migrate_no_candidates_for_namespace(furu_tmp_root) -> None:
    same_class_v2 = _same_class_v2_required()
    candidates = furu.find_migration_candidates(
        namespace="test_migrations.MissingSource",
        to_obj=cast(type[furu.Furu], same_class_v2),
    )
    assert candidates == []


def test_migrate_invalid_namespace_type(furu_tmp_root) -> None:
    with pytest.raises(
        ValueError, match="migration: namespace must be str or NamespacePair"
    ):
        furu.find_migration_candidates(namespace=123)  # type: ignore[arg-type]


def test_migrate_missing_to_obj_for_string_namespace(furu_tmp_root) -> None:
    same_class_v2 = _same_class_v2_required()
    with pytest.raises(ValueError, match="migration: to_obj must be a class"):
        furu.find_migration_candidates(
            namespace="test_migrations.SameClass",
            to_obj=cast(
                type[furu.Furu], same_class_v2(name="dummy", language="english")
            ),
        )


def test_migrate_initialized_target_requires_instance(furu_tmp_root) -> None:
    same_class_v2 = _same_class_v2_required()
    with pytest.raises(ValueError, match="migration: to_obj must be an instance"):
        furu.find_migration_candidates_initialized_target(  # type: ignore[arg-type, call-arg]
            to_obj=cast(furu.Furu, same_class_v2),
        )


def test_migrate_initialized_target_rejects_non_furu(furu_tmp_root) -> None:
    with pytest.raises(ValueError, match="migration: to_obj must be an instance"):
        furu.find_migration_candidates_initialized_target(  # type: ignore[arg-type, call-arg]
            to_obj=cast(furu.Furu, object()),
        )


def test_migrate_wrapper_rejects_invalid_defaults(furu_tmp_root) -> None:
    same_class_v1 = _same_class_v1()
    same_class_v2 = _same_class_v2_required()
    from_obj = same_class_v1(name="dummy")
    to_obj = same_class_v2(name="dummy", language="english")

    assert from_obj.get() == 5

    with pytest.raises(ValueError, match="missing required fields"):
        furu.migrate(from_obj, to_obj, default_values={"unknown": "bad"})


def test_migrate_git_root_records_git(furu_tmp_root) -> None:
    from_obj = GitDummy(value=3)
    to_obj = GitDummy(value=3)

    assert from_obj.get() == 3
    assert to_obj.get() == 3

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.GitDummy",
            to_namespace="test_migrations.GitDummy",
        ),
    )
    assert len(candidates) == 1

    furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="git-root",
        conflict="overwrite",
    )

    record = MigrationManager.read_migration(to_obj._base_furu_dir())
    assert record is not None
    assert record.from_root == "git"
    assert record.to_root == "git"


def test_migrate_conflict_skip_on_success(furu_tmp_root) -> None:
    from_obj = RenamedSource(value=4)
    to_obj = RenamedTarget(value=4)

    assert from_obj.get() == 4
    assert to_obj.get() == 4

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.RenamedSource",
            to_namespace="test_migrations.RenamedTarget",
        ),
    )
    assert len(candidates) == 1

    results = furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="skip-success",
        conflict="skip",
    )
    assert len(results) == 1
    assert isinstance(results[0], furu.MigrationSkip)


def test_migrate_conflict_overwrite_on_success(furu_tmp_root) -> None:
    from_obj = RenamedSource(value=5)
    to_obj = RenamedTarget(value=5)

    assert from_obj.get() == 5
    assert to_obj.get() == 5

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.RenamedSource",
            to_namespace="test_migrations.RenamedTarget",
        ),
    )
    assert len(candidates) == 1

    results = furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="overwrite-success",
        conflict="overwrite",
    )
    assert len(results) == 1
    record = MigrationManager.read_migration(to_obj._base_furu_dir())
    assert record is not None
    assert results[0].kind == record.kind
    assert results[0].overwritten_at is None

    events = [
        event
        for event in _events_for(to_obj._base_furu_dir())
        if event.get("type") == "migration_overwrite"
    ]
    assert len(events) == 1


def test_migrate_conflict_skip_on_running(furu_tmp_root) -> None:
    from_obj = RenamedSource(value=6)
    to_obj = RenamedTarget(value=6)

    assert from_obj.get() == 6

    running_state = StateManager.read_state(to_obj._base_furu_dir())

    def set_running(state) -> None:
        state.result = running_state.result
        state.attempt = {
            "id": "attempt-running",
            "number": 1,
            "backend": "local",
            "status": "running",
            "started_at": "2025-01-01T00:00:00+00:00",
            "lease_duration_sec": 120.0,
            "lease_expires_at": "2025-01-01T00:12:00+00:00",
            "owner": {"pid": 1, "host": "local", "user": "tester"},
            "scheduler": {},
        }

    StateManager.ensure_internal_dir(to_obj._base_furu_dir())
    StateManager.update_state(to_obj._base_furu_dir(), set_running)

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.RenamedSource",
            to_namespace="test_migrations.RenamedTarget",
        ),
    )
    assert len(candidates) == 1

    results = furu.apply_migration(
        candidates[0],
        policy="alias",
        origin="tests",
        note="skip-running",
        conflict="skip",
    )
    assert len(results) == 1
    assert isinstance(results[0], furu.MigrationSkip)


class CascadeChild(furu.Furu[int]):
    parent: RenamedSource | RenamedTarget

    def _create(self) -> int:
        path = self.parent.furu_dir / "value.txt"
        value = int(path.read_text())
        (self.furu_dir / "value.txt").write_text(str(value))
        return value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


def test_migrate_cascade_skip_conflicts(furu_tmp_root) -> None:
    parent = RenamedSource(value=7)
    child = CascadeChild(parent=parent)

    assert parent.get() == 7
    assert child.get() == 7

    target_parent = RenamedTarget(value=7)
    assert target_parent.get() == 7

    exp_candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.CascadeChild",
            to_namespace="test_migrations.CascadeChild",
        ),
        drop_fields=["parent"],
        default_values={"parent": target_parent},
    )
    assert len(exp_candidates) == 1

    furu.apply_migration(
        exp_candidates[0],
        policy="alias",
        cascade=False,
        origin="tests",
        note="parent-only",
    )

    target_child = CascadeChild(parent=target_parent)
    assert target_child.get() == 7

    candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.RenamedSource",
            to_namespace="test_migrations.RenamedTarget",
        ),
    )
    assert len(candidates) == 1

    results = furu.apply_migration(
        candidates[0],
        policy="alias",
        cascade=True,
        origin="tests",
        note="cascade-skip",
        conflict="skip",
    )
    assert len(results) == 2
    assert all(isinstance(result, furu.MigrationSkip) for result in results)


def test_migrate_cascade_updates_dependents(furu_tmp_root) -> None:
    data_obj = DataV1(value=3)
    exp_obj = ExperimentV1(data=data_obj)

    assert data_obj.get() == 3
    assert exp_obj.get() == 3

    data_candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.DataV1",
            to_namespace="test_migrations.DataV2",
        ),
        drop_fields=["value"],
        default_values={"value": 3, "language": "english"},
    )
    assert len(data_candidates) == 1

    furu.apply_migration(
        data_candidates[0],
        policy="alias",
        cascade=True,
        origin="tests",
        note="cascade-data",
    )

    data_alias = DataV2(value=3, language="english")
    exp_alias = ExperimentV1(data=data_alias)

    assert data_alias.exists() is True
    assert exp_alias.exists() is True

    exp_record = MigrationManager.read_migration(exp_alias._base_furu_dir())
    assert exp_record is not None
    assert exp_record.kind == "alias"

    exp_candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.ExperimentV1",
            to_namespace="test_migrations.ExperimentV2",
        ),
        drop_fields=["data"],
        default_values={"data": data_alias},
    )
    assert len(exp_candidates) == 2


def test_migrate_cascade_after_parent_migration(furu_tmp_root) -> None:
    data_obj = DataV1(value=7)
    exp_obj = ExperimentV1(data=data_obj)

    assert data_obj.get() == 7
    assert exp_obj.get() == 7

    exp_candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.ExperimentV1",
            to_namespace="test_migrations.ExperimentV2",
        ),
        drop_fields=["data"],
        default_values={"data": DataV2(value=7, language="english")},
    )
    assert len(exp_candidates) == 1

    exp_candidates_repeat = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.ExperimentV1",
            to_namespace="test_migrations.ExperimentV2",
        ),
        drop_fields=["data"],
        default_values={"data": DataV2(value=7, language="english")},
    )
    assert len(exp_candidates_repeat) == 1

    furu.apply_migration(
        exp_candidates[0],
        policy="alias",
        cascade=False,
        origin="tests",
        note="parent-only",
    )

    data_candidates = furu.find_migration_candidates(
        namespace=furu.NamespacePair(
            from_namespace="test_migrations.DataV1",
            to_namespace="test_migrations.DataV2",
        ),
        drop_fields=["value"],
        default_values={"value": 7, "language": "english"},
    )
    assert len(data_candidates) == 1

    furu.apply_migration(
        data_candidates[0],
        policy="alias",
        cascade=True,
        origin="tests",
        note="cascade-data",
    )

    data_alias = DataV2(value=7, language="english")
    exp_alias = ExperimentV2(data=data_alias)

    assert data_alias.exists() is True
    assert exp_alias.exists() is True

    exp_record = MigrationManager.read_migration(exp_alias._base_furu_dir())
    assert exp_record is not None
    assert exp_record.kind == "alias"
