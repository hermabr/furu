from __future__ import annotations

import json

import pytest

import furu
import furu.execution as execution_module
from furu import Spec
from furu._storage_layout import (
    compute_lock_path_in,
    result_link_path_in,
    result_manifest_path_in,
)
from furu.utils import JsonValue


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


_COUNTER = _Counter()


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _COUNTER.calls = 0


class _OldRun(Spec[dict[str, str]]):
    learning_rate: float
    dataset: str

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "learning_rate": str(self.learning_rate)}


def _from_old_run(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "dataset": fields["dataset"],
        "lr": fields["learning_rate"],
        "seed": 0,
    }


class _NewRun(Spec[dict[str, str]]):
    dataset: str
    lr: float
    seed: int = 0

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        old = _OldRun(learning_rate=0.0, dataset="ignored")
        return (
            furu.Migration(
                old_fully_qualified_name=old._fully_qualified_name,
                old_schema_hash=old._artifact_schema_hash,
                new_fully_qualified_name=cls(
                    dataset="ignored", lr=0.0
                )._fully_qualified_name,
                new_schema_hash=cls(dataset="ignored", lr=0.0)._artifact_schema_hash,
                transform_fn=_from_old_run,
            ),
        )

    def create(self) -> dict[str, str]:
        _COUNTER.calls += 1
        return {"dataset": self.dataset, "lr": str(self.lr), "seed": str(self.seed)}


class _NoMigrations(Spec[int]):
    n: int

    def create(self) -> int:
        return self.n


def test_migrations_default_to_empty_tuple() -> None:
    assert _NoMigrations(n=1).migrations() == ()


def test_migrate_no_matching_source_returns_false() -> None:
    obj = _NewRun(dataset="cifar10", lr=0.001)
    assert obj.migrate() is False
    assert obj.is_migrated() is False
    assert obj.status == "missing"


def test_migrate_with_matching_old_artifact_writes_result_link() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    assert old.create() == {"dataset": "cifar10", "learning_rate": "0.001"}

    new = _NewRun(dataset="cifar10", lr=0.001)
    assert new.is_migrated() is False

    assert new.migrate() is True
    assert new.is_migrated() is True
    assert new.status == "done"

    link_path = result_link_path_in(new._base_dir)
    link = json.loads(link_path.read_text())
    assert link["current"]["fully_qualified_name"] == new._fully_qualified_name
    assert link["current"]["schema_hash"] == new._artifact_schema_hash
    assert link["current"]["artifact_hash"] == new._artifact_hash
    assert link["source"]["base_dir"] == str(old._base_dir)
    assert link["source"]["schema_hash"] == old._artifact_schema_hash
    assert link["source"]["artifact_hash"] == old._artifact_hash
    assert link["migration_path"][0]["old_schema_hash"] == old._artifact_schema_hash
    assert link["migration_path"][0]["new_schema_hash"] == new._artifact_schema_hash


def test_migrate_loads_result_through_link() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    _COUNTER.calls = 0

    new = _NewRun(dataset="cifar10", lr=0.001)
    assert new.migrate() is True

    assert new.create() == {
        "dataset": "cifar10",
        "learning_rate": "0.001",
    }
    assert _COUNTER.calls == 0

    assert new.load_existing() == {"dataset": "cifar10", "learning_rate": "0.001"}


def test_migrate_returns_true_when_already_migrated() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()

    new = _NewRun(dataset="cifar10", lr=0.001)
    assert new.migrate() is True
    assert new.migrate() is True


def test_migrate_raises_when_existing_link_source_is_missing() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()

    new = _NewRun(dataset="cifar10", lr=0.001)
    assert new.migrate() is True

    result_manifest_path_in(old._base_dir).unlink()

    with pytest.raises(RuntimeError, match="points to a missing result"):
        new.migrate()
    with pytest.raises(RuntimeError, match="points to a missing result"):
        new.load_existing()


def test_migrate_returns_false_when_already_has_direct_result() -> None:
    new = _NewRun(dataset="cifar10", lr=0.001)
    new.create()
    assert new.migrate() is False
    assert not result_link_path_in(new._base_dir).exists()


def test_create_does_not_call_migrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()
    _COUNTER.calls = 0

    migrate_calls = 0
    real_migrate = furu.migration.migrate

    def spy_migrate(obj: Spec) -> bool:
        nonlocal migrate_calls
        migrate_calls += 1
        return real_migrate(obj)

    monkeypatch.setattr(furu.migration, "migrate", spy_migrate)

    new = _NewRun(dataset="cifar10", lr=0.001)
    new.create()
    assert migrate_calls == 0
    assert _COUNTER.calls == 1
    assert not result_link_path_in(new._base_dir).exists()


def test_status_treats_migrated_object_as_completed() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()

    new = _NewRun(dataset="cifar10", lr=0.001)
    assert new.status == "missing"
    new.migrate()
    assert new.status == "done"


class _MidRun(Spec[dict[str, str]]):
    dataset: str
    lr: float

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        old = _OldRun(learning_rate=0.0, dataset="x")
        mid = cls(dataset="x", lr=0.0)
        return (
            furu.Migration(
                old_fully_qualified_name=old._fully_qualified_name,
                old_schema_hash=old._artifact_schema_hash,
                new_fully_qualified_name=mid._fully_qualified_name,
                new_schema_hash=mid._artifact_schema_hash,
                transform_fn=_old_to_mid,
            ),
        )

    def create(self) -> dict[str, str]:
        return {"dataset": self.dataset, "lr": str(self.lr)}


def _old_to_mid(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"dataset": fields["dataset"], "lr": fields["learning_rate"]}


class _FinalRun(Spec[dict[str, str]]):
    dataset: str
    lr: float
    seed: int = 0

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        old = _OldRun(learning_rate=0.0, dataset="x")
        mid = _MidRun(dataset="x", lr=0.0)
        final = cls(dataset="x", lr=0.0)
        return (
            furu.Migration(
                old_fully_qualified_name=old._fully_qualified_name,
                old_schema_hash=old._artifact_schema_hash,
                new_fully_qualified_name=mid._fully_qualified_name,
                new_schema_hash=mid._artifact_schema_hash,
                transform_fn=_old_to_mid,
            ),
            furu.Migration(
                old_fully_qualified_name=mid._fully_qualified_name,
                old_schema_hash=mid._artifact_schema_hash,
                new_fully_qualified_name=final._fully_qualified_name,
                new_schema_hash=final._artifact_schema_hash,
                transform_fn=_mid_to_final,
            ),
        )

    def create(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "lr": str(self.lr),
            "seed": str(self.seed),
        }


def _mid_to_final(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"dataset": fields["dataset"], "lr": fields["lr"], "seed": 0}


def test_multi_hop_migration_points_directly_at_ultimate_source() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()

    mid = _MidRun(dataset="cifar10", lr=0.001)
    assert mid.migrate() is True
    assert mid.is_migrated() is True

    final = _FinalRun(dataset="cifar10", lr=0.001)
    assert final.migrate() is True

    link = json.loads(result_link_path_in(final._base_dir).read_text())
    assert link["source"]["base_dir"] == str(old._base_dir)
    assert link["source"]["fully_qualified_name"] == old._fully_qualified_name
    assert [hop["new_schema_hash"] for hop in link["migration_path"]] == [
        mid._artifact_schema_hash,
        final._artifact_schema_hash,
    ]

    assert final.create() == {
        "dataset": "cifar10",
        "learning_rate": "0.001",
    }


def test_multi_hop_works_without_intermediate_migrate() -> None:
    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()

    final = _FinalRun(dataset="cifar10", lr=0.001)
    assert final.migrate() is True

    link = json.loads(result_link_path_in(final._base_dir).read_text())
    assert link["source"]["base_dir"] == str(old._base_dir)
    assert [hop["new_schema_hash"] for hop in link["migration_path"]] == [
        _MidRun(dataset="cifar10", lr=0.001)._artifact_schema_hash,
        final._artifact_schema_hash,
    ]


def test_create_post_lock_check_uses_result_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from pathlib import Path

    old = _OldRun(learning_rate=0.001, dataset="cifar10")
    old.create()

    new = _NewRun(dataset="cifar10", lr=0.001)

    @contextmanager
    def fake_lock(lock_paths: list[Path], **_: object):
        assert lock_paths == [compute_lock_path_in(new._base_dir)]
        new.migrate()
        yield lambda: True

    monkeypatch.setattr(execution_module, "lock", fake_lock)

    _COUNTER.calls = 0
    assert new.create() == {"dataset": "cifar10", "learning_rate": "0.001"}
    assert _COUNTER.calls == 0
