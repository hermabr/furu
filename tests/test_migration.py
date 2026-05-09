from __future__ import annotations

import json
from typing import ClassVar, cast

import pytest
from pydantic import JsonValue

import furu


class TrainJobV1(furu.Furu[str]):
    dataset: str
    learning_rate: float
    create_calls: ClassVar[int] = 0

    def _create(self) -> str:
        type(self).create_calls += 1
        return f"trained:{self.dataset}:{self.learning_rate}"


class TrainingRunV2(furu.Furu[str]):
    dataset: str
    lr: float
    seed: int = 0
    create_calls: ClassVar[int] = 0

    @staticmethod
    def _from_train_job_v1(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            "dataset": fields["dataset"],
            "lr": fields["learning_rate"],
            "seed": 0,
        }

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        return (
            furu.Migration(
                old_fully_qualified_name=TrainJobV1(
                    dataset="", learning_rate=0.0
                )._fully_qualified_name,
                old_schema_hash=TrainJobV1(
                    dataset="", learning_rate=0.0
                ).artifact_schema_hash,
                new_fully_qualified_name=cls(dataset="", lr=0.0)._fully_qualified_name,
                new_schema_hash=cls(dataset="", lr=0.0).artifact_schema_hash,
                transform_fn=cls._from_train_job_v1,
            ),
        )

    def _create(self) -> str:
        type(self).create_calls += 1
        return f"trained-v2:{self.dataset}:{self.lr}:{self.seed}"


class DuplicateMigration(furu.Furu[str]):
    def _create(self) -> str:
        return "created"

    @staticmethod
    def _same(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return fields

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        edge = furu.Migration(
            old_fully_qualified_name=TrainJobV1(
                dataset="", learning_rate=0.0
            )._fully_qualified_name,
            old_schema_hash=TrainJobV1(
                dataset="", learning_rate=0.0
            ).artifact_schema_hash,
            new_fully_qualified_name=cls()._fully_qualified_name,
            new_schema_hash=cls().artifact_schema_hash,
            transform_fn=cls._same,
        )
        return (edge, edge)


class ReturnsFuruObject(furu.Furu[str]):
    dataset: str
    lr: float

    @staticmethod
    def _bad(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            TrainingRunV2(dataset=str(fields["dataset"]), lr=0.001),
        )

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        return (
            furu.Migration(
                old_fully_qualified_name=TrainJobV1(
                    dataset="", learning_rate=0.0
                )._fully_qualified_name,
                old_schema_hash=TrainJobV1(
                    dataset="", learning_rate=0.0
                ).artifact_schema_hash,
                new_fully_qualified_name=cls(dataset="", lr=0.0)._fully_qualified_name,
                new_schema_hash=cls(dataset="", lr=0.0).artifact_schema_hash,
                transform_fn=cls._bad,
            ),
        )

    def _create(self) -> str:
        return "created"


class PrefixedHashMigration(furu.Furu[str]):
    def _create(self) -> str:
        return "created"

    @classmethod
    def migrations(cls) -> tuple[furu.Migration, ...]:
        return (
            furu.Migration(
                old_fully_qualified_name=TrainJobV1(
                    dataset="", learning_rate=0.0
                )._fully_qualified_name,
                old_schema_hash="furu-schema-sha256-v1:abc",
                new_fully_qualified_name=cls()._fully_qualified_name,
                new_schema_hash=cls().artifact_schema_hash,
                transform_fn=lambda fields: fields,
            ),
        )


def test_migration_reuses_old_result_and_writes_result_link() -> None:
    TrainJobV1.create_calls = 0
    TrainingRunV2.create_calls = 0

    old = TrainJobV1(dataset="cifar10", learning_rate=0.001)
    assert old.load_or_create() == "trained:cifar10:0.001"

    new = TrainingRunV2(dataset="cifar10", lr=0.001, seed=0)

    assert new.status() == "completed"
    assert new.load_or_create() == "trained:cifar10:0.001"
    assert TrainingRunV2.create_calls == 0
    assert new.result_path == old._result_dir
    assert not new._result_manifest_path.exists()
    assert new._result_link_path.exists()

    marker = json.loads(new._result_link_path.read_text())
    assert marker["kind"] == "result_link"
    assert marker["source"]["data_dir"] == str(old.data_dir)
    assert marker["migration_path"] == [
        {
            "old_fully_qualified_name": old._fully_qualified_name,
            "old_schema_hash": old.artifact_schema_hash,
            "new_fully_qualified_name": new._fully_qualified_name,
            "new_schema_hash": new.artifact_schema_hash,
        }
    ]


def test_invalid_result_link_is_ignored_and_recomputed() -> None:
    TrainJobV1(dataset="cifar10", learning_rate=0.001).load_or_create()
    new = TrainingRunV2(dataset="cifar10", lr=0.001, seed=0)
    assert new.load_or_create() == "trained:cifar10:0.001"

    marker = json.loads(new._result_link_path.read_text())
    marker["current"]["artifact_hash"] = "bad"
    new._result_link_path.write_text(json.dumps(marker))

    TrainingRunV2.create_calls = 0
    assert new.load_or_create() == "trained:cifar10:0.001"
    assert TrainingRunV2.create_calls == 0


def test_result_link_is_ignored_when_source_result_is_missing() -> None:
    old = TrainJobV1(dataset="cifar10", learning_rate=0.001)
    old.load_or_create()
    new = TrainingRunV2(dataset="cifar10", lr=0.001, seed=0)
    new.load_or_create()

    old.delete(mode="force")

    TrainingRunV2.create_calls = 0
    assert new.status() == "missing"
    assert new.load_or_create() == "trained-v2:cifar10:0.001:0"
    assert TrainingRunV2.create_calls == 1


def test_duplicate_migration_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate migration edge"):
        DuplicateMigration().status()


def test_migration_transform_must_return_fields_not_furu_object() -> None:
    TrainJobV1(dataset="cifar10", learning_rate=0.001).load_or_create()

    with pytest.raises(TypeError, match="must return fields"):
        ReturnsFuruObject(dataset="cifar10", lr=0.001).load_or_create()


def test_schema_hashes_must_be_raw_lowercase_hashes() -> None:
    with pytest.raises(ValueError, match="raw lowercase hexadecimal hash"):
        PrefixedHashMigration().status()
