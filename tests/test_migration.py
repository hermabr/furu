from __future__ import annotations

import json
import shutil
from typing import ClassVar

import pytest
from pydantic import JsonValue

import furu
from furu import DuplicateMigrationError, Furu, Migration
from furu.schema import schema_type
from furu.utils import _hash_dict_deterministically, fully_qualified_name


def _schema_hash_of(cls: type[Furu]) -> str:
    return _hash_dict_deterministically(schema_type(cls, set()))


class V1Job(Furu[dict[str, object]]):
    a: int

    def _create(self) -> dict[str, object]:
        return {"a": self.a, "from": "v1"}


_V1_FQN = fully_qualified_name(V1Job)
_V1_HASH = _schema_hash_of(V1Job)


class V2Job(Furu[dict[str, object]]):
    a: int
    extra: str = "default"

    create_calls: ClassVar[list[int]] = []

    @staticmethod
    def _from_v1(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {"a": fields["a"], "extra": "default"}

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return (
            Migration(
                old_fully_qualified_name=_V1_FQN,
                old_schema_hash=_V1_HASH,
                new_fully_qualified_name=fully_qualified_name(cls),
                new_schema_hash=_schema_hash_of(cls),
                transform_fn=cls._from_v1,
            ),
        )

    def _create(self) -> dict[str, object]:
        type(self).create_calls.append(self.a)
        return {"a": self.a, "extra": self.extra, "from": "v2"}


_V2_FQN = fully_qualified_name(V2Job)
_V2_HASH = _schema_hash_of(V2Job)


class V3Job(Furu[dict[str, object]]):
    a: int
    extra: str = "default"
    label: str = "v3"

    create_calls: ClassVar[list[int]] = []

    @staticmethod
    def _from_v1(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {"a": fields["a"], "extra": "default"}

    @staticmethod
    def _from_v2(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {"a": fields["a"], "extra": fields["extra"], "label": "v3"}

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return (
            Migration(
                old_fully_qualified_name=_V1_FQN,
                old_schema_hash=_V1_HASH,
                new_fully_qualified_name=_V2_FQN,
                new_schema_hash=_V2_HASH,
                transform_fn=cls._from_v1,
            ),
            Migration(
                old_fully_qualified_name=_V2_FQN,
                old_schema_hash=_V2_HASH,
                new_fully_qualified_name=fully_qualified_name(cls),
                new_schema_hash=_schema_hash_of(cls),
                transform_fn=cls._from_v2,
            ),
        )

    def _create(self) -> dict[str, object]:
        type(self).create_calls.append(self.a)
        return {"a": self.a, "extra": self.extra, "label": self.label, "from": "v3"}


class _BadFuruReturningV2(Furu[dict[str, object]]):
    a: int

    @staticmethod
    def _bad_transform(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        a = fields["a"]
        assert isinstance(a, int)
        return V1Job(a=a)  # ty: ignore[invalid-return-type]

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return (
            Migration(
                old_fully_qualified_name=_V1_FQN,
                old_schema_hash=_V1_HASH,
                new_fully_qualified_name=fully_qualified_name(cls),
                new_schema_hash=_schema_hash_of(cls),
                transform_fn=cls._bad_transform,
            ),
        )

    def _create(self) -> dict[str, object]:
        return {"a": self.a, "from": "bad_furu"}


class _BadNonDictReturningV2(Furu[dict[str, object]]):
    a: int

    @staticmethod
    def _bad_transform(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return ["not", "a", "dict"]  # ty: ignore[invalid-return-type]

    @classmethod
    def migrations(cls) -> tuple[Migration, ...]:
        return (
            Migration(
                old_fully_qualified_name=_V1_FQN,
                old_schema_hash=_V1_HASH,
                new_fully_qualified_name=fully_qualified_name(cls),
                new_schema_hash=_schema_hash_of(cls),
                transform_fn=cls._bad_transform,
            ),
        )

    def _create(self) -> dict[str, object]:
        return {"a": self.a, "from": "bad_non_dict"}


@pytest.fixture(autouse=True)
def _reset_create_call_trackers() -> None:
    V2Job.create_calls.clear()
    V3Job.create_calls.clear()


def test_default_migrations_is_empty_tuple() -> None:
    class Plain(Furu[str]):
        x: int

        def _create(self) -> str:
            return f"plain:{self.x}"

    assert Plain.migrations() == ()


def test_migration_returns_old_result_for_matching_fields() -> None:
    v1 = V1Job(a=5)
    v1_result = v1.load_or_create()

    v2 = V2Job(a=5)
    result = v2.load_or_create()

    assert result == v1_result
    assert V2Job.create_calls == []
    assert v2._result_link_path.exists()


def test_migration_does_not_apply_when_no_old_artifact_exists() -> None:
    v2 = V2Job(a=7)
    result = v2.load_or_create()

    assert result == {"a": 7, "extra": "default", "from": "v2"}
    assert V2Job.create_calls == [7]
    assert not v2._result_link_path.exists()


def test_migration_does_not_apply_when_field_values_diverge() -> None:
    V1Job(a=5).load_or_create()

    v2 = V2Job(a=5, extra="different")
    result = v2.load_or_create()

    assert result == {"a": 5, "extra": "different", "from": "v2"}
    assert V2Job.create_calls == [5]
    assert not v2._result_link_path.exists()


def test_status_completed_via_marker_after_migration() -> None:
    V1Job(a=5).load_or_create()
    V2Job(a=5).load_or_create()

    fresh = V2Job(a=5)
    assert fresh.status() == "completed"
    assert V2Job.create_calls == []


def test_status_missing_when_no_migration_path() -> None:
    v2 = V2Job(a=42)
    assert v2.status() == "missing"


def test_marker_reused_on_repeat_calls_without_rescanning() -> None:
    V1Job(a=5).load_or_create()
    first = V2Job(a=5)
    first.load_or_create()
    marker_text = first._result_link_path.read_text()

    second = V2Job(a=5)
    result = second.load_or_create()

    assert result == {"a": 5, "from": "v1"}
    assert second._result_link_path.read_text() == marker_text


def test_marker_invalidated_when_source_disappears() -> None:
    v1 = V1Job(a=5)
    v1.load_or_create()
    v2 = V2Job(a=5)
    v2.load_or_create()

    shutil.rmtree(v1.data_dir)

    fresh = V2Job(a=5)
    assert fresh.status() != "completed"
    result = fresh.load_or_create()

    assert result == {"a": 5, "extra": "default", "from": "v2"}
    assert V2Job.create_calls == [5]


def test_marker_invalidated_when_source_artifact_hash_mismatches() -> None:
    v1 = V1Job(a=5)
    v1.load_or_create()
    v2 = V2Job(a=5)
    v2.load_or_create()

    metadata = json.loads(v1._metadata_path.read_text())
    metadata["artifact"]["hash"] = "0" * len(metadata["artifact"]["hash"])
    v1._metadata_path.write_text(json.dumps(metadata))

    fresh = V2Job(a=5)
    assert fresh.status() != "completed"


def test_multi_step_migration_chains_v1_to_v3() -> None:
    v1 = V1Job(a=5)
    v1_result = v1.load_or_create()

    v3 = V3Job(a=5)
    result = v3.load_or_create()

    assert result == v1_result
    assert V3Job.create_calls == []
    assert v3._result_link_path.exists()


def test_multi_step_migration_prefers_v2_when_both_old_artifacts_exist() -> None:
    V1Job(a=5).load_or_create()
    v2 = V2Job(a=5, extra="default")
    v2_result = v2.load_or_create()

    fresh_v3 = V3Job(a=5)
    result = fresh_v3.load_or_create()

    assert result in (v2_result, V1Job(a=5).load_or_create())
    assert V3Job.create_calls == []


def test_empty_fully_qualified_name_rejected_at_class_definition() -> None:
    with pytest.raises(ValueError, match="old_fully_qualified_name"):

        class Bad(Furu[str]):
            x: int

            @classmethod
            def migrations(cls) -> tuple[Migration, ...]:
                return (
                    Migration(
                        old_fully_qualified_name="",
                        old_schema_hash="abcd1234",
                        new_fully_qualified_name="m.Bad",
                        new_schema_hash="abcd5678",
                        transform_fn=lambda f: f,
                    ),
                )

            def _create(self) -> str:
                return f"bad:{self.x}"


def test_namespaced_schema_hash_rejected() -> None:
    with pytest.raises(ValueError, match="old_schema_hash"):

        class Bad(Furu[str]):
            x: int

            @classmethod
            def migrations(cls) -> tuple[Migration, ...]:
                return (
                    Migration(
                        old_fully_qualified_name="m.Old",
                        old_schema_hash="furu-schema-sha256-v1:abcd1234",
                        new_fully_qualified_name="m.Bad",
                        new_schema_hash="abcd5678",
                        transform_fn=lambda f: f,
                    ),
                )

            def _create(self) -> str:
                return f"bad:{self.x}"


def test_uppercase_hash_rejected() -> None:
    with pytest.raises(ValueError, match="schema_hash"):

        class Bad(Furu[str]):
            x: int

            @classmethod
            def migrations(cls) -> tuple[Migration, ...]:
                return (
                    Migration(
                        old_fully_qualified_name="m.Old",
                        old_schema_hash="ABCD1234",
                        new_fully_qualified_name="m.Bad",
                        new_schema_hash="abcd5678",
                        transform_fn=lambda f: f,
                    ),
                )

            def _create(self) -> str:
                return f"bad:{self.x}"


def test_non_callable_transform_fn_rejected() -> None:
    with pytest.raises(TypeError, match="transform_fn must be callable"):

        class Bad(Furu[str]):
            x: int

            @classmethod
            def migrations(cls) -> tuple[Migration, ...]:
                return (
                    Migration(
                        old_fully_qualified_name="m.Old",
                        old_schema_hash="abcd1234",
                        new_fully_qualified_name="m.Bad",
                        new_schema_hash="abcd5678",
                        transform_fn="not callable",  # ty: ignore[invalid-argument-type]
                    ),
                )

            def _create(self) -> str:
                return f"bad:{self.x}"


def test_non_tuple_migrations_return_rejected() -> None:
    with pytest.raises(TypeError, match="must return a tuple"):

        class Bad(Furu[str]):
            x: int

            @classmethod
            def migrations(cls):  # type: ignore[override]
                return [
                    Migration(
                        old_fully_qualified_name="m.Old",
                        old_schema_hash="abcd1234",
                        new_fully_qualified_name="m.Bad",
                        new_schema_hash="abcd5678",
                        transform_fn=lambda f: f,
                    ),
                ]

            def _create(self) -> str:
                return f"bad:{self.x}"


def test_duplicate_migration_edges_rejected() -> None:
    with pytest.raises(DuplicateMigrationError):

        class Bad(Furu[str]):
            x: int

            @classmethod
            def migrations(cls) -> tuple[Migration, ...]:
                edge = dict(
                    old_fully_qualified_name="m.Old",
                    old_schema_hash="abcd1234",
                    new_fully_qualified_name="m.Bad",
                    new_schema_hash="abcd5678",
                )
                return (
                    Migration(**edge, transform_fn=lambda f: f),
                    Migration(**edge, transform_fn=lambda f: dict(f, extra=1)),
                )

            def _create(self) -> str:
                return f"bad:{self.x}"


def test_transform_fn_returning_furu_object_does_not_match() -> None:
    V1Job(a=99).load_or_create()
    bad = _BadFuruReturningV2(a=99)
    bad.load_or_create()

    assert not bad._result_link_path.exists()


def test_transform_fn_returning_non_dict_does_not_match() -> None:
    V1Job(a=98).load_or_create()
    obj = _BadNonDictReturningV2(a=98)
    obj.load_or_create()

    assert not obj._result_link_path.exists()


def test_marker_invalidated_when_migration_path_no_longer_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    V1Job(a=5).load_or_create()
    v2 = V2Job(a=5)
    v2.load_or_create()
    assert v2._result_link_path.exists()

    monkeypatch.setattr(V2Job, "migrations", classmethod(lambda cls: ()))

    fresh = V2Job(a=5)
    assert fresh.status() != "completed"


def test_module_exports_migration_api() -> None:
    assert furu.Migration is Migration
    assert furu.DuplicateMigrationError is DuplicateMigrationError
