import json
import sys
import textwrap
from typing import cast

import pytest

import furu
from furu.storage import MetadataManager, MigrationManager


class SourceV1(furu.Furu[int]):
    value: int = furu.chz.field(default=0)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class SourceV2(furu.Furu[int]):
    value: int = furu.chz.field(default=0)
    extra: str = furu.chz.field(default="default")

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class SourceV3(furu.Furu[int]):
    value: int = furu.chz.field(default=0)
    extra: str = furu.chz.field(default="default")
    flag: bool = furu.chz.field(default=True)

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class RenameSource(furu.Furu[int]):
    value: int = furu.chz.field(default=0)
    obsolete: str = furu.chz.field(default="old")

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class RenameTarget(furu.Furu[int]):
    count: int = furu.chz.field()

    def _create(self) -> int:
        (self.furu_dir / "value.txt").write_text(str(self.count))
        return self.count

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


def _define_same_class(source: str) -> type[furu.Furu[int]]:
    namespace = {"furu": furu, "__name__": __name__}
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


def _same_class_v1() -> type[furu.Furu[int]]:
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


def _same_class_v2_required() -> type[furu.Furu[int]]:
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


def test_migrate_by_schema_with_defaults(furu_tmp_root) -> None:
    source = SourceV1(value=5)
    assert source.get() == 5

    report = SourceV2.migrate(
        from_schema=SourceV1.schema_key(),
        from_namespace="test_migrations.SourceV1",
        default_field=("extra",),
        origin="tests",
    )
    assert len(report.records) == 1

    alias_obj = SourceV2(value=5, extra="default")
    alias_record = MigrationManager.read_migration(alias_obj._base_furu_dir())
    assert alias_record is not None
    assert alias_record.kind == "alias"
    assert alias_record.from_hash == source.furu_hash
    assert alias_obj.get() == 5


def test_migrate_rename_and_drop(furu_tmp_root) -> None:
    source = RenameSource(value=7, obsolete="old")
    assert source.get() == 7

    report = RenameTarget.migrate(
        from_schema=RenameSource.schema_key(),
        from_namespace="test_migrations.RenameSource",
        rename_field={"value": "count"},
        drop_field=("obsolete",),
        origin="tests",
    )
    assert len(report.records) == 1

    alias_obj = RenameTarget(count=7)
    alias_record = MigrationManager.read_migration(alias_obj._base_furu_dir())
    assert alias_record is not None
    assert alias_obj.get() == 7


def test_set_field_requires_missing_field(furu_tmp_root) -> None:
    source = SourceV1(value=4)
    assert source.get() == 4

    with pytest.raises(ValueError, match="set_field already set"):
        SourceV2.migrate(
            from_schema=SourceV1.schema_key(),
            from_namespace="test_migrations.SourceV1",
            set_field={"value": 5},
            origin="tests",
        )


def test_migrate_from_drop_sets_required_field(furu_tmp_root) -> None:
    same_v1 = _same_class_v1()
    source = cast(type, same_v1)(name="mnist")
    assert source.get() == 5

    same_v2 = _same_class_v2_required()
    report = same_v2.migrate(
        from_drop=("language",),
        set_field={"language": "fr"},
        origin="tests",
    )
    assert len(report.records) == 1

    alias_obj = cast(type, same_v2)(name="mnist", language="fr")
    assert alias_obj.get() == 5


def test_alias_uniqueness_guard(furu_tmp_root) -> None:
    source = SourceV1(value=1)
    assert source.get() == 1

    SourceV2.migrate(
        from_schema=SourceV1.schema_key(),
        from_namespace="test_migrations.SourceV1",
        default_field=("extra",),
        origin="tests",
    )

    with pytest.raises(ValueError, match="alias schema already exists"):
        SourceV2.migrate(
            from_schema=SourceV1.schema_key(),
            from_namespace="test_migrations.SourceV1",
            default_field=("extra",),
            origin="tests",
        )

    report = SourceV2.migrate(
        from_schema=SourceV1.schema_key(),
        from_namespace="test_migrations.SourceV1",
        default_field=("extra",),
        conflict="skip",
        origin="tests",
    )
    assert report.records == []
    assert len(report.skips) == 1


def test_alias_chain_flattens_original(furu_tmp_root) -> None:
    source = SourceV1(value=2)
    assert source.get() == 2

    SourceV2.migrate(
        from_schema=SourceV1.schema_key(),
        from_namespace="test_migrations.SourceV1",
        default_field=("extra",),
        origin="tests",
    )

    alias_v2 = SourceV2(value=2, extra="default")
    SourceV3.migrate(
        from_hash=alias_v2.furu_hash,
        from_namespace="test_migrations.SourceV2",
        default_field=("flag",),
        include_alias_sources=True,
        origin="tests",
    )

    alias_v3 = SourceV3(value=2, extra="default", flag=True)
    record = MigrationManager.read_migration(alias_v3._base_furu_dir())
    assert record is not None
    assert record.from_hash == source.furu_hash

    original_ref = alias_v2.original()
    assert original_ref.furu_hash == source.furu_hash
    aliases = source.aliases()
    assert any(ref.furu_hash == alias_v2.furu_hash for ref in aliases)


def test_current_and_stale_refs(furu_tmp_root) -> None:
    current_obj = SourceV1(value=10)
    stale_obj = SourceV1(value=11)
    current_obj.get()
    stale_obj.get()

    metadata_path = MetadataManager.get_metadata_path(stale_obj._base_furu_dir())
    data = json.loads(metadata_path.read_text())
    data["schema_key"] = ["value", "missing"]
    metadata_path.write_text(json.dumps(data, indent=2))

    current_refs = {ref.furu_hash for ref in SourceV1.current()}
    stale_refs = {ref.furu_hash for ref in SourceV1.stale()}
    assert current_obj.furu_hash in current_refs
    assert stale_obj.furu_hash in stale_refs
