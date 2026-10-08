import json

import pytest

from furu import Spec
from furu.storage._layout import schema_snapshot_path_in


class SnapshotValue(Spec[str]):
    name: str

    def create(self) -> str:
        return self.name


class FailingValue(Spec[str]):
    name: str

    def create(self) -> str:
        raise RuntimeError("boom")


def test_schema_snapshot_written_on_first_store_and_never_rewritten() -> None:
    first = SnapshotValue(name="first")
    schema_path = schema_snapshot_path_in(first._base_dir)
    assert not schema_path.exists()

    first.create()
    assert json.loads(schema_path.read_text()) == first._schema_data

    stat_before = schema_path.stat()
    second = SnapshotValue(name="second")
    assert schema_snapshot_path_in(second._base_dir) == schema_path
    second.create()
    stat_after = schema_path.stat()
    assert (stat_after.st_ino, stat_after.st_mtime_ns) == (
        stat_before.st_ino,
        stat_before.st_mtime_ns,
    )


def test_schema_snapshot_not_written_when_create_fails() -> None:
    failing = FailingValue(name="x")
    with pytest.raises(RuntimeError, match="boom"):
        failing.create()
    assert not schema_snapshot_path_in(failing._base_dir).exists()
