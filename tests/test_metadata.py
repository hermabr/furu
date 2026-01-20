import json

import pytest

import furu
from furu.storage.metadata import GitInfo


class Dummy(furu.Furu[int]):
    value: int = furu.chz.field(default=1)

    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


def test_metadata_roundtrip_and_get_metadata(furu_tmp_root, monkeypatch) -> None:
    monkeypatch.setattr(
        furu.MetadataManager,
        "collect_git_info",
        lambda ignore_diff=False: GitInfo(
            git_commit="<test>",
            git_branch="<test>",
            git_remote="<test>",
            git_patch="<test>",
            git_submodules={},
        ),
    )

    obj = Dummy(value=42)
    with pytest.raises(FileNotFoundError):
        obj.get_metadata()

    assert obj.load_or_create() == 42
    meta = obj.get_metadata()
    assert meta.furu_hash == obj._furu_hash
    assert meta.furu_obj["value"] == 42
    assert meta.git_commit == "<test>"


def test_metadata_read_raises_when_missing(furu_tmp_root, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        furu.MetadataManager.read_metadata(tmp_path / "missing")
