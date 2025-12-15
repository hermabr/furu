import json

import pytest

import huldra


class Dummy(huldra.Huldra[int]):
    value: int = huldra.chz.field(default=1)

    def _create(self) -> int:
        (self.huldra_dir / "value.json").write_text(json.dumps(self.value))
        return self.value

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "value.json").read_text())


def test_metadata_roundtrip_and_get_metadata(huldra_tmp_root, monkeypatch) -> None:
    monkeypatch.setattr(
        huldra.MetadataManager,
        "collect_git_info",
        lambda ignore_diff=False: {
            "git_commit": "<test>",
            "git_branch": "<test>",
            "git_remote": "<test>",
            "git_patch": "<test>",
            "git_submodules": {},
        },
    )

    obj = Dummy(value=42)
    with pytest.raises(FileNotFoundError):
        obj.get_metadata()

    assert obj.load_or_create() == 42
    meta = obj.get_metadata()
    assert meta["huldra_hash"] == obj.hexdigest
    assert meta["huldra_obj"]["value"] == 42
    assert meta["git_commit"] == "<test>"


def test_metadata_read_raises_when_missing(huldra_tmp_root, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        huldra.MetadataManager.read_metadata(tmp_path / "missing")
