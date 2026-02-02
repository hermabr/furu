import importlib.metadata
import json
import subprocess

import pytest

import furu
from furu.storage.metadata import GitInfo, clear_metadata_cache


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

    assert obj.get() == 42
    meta = obj.get_metadata()
    assert meta.furu_hash == obj.furu_hash
    assert meta.furu_obj["value"] == 42
    assert meta.schema_key == ("value",)
    assert meta.git_commit == "<test>"
    assert meta.furu_package_version == importlib.metadata.version("furu")


def test_metadata_read_raises_when_missing(furu_tmp_root, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        furu.MetadataManager.read_metadata(tmp_path / "missing")


def test_collect_git_info_requires_git(monkeypatch) -> None:
    clear_metadata_cache()

    def boom(_args):
        raise FileNotFoundError("git")

    monkeypatch.setattr(furu.MetadataManager, "run_git_command", boom)
    monkeypatch.setattr(furu.FURU_CONFIG, "record_git", "uncached")
    monkeypatch.setattr(furu.FURU_CONFIG, "allow_no_git_origin", False)

    with pytest.raises(RuntimeError, match="FURU_RECORD_GIT=ignore"):
        furu.MetadataManager.collect_git_info(ignore_diff=True)


def test_collect_git_info_requires_git_remote(monkeypatch) -> None:
    clear_metadata_cache()

    def run_git_command(args):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "main"
        if args == ["remote", "get-url", "origin"]:
            raise subprocess.CalledProcessError(1, ["git", *args])
        if args == ["submodule", "status"]:
            return ""
        return ""

    monkeypatch.setattr(furu.MetadataManager, "run_git_command", run_git_command)
    monkeypatch.setattr(furu.FURU_CONFIG, "record_git", "uncached")
    monkeypatch.setattr(furu.FURU_CONFIG, "allow_no_git_origin", False)

    with pytest.raises(
        RuntimeError,
        match="FURU_ALLOW_NO_GIT_ORIGIN=1.*FURU_RECORD_GIT=ignore",
    ):
        furu.MetadataManager.collect_git_info(ignore_diff=True)


def test_collect_git_info_allows_missing_git_remote(monkeypatch) -> None:
    clear_metadata_cache()

    def run_git_command(args):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "main"
        if args == ["remote", "get-url", "origin"]:
            raise subprocess.CalledProcessError(1, ["git", *args])
        if args == ["submodule", "status"]:
            return ""
        return ""

    monkeypatch.setattr(furu.MetadataManager, "run_git_command", run_git_command)
    monkeypatch.setattr(furu.FURU_CONFIG, "record_git", "uncached")
    monkeypatch.setattr(furu.FURU_CONFIG, "allow_no_git_origin", True)

    git_info = furu.MetadataManager.collect_git_info(ignore_diff=True)
    assert git_info.git_remote is None


def test_collect_git_info_ignore_skips_git(monkeypatch) -> None:
    clear_metadata_cache()

    def boom(_args):
        raise AssertionError("git should not be called")

    monkeypatch.setattr(furu.MetadataManager, "run_git_command", boom)
    monkeypatch.setattr(furu.FURU_CONFIG, "record_git", "ignore")
    monkeypatch.setattr(furu.FURU_CONFIG, "allow_no_git_origin", False)

    git_info = furu.MetadataManager.collect_git_info(ignore_diff=True)
    assert git_info.git_commit == "<ignored>"
