import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import furu
from furu import Spec
from furu.config import _Config, get_config
from furu.provenance import EnvironmentIdentity, Provenance
from furu.storage._layout import metadata_path_in, provenance_path_in
from furu.testing import override_config


@pytest.fixture(autouse=True)
def _primed_environment_identity() -> Iterator[None]:
    # Environment capture is cwd-dependent and cached per process. Other test
    # modules clear the cache; re-prime it from the project root before tests
    # here chdir into scratch worktrees without a pyproject.toml.
    EnvironmentIdentity.capture()
    yield


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "tracked.txt").write_text("content\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _with_snapshot_default(enabled: bool) -> _Config:
    data = get_config().model_dump()
    data["provenance"]["snapshot_default"] = enabled
    return _Config.model_validate(data)


_created: list[str] = []


class _Node(Spec[int]):
    value: int = 0

    def create(self) -> int:
        _created.append(self.object_id)
        return self.value + 1


def test_create_writes_provenance_next_to_metadata(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=1)

    assert node.create() == 2

    path = provenance_path_in(node._base_dir)
    assert path.parent == metadata_path_in(node._base_dir).parent
    provenance = Provenance.model_validate_json(path.read_text())
    assert provenance.git is not None
    assert provenance.git.commit == _git(git_repo, "rev-parse", "HEAD")
    assert provenance.snapshot_id is None
    assert provenance.submitted.hostname == provenance.executed.hostname
    assert provenance.executed.worker_backend == "local"
    assert provenance.executed.pid == os.getpid()
    assert node.provenance() == provenance
    assert provenance.snapshot_path is None


def test_snapshot_true_builds_tarball_referenced_by_snapshot_id(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)

    furu.create(_Node(value=2), snapshot=True)

    provenance = _Node(value=2).provenance()
    assert provenance.snapshot_id is not None
    assert provenance.snapshot_path is not None
    assert provenance.snapshot_path.is_file()
    assert provenance.snapshot_path.parent.name == provenance.snapshot_id


def test_cache_hit_performs_no_capture_and_no_writes(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=3)
    node.create()
    original = provenance_path_in(node._base_dir).read_text()

    def fail_capture(**kwargs: object) -> object:
        raise AssertionError("cache hits must not capture provenance")

    monkeypatch.setattr(
        "furu.execution.load_or_create.capture_submit_provenance", fail_capture
    )

    assert node.create() == 4
    assert furu.create(_Node(value=3), snapshot=True) == 4
    assert provenance_path_in(node._base_dir).read_text() == original


def test_result_without_provenance_loads_but_provenance_raises(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=4)
    node.create()
    computed = len(_created)
    provenance_path_in(node._base_dir).unlink()

    assert node.create() == 5
    assert len(_created) == computed
    with pytest.raises(furu.Missing, match="provenance.json is missing"):
        node.provenance()


def test_outside_git_repo_records_null_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    node = _Node(value=5)
    node.create()

    provenance = node.provenance()
    assert provenance.git is None
    assert provenance.snapshot_id is None
    assert json.loads(provenance_path_in(node._base_dir).read_text())["git"] is None


def test_snapshot_outside_git_repo_fails_before_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    computed = len(_created)

    with pytest.raises(RuntimeError, match="not inside a git worktree"):
        furu.create(_Node(value=6), snapshot=True)

    assert len(_created) == computed


def test_snapshot_default_config_applies_to_plain_create(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    with override_config(_with_snapshot_default(True)):
        _Node(value=7).create()

        provenance = _Node(value=7).provenance()
        assert provenance.snapshot_id is not None
        assert provenance.snapshot_path is not None
        assert provenance.snapshot_path.is_file()


def test_explicit_snapshot_false_overrides_config_default(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    with override_config(_with_snapshot_default(True)):
        furu.create(_Node(value=8), snapshot=False)

        provenance = _Node(value=8).provenance()
        assert provenance.snapshot_id is None


def test_provenance_raises_missing_for_missing_result() -> None:
    with pytest.raises(furu.Missing, match="could not find a result"):
        _Node(value=9).provenance()
