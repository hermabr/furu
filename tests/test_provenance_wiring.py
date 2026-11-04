import os
import subprocess
from pathlib import Path

import pytest
from gitrepo import commit_and_push, git

import furu
from furu import Spec
from furu import provenance as furu_provenance
from furu.provenance import (
    GitIdentity,
    Provenance,
)
from furu.storage._layout import provenance_path_in, version_dir_in
from furu.worker.backends.local import LocalThreadWorkerBackend


@pytest.fixture
def require_pushed(
    git_repo: Path, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Undo the furu pytest plugin's opt-out for this test, in a real uv project:
    # submitting also checks the lock's path dependencies against the repo.
    monkeypatch.setattr(furu_provenance, "_require_pushed_code", True)
    (git_repo / "pyproject.toml").write_text(
        '[project]\nname = "repo"\nversion = "0"\nrequires-python = ">=3.12"\n'
    )
    subprocess.run(["uv", "lock", "-q", "--offline"], cwd=git_repo, check=True)
    commit_and_push(git_repo)


_created: list[str] = []


class _Node(Spec[int]):
    value: int = 0

    def create(self) -> int:
        _created.append(self.object_id)
        return self.value + 1


def test_pytest_harness_skips_the_pushed_check() -> None:
    # Tests run whatever the working tree holds; tests of the check opt back
    # in with the require_pushed fixture.
    assert furu_provenance._require_pushed_code is False


def test_create_writes_provenance_into_the_version_directory(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=1)

    assert node.create() == 2

    path = provenance_path_in(version_dir_in(node._base_dir))
    provenance = Provenance.model_validate_json(path.read_text())
    assert provenance.git.commit == git(git_repo, "rev-parse", "HEAD")
    assert provenance.submitted.hostname == provenance.executed.hostname
    assert provenance.executed.worker_backend == "local"
    assert provenance.executed.pid == os.getpid()
    assert node.provenance() == provenance


@pytest.mark.usefixtures("require_pushed")
def test_uncommitted_changes_fail_before_compute(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    (git_repo / "tracked.txt").write_text("edited\n")
    computed = len(_created)

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        furu.create(_Node(value=2))

    assert len(_created) == computed


def test_cache_hit_performs_no_capture_and_no_writes(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=3)
    node.create()
    original = provenance_path_in(version_dir_in(node._base_dir)).read_text()

    def fail_capture(**kwargs: object) -> object:
        raise AssertionError("cache hits must not capture provenance")

    monkeypatch.setattr(
        "furu.execution.load_or_create.capture_submit_provenance", fail_capture
    )

    assert node.create() == 4
    # Loading needs no pushed code, even from a dirty tree.
    monkeypatch.setattr(furu_provenance, "_require_pushed_code", True)
    (git_repo / "tracked.txt").write_text("edited\n")
    assert furu.create(_Node(value=3)) == 4
    assert provenance_path_in(version_dir_in(node._base_dir)).read_text() == original


def test_result_without_provenance_loads_but_provenance_raises(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=4)
    node.create()
    computed = len(_created)
    provenance_path_in(version_dir_in(node._base_dir)).unlink()

    assert node.create() == 5
    assert len(_created) == computed
    with pytest.raises(furu.Missing, match="provenance.json is missing"):
        node.provenance()


def test_outside_git_repo_fails_before_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    computed = len(_created)

    with pytest.raises(RuntimeError, match="not inside a git repository"):
        furu.create(_Node(value=6))

    assert len(_created) == computed


@pytest.mark.usefixtures("require_pushed")
def test_pushed_commit_is_recorded(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    _Node(value=7).create()

    provenance = _Node(value=7).provenance()
    assert provenance.git.commit == git(git_repo, "rev-parse", "origin/main")
    assert provenance.git.remote == str(git_repo.parent / "repo-origin.git")


def test_provenance_raises_missing_for_missing_result() -> None:
    with pytest.raises(furu.Missing, match="could not find a result"):
        _Node(value=9).provenance()


@pytest.mark.usefixtures("require_pushed")
def test_executor_sweep_captures_once_and_stamps_every_job(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    git_captures: list[Path] = []
    original_capture = GitIdentity.capture.__func__  # type: ignore[attr-defined]

    def counting_capture(cls: type[GitIdentity], cwd: Path) -> GitIdentity:
        git_captures.append(cwd)
        return original_capture(cls, cwd)

    monkeypatch.setattr(GitIdentity, "capture", classmethod(counting_capture))
    nodes = [_Node(value=value) for value in (20, 21, 22)]

    assert furu.create(nodes, on=(LocalThreadWorkerBackend(max_workers=2),)) == [
        21,
        22,
        23,
    ]

    assert len(git_captures) == 1
    provenances = [node.provenance() for node in nodes]
    assert {provenance.git.commit for provenance in provenances} == {
        git(git_repo, "rev-parse", "HEAD")
    }
    assert all(
        provenance.submitted == provenances[0].submitted for provenance in provenances
    )
    assert all(
        provenance.executed.worker_backend == "local-thread"
        for provenance in provenances
    )
