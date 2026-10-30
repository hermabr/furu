import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import furu
from furu import Spec
from furu import provenance as furu_provenance
from furu.metadata import ArtifactSpec
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    Provenance,
    capture_submit_provenance,
)
from furu.storage._layout import provenance_path_in
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.execute import ChildSlot
from furu.worker.protocol import Job, ProcessSettings


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
    """A committed repo whose ``main`` is pushed to a bare ``origin``."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "remote", "add", "origin", str(origin))
    (repo / "tracked.txt").write_text("content\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "push", "-q", "origin", "main")
    return repo


@pytest.fixture
def require_pushed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Undo the furu pytest plugin's opt-out for this test.
    monkeypatch.setattr(furu_provenance, "_require_pushed_code", True)


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

    path = provenance_path_in(node._base_dir / "v-fixed")
    provenance = Provenance.model_validate_json(path.read_text())
    assert provenance.git.commit == _git(git_repo, "rev-parse", "HEAD")
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
    original = provenance_path_in(node._base_dir / "v-fixed").read_text()

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
    assert provenance_path_in(node._base_dir / "v-fixed").read_text() == original


def test_result_without_provenance_loads_but_provenance_raises(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=4)
    node.create()
    computed = len(_created)
    provenance_path_in(node._base_dir / "v-fixed").unlink()

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
    assert provenance.git.commit == _git(git_repo, "rev-parse", "origin/main")
    assert provenance.git.remote == str(git_repo.parent / "origin.git")


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
        _git(git_repo, "rev-parse", "HEAD")
    }
    assert all(
        provenance.submitted == provenances[0].submitted for provenance in provenances
    )
    assert all(
        provenance.executed.worker_backend == "local-thread"
        for provenance in provenances
    )


def test_worker_fails_job_on_stale_uv_lock_hash(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=30)
    submit = capture_submit_provenance()
    stale = submit.model_copy(
        update={
            "environment": submit.environment.model_copy(
                update={"uv_lock_hash": "blake2s:stale"}
            )
        }
    )

    with pytest.raises(RuntimeError, match="worker uv.lock does not match") as excinfo:
        ChildSlot(worker="w0", backend="test", materialize_snapshot=False).run(
            Job(
                artifacts=[ArtifactSpec.from_furu(node)],
                run_logs=[git_repo / "run.log"],
                attempt=1,
                execution_log=git_repo / "execution.log",
                provenance=stale,
                process=ProcessSettings.from_metadata(node._metadata),
            ),
            cancelled=threading.Event(),
        )

    assert "blake2s:stale" in str(excinfo.value)
    assert EnvironmentIdentity.capture().uv_lock_hash in str(excinfo.value)
    assert node.status == "missing"
