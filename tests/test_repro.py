import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import furu
from furu import Spec, provenance
from furu._cli import main
from furu.config import _Config, get_config
from furu.provenance import EnvironmentIdentity, find_snapshot_marker
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


def _with_snapshot(enabled: bool) -> _Config:
    data = get_config().model_dump()
    data["provenance"]["snapshot"] = enabled
    return _Config.model_validate(data)


class _Node(Spec[int]):
    value: int = 0

    def create(self) -> int:
        return self.value + 1


def test_extract_materializes_the_snapshot_behind_a_result(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(git_repo)
    with override_config(_with_snapshot(True)):
        furu.create(_Node(value=1))

    dest = provenance.extract(_Node(value=1), tmp_path / "code", verify=True)

    assert dest == tmp_path / "code"
    assert (dest / "tracked.txt").read_text() == "content\n"


def test_extract_without_snapshot_explains_the_gap(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=2)
    node.create()  # the pytest harness disables snapshotting by default

    with pytest.raises(RuntimeError, match="without a snapshot"):
        provenance.extract(node, tmp_path / "code")


def test_repro_cli_extracts_and_summarizes(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=3)
    with override_config(_with_snapshot(True)):
        furu.create(node)
    dest = tmp_path / "repro"

    exit_code = main(
        ["repro", str(node._base_dir), "--dest", str(dest), "--verify", "--no-sync"]
    )

    assert exit_code == 0
    assert (dest / "tracked.txt").read_text() == "content\n"
    # No git repo is created: the marker carries the recorded identity.
    assert not (dest / ".git").exists()
    marker = find_snapshot_marker(dest)
    assert marker is not None
    assert marker.git.commit == _git(git_repo, "rev-parse", "HEAD")
    out = capsys.readouterr().out
    snapshot_id = node.provenance().snapshot_id
    assert snapshot_id is not None
    assert snapshot_id in out
    assert "_Node(value=3)" in out
    assert _git(git_repo, "rev-parse", "HEAD")[:7] in out
    assert "extracted snapshot" in out
    assert f"cd {dest}" in out


def test_repro_cli_accepts_paths_below_the_result_directory(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=4)
    with override_config(_with_snapshot(True)):
        furu.create(node)
    dest = tmp_path / "repro"

    exit_code = main(
        [
            "repro",
            str(node._base_dir / "metadata.json"),
            "--dest",
            str(dest),
            "--no-sync",
        ]
    )

    assert exit_code == 0
    assert (dest / "tracked.txt").read_text() == "content\n"


def test_repro_cli_without_snapshot_prints_instructions(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(git_repo)
    node = _Node(value=5)
    node.create()

    exit_code = main(["repro", str(node._base_dir), "--no-sync"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "without a snapshot" in out
    assert _git(git_repo, "rev-parse", "HEAD") in out


def test_repro_cli_reports_missing_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["repro", str(tmp_path / "nowhere"), "--no-sync"])

    assert exit_code == 1
    assert "no provenance.json" in capsys.readouterr().err
