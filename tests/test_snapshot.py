import subprocess
from pathlib import Path

import pytest

from furu.config import _Config, get_config
from furu.provenance import GitIdentity
from furu.snapshot import extract_snapshot, require_pushed
from furu.testing import override_config


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
    sub = repo / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested\n")
    _commit_and_push(repo, "init")
    return repo


def _commit_and_push(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    _git(repo, "push", "-q", "origin", "HEAD")
    return _git(repo, "rev-parse", "HEAD")


def _require_pushed(path: Path) -> None:
    require_pushed(GitIdentity.capture(path))


def test_clean_pushed_checkout_passes(git_repo: Path) -> None:
    _require_pushed(git_repo)
    _require_pushed(git_repo / "sub")


def test_modified_file_is_refused_with_the_fix(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")

    with pytest.raises(
        RuntimeError,
        match=r"(?s)uncommitted changes.*M tracked\.txt.*git push -u origin main",
    ):
        _require_pushed(git_repo)


def test_untracked_file_is_refused(git_repo: Path) -> None:
    (git_repo / "augment.py").write_text("new\n")

    with pytest.raises(RuntimeError, match=r"\?\? augment\.py"):
        _require_pushed(git_repo)


def test_ignored_files_do_not_count_as_changes(git_repo: Path) -> None:
    (git_repo / ".gitignore").write_text("ignored.txt\n")
    _commit_and_push(git_repo, "ignore")
    (git_repo / "ignored.txt").write_text("secret\n")

    _require_pushed(git_repo)


def test_furu_data_directories_do_not_count_as_changes(git_repo: Path) -> None:
    data = get_config().model_dump()
    data["directories"] = {
        name: git_repo / "furu-data" / name for name in data["directories"]
    }
    result = git_repo / "furu-data" / "objects" / "result.txt"
    result.parent.mkdir(parents=True)
    result.write_text("x\n")

    with override_config(_Config.model_validate(data)):
        _require_pushed(git_repo)


def test_long_change_lists_are_truncated(git_repo: Path) -> None:
    for i in range(12):
        (git_repo / f"new{i:02}.txt").write_text("x\n")

    with pytest.raises(RuntimeError, match=r"\.\.\. and 2 more") as excinfo:
        _require_pushed(git_repo)
    assert "new10.txt" not in str(excinfo.value)


def test_committed_but_unpushed_head_is_refused_until_pushed(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("v2\n")
    _git(git_repo, "commit", "-qam", "v2")

    with pytest.raises(
        RuntimeError, match=r"(?s)not on any branch of origin.*git push -u origin main"
    ):
        _require_pushed(git_repo)

    _git(git_repo, "push", "-q", "origin", "main")
    _require_pushed(git_repo)


def test_head_pushed_to_any_origin_branch_counts(git_repo: Path) -> None:
    _git(git_repo, "switch", "-qc", "exp/lr-sweep")
    (git_repo / "tracked.txt").write_text("lr=3e-4\n")
    _git(git_repo, "commit", "-qam", "lr")
    _git(git_repo, "push", "-q", "origin", "exp/lr-sweep")
    _git(git_repo, "switch", "-q", "--detach")

    _require_pushed(git_repo)


def test_detached_unpushed_head_suggests_pushing_to_a_branch(git_repo: Path) -> None:
    _git(git_repo, "switch", "-q", "--detach")
    _git(git_repo, "commit", "-q", "--allow-empty", "-m", "detached")

    with pytest.raises(RuntimeError, match=r"git push origin HEAD:refs/heads/"):
        _require_pushed(git_repo)


def test_repo_without_origin_is_refused(git_repo: Path) -> None:
    _git(git_repo, "remote", "remove", "origin")

    with pytest.raises(
        RuntimeError, match=r"(?s)no remote named origin.*git remote add origin"
    ):
        _require_pushed(git_repo)


def test_linked_worktree_is_checked_on_its_own(git_repo: Path, tmp_path: Path) -> None:
    linked = tmp_path / "linked"
    _git(git_repo, "worktree", "add", "-q", str(linked))
    _require_pushed(linked)

    (linked / "tracked.txt").write_text("edited in worktree\n")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        _require_pushed(linked)
    _require_pushed(git_repo)


def test_extract_snapshot_materializes_the_commit(git_repo: Path) -> None:
    (git_repo / "run.sh").write_text("#!/bin/sh\n")
    (git_repo / "run.sh").chmod(0o744)
    (git_repo / "link.txt").symlink_to("tracked.txt")
    commit = _commit_and_push(git_repo, "more")
    (git_repo / "tracked.txt").write_text("edited after the commit\n")

    code_dir = extract_snapshot(commit, repo=git_repo)

    assert code_dir == get_config().run_directories.snapshots / commit
    assert (code_dir / "tracked.txt").read_text() == "content\n"
    assert (code_dir / "sub" / "nested.txt").read_text() == "nested\n"
    assert (code_dir / "link.txt").readlink() == Path("tracked.txt")
    assert (code_dir / "run.sh").stat().st_mode & 0o111
    assert not (code_dir / "tracked.txt").stat().st_mode & 0o111


def test_extract_snapshot_reuses_an_existing_extraction(git_repo: Path) -> None:
    commit = _git(git_repo, "rev-parse", "HEAD")
    code_dir = extract_snapshot(commit, repo=git_repo)
    (code_dir / "witness.txt").write_text("kept\n")

    assert extract_snapshot(commit, repo=git_repo / "does-not-exist") == code_dir
    assert (code_dir / "witness.txt").read_text() == "kept\n"


def test_concurrent_extractor_losing_the_rename_discards_its_work(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def race_loser_rename(self: Path, target: Path) -> Path:
        target.mkdir(parents=True)
        (target / "winner").touch()
        raise OSError("simulated concurrent snapshot rename")

    commit = _git(git_repo, "rev-parse", "HEAD")
    monkeypatch.setattr(Path, "rename", race_loser_rename)

    code_dir = extract_snapshot(commit, repo=git_repo)

    assert (code_dir / "winner").is_file()
    assert list(code_dir.parent.iterdir()) == [code_dir]


def test_extract_missing_snapshot_raises(git_repo: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        extract_snapshot("0" * 40, repo=git_repo)
