import subprocess
from pathlib import Path

import pytest

from furu.config import _Config, get_config
from furu.snapshot import create_snapshot, extract_snapshot
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
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "tracked.txt").write_text("content\n")
    sub = repo / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _snapshots_root() -> Path:
    return get_config().run_directories.snapshots


def _with_max_snapshot_bytes(limit: int) -> _Config:
    data = get_config().model_dump()
    data["provenance"]["max_snapshot_bytes"] = limit
    return _Config.model_validate(data)


def _checkout_state(repo: Path) -> tuple[str, str, str]:
    """HEAD, index and working-tree status, everything a snapshot must not move."""
    return (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "ls-files", "-s"),
        _git(repo, "status", "--porcelain"),
    )


def _tree_paths(repo: Path, sha: str) -> list[str]:
    return _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()


def test_clean_tree_snapshots_to_head(git_repo: Path) -> None:
    head = _git(git_repo, "rev-parse", "HEAD")

    assert create_snapshot(git_repo) == head
    assert _git(git_repo, "rev-parse", f"refs/furu/{head}") == head


def test_dirty_tree_becomes_a_commit_on_top_of_head(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")
    (git_repo / "untracked.txt").write_text("new\n")
    head = _git(git_repo, "rev-parse", "HEAD")

    sha = create_snapshot(git_repo)

    assert sha != head
    assert _git(git_repo, "rev-parse", f"{sha}~1") == head
    assert _git(git_repo, "rev-parse", f"refs/furu/{sha}") == sha
    assert _git(git_repo, "diff", "--name-status", head, sha).splitlines() == [
        "M\ttracked.txt",
        "A\tuntracked.txt",
    ]
    assert _git(git_repo, "show", "-s", "--format=%an <%ae>|%s", sha) == (
        "furu <furu@localhost>|furu: working tree at submit"
    )


def test_snapshot_never_moves_head_index_or_worktree(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")
    (git_repo / "untracked.txt").write_text("new\n")
    _git(git_repo, "add", "sub/nested.txt")
    before = _checkout_state(git_repo)

    create_snapshot(git_repo)

    assert _checkout_state(git_repo) == before
    assert (git_repo / "tracked.txt").read_text() == "dirty\n"


def test_same_dirty_tree_gives_the_same_sha_across_runs_and_identities(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")
    first = create_snapshot(git_repo)
    _git(git_repo, "update-ref", "-d", f"refs/furu/{first}")

    monkeypatch.setenv("GIT_AUTHOR_NAME", "someone else")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2000-01-01T00:00:00Z")
    assert create_snapshot(git_repo) == first


def test_snapshot_accepts_paths_below_the_repo_root(git_repo: Path) -> None:
    assert create_snapshot(git_repo / "sub") == create_snapshot(git_repo)


def test_snapshot_works_from_a_linked_worktree(git_repo: Path, tmp_path: Path) -> None:
    linked = tmp_path / "linked"
    _git(git_repo, "worktree", "add", "-q", str(linked))
    (linked / "tracked.txt").write_text("edited in worktree\n")

    sha = create_snapshot(linked)

    assert _git(git_repo, "rev-parse", f"refs/furu/{sha}") == sha
    assert _git(git_repo, "show", f"{sha}:tracked.txt") == "edited in worktree"


def test_ignored_files_never_appear(git_repo: Path) -> None:
    (git_repo / ".gitignore").write_text("ignored.txt\n")
    (git_repo / "ignored.txt").write_text("secret\n")
    _git(git_repo, "add", ".gitignore")
    _git(git_repo, "commit", "-qm", "ignore")

    assert "ignored.txt" not in _tree_paths(git_repo, create_snapshot(git_repo))


def test_file_deleted_from_worktree_is_excluded(git_repo: Path) -> None:
    (git_repo / "tracked.txt").unlink()

    assert "tracked.txt" not in _tree_paths(git_repo, create_snapshot(git_repo))


def test_worktree_wins_over_the_index(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("staged\n")
    _git(git_repo, "add", "tracked.txt")
    (git_repo / "tracked.txt").write_text("worktree\n")

    sha = create_snapshot(git_repo)

    assert _git(git_repo, "show", f"{sha}:tracked.txt") == "worktree"


def test_oversize_worktree_fails_before_committing(git_repo: Path) -> None:
    (git_repo / "big.bin").write_bytes(b"x" * 4096)

    with (
        override_config(_with_max_snapshot_bytes(1024)),
        pytest.raises(RuntimeError, match=r"(?s)4\.0 KiB  big\.bin.*gitignore"),
    ):
        create_snapshot(git_repo)

    assert _git(git_repo, "for-each-ref", "refs/furu/") == ""


def test_outside_a_git_worktree_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not inside a git worktree"):
        create_snapshot(tmp_path)


def test_extract_snapshot_materializes_worktree_bytes(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("worktree version\n")
    (git_repo / "untracked.txt").write_text("brand new\n")
    (git_repo / "run.sh").write_text("#!/bin/sh\n")
    (git_repo / "run.sh").chmod(0o744)
    (git_repo / "link.txt").symlink_to("tracked.txt")
    sha = create_snapshot(git_repo)

    code_dir = extract_snapshot(sha, repo=git_repo)

    assert code_dir == _snapshots_root() / sha / "code"
    assert (code_dir / "tracked.txt").read_text() == "worktree version\n"
    assert (code_dir / "untracked.txt").read_text() == "brand new\n"
    assert (code_dir / "sub" / "nested.txt").read_text() == "nested\n"
    assert (code_dir / "link.txt").readlink() == Path("tracked.txt")
    assert (code_dir / "run.sh").stat().st_mode & 0o111
    assert not (code_dir / "tracked.txt").stat().st_mode & 0o111


def test_extract_snapshot_reuses_an_existing_extraction(git_repo: Path) -> None:
    sha = create_snapshot(git_repo)
    code_dir = extract_snapshot(sha, repo=git_repo)
    (code_dir / "witness.txt").write_text("kept\n")

    assert extract_snapshot(sha, repo=git_repo / "does-not-exist") == code_dir
    assert (code_dir / "witness.txt").read_text() == "kept\n"


def test_concurrent_extractor_losing_the_rename_discards_its_work(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def race_loser_rename(self: Path, target: Path) -> Path:
        target.mkdir(parents=True)
        (target / "winner").touch()
        raise OSError("simulated concurrent snapshot rename")

    sha = create_snapshot(git_repo)
    monkeypatch.setattr(Path, "rename", race_loser_rename)

    code_dir = extract_snapshot(sha, repo=git_repo)

    assert (code_dir / "winner").is_file()
    assert list(code_dir.parent.iterdir()) == [code_dir]


def test_extract_missing_snapshot_raises(git_repo: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        extract_snapshot("0" * 40, repo=git_repo)
