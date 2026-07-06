import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from furu.config import _Config, get_config
from furu.provenance import GitIdentity, find_snapshot_marker
from furu.snapshot import SnapshotManifest, create_snapshot, extract_snapshot
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


def _snapshot_dir(snapshot_id: str) -> Path:
    return _snapshots_root() / snapshot_id


def _tarball(snapshot_id: str) -> Path:
    return _snapshot_dir(snapshot_id) / "snapshot.tar.gz"


def _extract(snapshot_id: str, dest: Path) -> Path:
    with tarfile.open(_tarball(snapshot_id)) as tar:
        tar.extractall(dest, filter="tar")
    return dest


def _snapshots_root() -> Path:
    return get_config().run_directories.snapshots


def _with_max_snapshot_bytes(limit: int) -> _Config:
    data = get_config().model_dump()
    data["provenance"]["max_snapshot_bytes"] = limit
    return _Config.model_validate(data)


def test_same_worktree_state_produces_same_id_and_identical_bytes(
    git_repo: Path,
) -> None:
    (git_repo / "tracked.txt").write_text("dirty\n")
    (git_repo / "untracked.txt").write_text("new\n")
    first_id = create_snapshot(git_repo)
    first_bytes = _tarball(first_id).read_bytes()
    shutil.rmtree(_snapshots_root())

    second_id = create_snapshot(git_repo)

    assert second_id == first_id
    assert _tarball(second_id).read_bytes() == first_bytes


def test_snapshot_accepts_paths_below_the_repo_root(git_repo: Path) -> None:
    assert create_snapshot(git_repo / "sub") == create_snapshot(git_repo)


def test_existing_snapshot_is_reused_without_rebuilding(
    git_repo: Path,
) -> None:
    snapshot_id = create_snapshot(git_repo)
    _tarball(snapshot_id).unlink()

    assert create_snapshot(git_repo) == snapshot_id
    assert not _tarball(snapshot_id).exists()


def test_dirty_and_untracked_files_land_with_worktree_bytes(
    git_repo: Path, tmp_path: Path
) -> None:
    (git_repo / "tracked.txt").write_text("worktree version\n")
    (git_repo / "untracked.txt").write_text("brand new\n")

    extracted = _extract(create_snapshot(git_repo), tmp_path / "out")

    assert (extracted / "tracked.txt").read_text() == "worktree version\n"
    assert (extracted / "untracked.txt").read_text() == "brand new\n"
    assert (extracted / "sub" / "nested.txt").read_text() == "nested\n"


def test_ignored_files_never_appear(git_repo: Path) -> None:
    (git_repo / ".gitignore").write_text("ignored.txt\n")
    (git_repo / "ignored.txt").write_text("secret\n")
    _git(git_repo, "add", ".gitignore")
    _git(git_repo, "commit", "-qm", "ignore")

    with tarfile.open(_tarball(create_snapshot(git_repo))) as tar:
        assert "ignored.txt" not in tar.getnames()


def test_file_deleted_from_worktree_is_excluded(git_repo: Path) -> None:
    (git_repo / "tracked.txt").unlink()

    with tarfile.open(_tarball(create_snapshot(git_repo))) as tar:
        assert "tracked.txt" not in tar.getnames()


def test_untracked_content_addresses_like_committed_content(git_repo: Path) -> None:
    (git_repo / "new.txt").write_text("stable\n")
    untracked_id = create_snapshot(git_repo)
    shutil.rmtree(_snapshots_root())

    _git(git_repo, "add", "new.txt")
    _git(git_repo, "commit", "-qm", "add new.txt")

    assert create_snapshot(git_repo) == untracked_id


def test_symlinks_and_exec_bits_are_preserved_and_normalized(
    git_repo: Path,
) -> None:
    (git_repo / "run.sh").write_text("#!/bin/sh\n")
    (git_repo / "run.sh").chmod(0o744)
    (git_repo / "link.txt").symlink_to("tracked.txt")

    with tarfile.open(_tarball(create_snapshot(git_repo))) as tar:
        link = tar.getmember("link.txt")
        assert link.issym()
        assert link.linkname == "tracked.txt"
        assert tar.getmember("run.sh").mode == 0o755
        assert tar.getmember("tracked.txt").mode == 0o644
        for member in tar.getmembers():
            assert member.mtime == 0
            assert member.uid == member.gid == 0
            assert member.uname == member.gname == ""


def test_manifest_records_commit_entries_and_totals(git_repo: Path) -> None:
    (git_repo / "untracked.txt").write_text("new\n")
    snapshot_id = create_snapshot(git_repo)

    manifest = SnapshotManifest.model_validate_json(
        (_snapshot_dir(snapshot_id) / "manifest.json").read_text()
    )

    assert manifest.git.commit == _git(git_repo, "rev-parse", "HEAD")
    assert manifest.git.dirty  # untracked.txt
    assert [entry.path for entry in manifest.entries] == [
        "sub/nested.txt",
        "tracked.txt",
        "untracked.txt",
    ]
    tracked = next(e for e in manifest.entries if e.path == "tracked.txt")
    assert tracked.size == len("content\n")
    assert tracked.hash == _git(git_repo, "rev-parse", "HEAD:tracked.txt")
    assert manifest.total_bytes == sum(entry.size for entry in manifest.entries)


def test_oversize_worktree_fails_before_tarring(git_repo: Path) -> None:
    (git_repo / "big.bin").write_bytes(b"x" * 4096)

    with override_config(_with_max_snapshot_bytes(1024)):
        with pytest.raises(RuntimeError, match=r"(?s)4\.0 KiB  big\.bin.*gitignore"):
            create_snapshot(git_repo)

    assert not _snapshots_root().exists()


def test_concurrent_writer_losing_the_rename_discards_its_work(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def race_loser_rename(self: Path, target: Path) -> Path:
        target.mkdir(parents=True)
        (target / "winner").touch()
        raise OSError("simulated concurrent snapshot rename")

    monkeypatch.setattr(Path, "rename", race_loser_rename)
    snapshot_id = create_snapshot(git_repo)

    assert (_snapshot_dir(snapshot_id) / "winner").is_file()
    assert list(_snapshots_root().iterdir()) == [_snapshot_dir(snapshot_id)]


def test_outside_a_git_worktree_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not inside a git repository"):
        create_snapshot(tmp_path)


def test_extract_snapshot_round_trips_and_verifies(
    git_repo: Path, tmp_path: Path
) -> None:
    (git_repo / "tracked.txt").write_text("worktree version\n")
    (git_repo / "untracked.txt").write_text("brand new\n")
    (git_repo / "link.txt").symlink_to("tracked.txt")
    snapshot_id = create_snapshot(git_repo)

    dest = extract_snapshot(snapshot_id, tmp_path / "out", verify=True)

    assert (dest / "tracked.txt").read_text() == "worktree version\n"
    assert (dest / "untracked.txt").read_text() == "brand new\n"
    assert (dest / "sub" / "nested.txt").read_text() == "nested\n"
    assert (dest / "link.txt").is_symlink()
    assert (dest / "link.txt").readlink() == Path("tracked.txt")


def test_extracted_tree_is_stamped_and_re_snapshots_as_itself(
    git_repo: Path, tmp_path: Path
) -> None:
    snapshot_id = create_snapshot(git_repo)

    dest = extract_snapshot(snapshot_id, tmp_path / "out", verify=True)

    marker = find_snapshot_marker(dest / "sub")
    assert marker is not None
    assert marker.snapshot_id == snapshot_id
    assert marker.git.commit == _git(git_repo, "rev-parse", "HEAD")
    # The extracted tree is not a git repository, yet snapshotting and git
    # capture both resolve from the marker alone.
    assert create_snapshot(dest) == snapshot_id
    assert GitIdentity.capture(dest) == marker.git


def test_extract_verify_detects_content_drift(git_repo: Path, tmp_path: Path) -> None:
    snapshot_id = create_snapshot(git_repo)
    tarball = _tarball(snapshot_id)
    corrupted: dict[str, bytes] = {}
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            assert file is not None
            corrupted[member.name] = (
                b"corrupted\n" if member.name == "tracked.txt" else file.read()
            )
    with tarfile.open(tarball, "w:gz") as tar:
        for name, data in corrupted.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    with pytest.raises(RuntimeError, match=r"modified   tracked\.txt"):
        extract_snapshot(snapshot_id, tmp_path / "out", verify=True)


def test_extract_refuses_non_empty_destination(git_repo: Path, tmp_path: Path) -> None:
    snapshot_id = create_snapshot(git_repo)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "existing.txt").write_text("keep me\n")

    with pytest.raises(RuntimeError, match="not an empty directory"):
        extract_snapshot(snapshot_id, dest, verify=False)

    assert (dest / "existing.txt").read_text() == "keep me\n"


def test_extract_missing_snapshot_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="has no tarball"):
        extract_snapshot("0" * 20, tmp_path / "out")
