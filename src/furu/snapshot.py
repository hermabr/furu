from __future__ import annotations

import gzip
import os
import shutil
import stat
import tarfile
from pathlib import Path

from pydantic import BaseModel, ByteSize, ConfigDict

from furu.config import get_config
from furu.provenance import (
    SNAPSHOT_MARKER_NAME,
    GitIdentity,
    SnapshotMarker,
    _run_git,
    find_snapshot_marker,
)
from furu.utils import _hash_dict_deterministically, nfs_safe_unique_name


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    hash: str
    size: int


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    git: GitIdentity
    entries: tuple[SnapshotEntry, ...]
    total_bytes: int


def create_snapshot(worktree: Path, *, git: GitIdentity | None = None) -> str:
    """Snapshot the git worktree containing ``worktree``; return its id.

    ``git`` lets callers that already captured the worktree's git identity
    pass it along instead of capturing it a second time.
    """
    if (marker := find_snapshot_marker(worktree)) is not None:
        # An extracted snapshot re-snapshots as itself: its identity was
        # established when the snapshot was created.
        return marker.snapshot_id
    if git is None:
        git = GitIdentity.capture(worktree)
    repo_root = Path(git.repo_root)

    # Split the manifest paths into clean files (path -> index blob hash) and
    # dirty files (paths whose worktree bytes must be hashed). The two are disjoint.
    clean_blobs: dict[str, str] = {}
    dirty_paths: set[str] = set()
    for line in _split_z(_run_git(["ls-files", "-s", "-z"], cwd=repo_root)):
        meta, _, path = line.partition("\t")
        mode, blob, stage = meta.split()
        if mode == "160000":  # gitlink (submodule): no worktree bytes to archive
            continue
        if stage == "0":
            clean_blobs[path] = blob
        else:  # unmerged (conflict): the index blob is ambiguous, hash the worktree
            dirty_paths.add(path)
    tokens = _split_z(_run_git(["diff-files", "--name-status", "-z"], cwd=repo_root))
    for status, path in zip(tokens[::2], tokens[1::2], strict=True):
        clean_blobs.pop(path, None)  # differs from the index, so no longer clean
        if status == "D":
            dirty_paths.discard(path)  # deleted from the worktree: nothing to archive
        else:
            dirty_paths.add(path)
    for path in _split_z(
        _run_git(["ls-files", "-o", "--exclude-standard", "-z"], cwd=repo_root)
    ):
        dirty_paths.add(path)

    sizes = {
        path: os.lstat(repo_root / path).st_size
        for path in (*clean_blobs, *dirty_paths)
    }
    total_bytes = sum(sizes.values())
    limit = get_config().provenance.max_snapshot_bytes
    if total_bytes > limit:
        largest = sorted(sizes.items(), key=lambda item: item[1], reverse=True)
        offenders = "\n".join(
            f"  {ByteSize(size).human_readable(separator=' '):>10}  {path}"
            for path, size in largest[:10]
        )
        total = ByteSize(total_bytes).human_readable(separator=" ")
        limit_text = ByteSize(limit).human_readable(separator=" ")
        raise RuntimeError(
            f"worktree snapshot would be {total} "
            f"(limit: {limit_text}). Largest files in the manifest:\n"
            f"{offenders}\n"
            "These files are tracked or not ignored. Either add them to .gitignore,\n"
            "or raise [tool.furu.provenance] max_snapshot_bytes if this is intentional."
        )

    # Hash dirty files with ``git hash-object`` semantics, so they hash identically
    # once committed later. Symlinks hash their target; regular files their bytes.
    blobs = dict(clean_blobs)
    regular_paths: list[str] = []
    for path in sorted(dirty_paths):
        if (repo_root / path).is_symlink():
            blobs[path] = _run_git(
                ["hash-object", "--stdin"],
                cwd=repo_root,
                input=os.readlink(repo_root / path),
            )
        else:
            regular_paths.append(path)
    if regular_paths:
        output = _run_git(
            ["hash-object", "--stdin-paths"],
            cwd=repo_root,
            input="".join(f"{path}\n" for path in regular_paths),
        )
        blobs.update(zip(regular_paths, output.splitlines(), strict=True))
    snapshot_id = _hash_dict_deterministically(blobs)

    final_dir = get_config().run_directories.snapshots / snapshot_id
    if final_dir.is_dir():
        return snapshot_id

    manifest = SnapshotManifest(
        git=git,
        entries=tuple(
            SnapshotEntry(path=path, hash=blob, size=sizes[path])
            for path, blob in sorted(blobs.items())
        ),
        total_bytes=total_bytes,
    )

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = nfs_safe_unique_name(final_dir, name="tmp")
    tmp_dir.mkdir()
    try:
        tarball_path = tmp_dir / "snapshot.tar.gz"
        with open(tarball_path, "wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    for rel_path in sorted(blobs):
                        full_path = repo_root / rel_path
                        file_stat = os.lstat(full_path)
                        info = tarfile.TarInfo(rel_path)
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        if stat.S_ISLNK(file_stat.st_mode):
                            info.type = tarfile.SYMTYPE
                            info.linkname = os.readlink(full_path)
                            info.mode = 0o777
                            tar.addfile(info)
                        else:
                            info.size = file_stat.st_size
                            info.mode = (
                                0o755 if file_stat.st_mode & stat.S_IXUSR else 0o644
                            )
                            with open(full_path, "rb") as file:
                                tar.addfile(info, file)
            raw.flush()
            os.fsync(raw.fileno())
        _write_file(
            tmp_dir / "manifest.json", manifest.model_dump_json(indent=2).encode()
        )
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    try:
        tmp_dir.rename(final_dir)
    except OSError:
        # A concurrent snapshotter of the same worktree state won the rename.
        shutil.rmtree(tmp_dir)
        if not final_dir.is_dir():
            raise
    return snapshot_id


def read_snapshot_manifest(snapshot_id: str) -> SnapshotManifest:
    manifest_path = (
        get_config().run_directories.snapshots / snapshot_id / "manifest.json"
    )
    try:
        return SnapshotManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise RuntimeError(
            f"snapshot {snapshot_id} has no readable manifest at {manifest_path}.\n"
            "Snapshots live in the furu-data snapshots directory of the project "
            "that created the result. Run from that project, or copy its "
            "snapshot directory here."
        ) from exc


def extract_snapshot(snapshot_id: str, dest: Path, *, verify: bool = False) -> Path:
    """Materialize snapshot ``snapshot_id`` into ``dest``; return ``dest``.

    The extracted tree is stamped with a snapshot marker recording the
    snapshot id and the git identity captured when the snapshot was created,
    so it is runnable without a git repository of its own.

    With ``verify=True``, re-hash the extracted tree with ``git hash-object``
    semantics and require the (path -> hash) map to re-derive ``snapshot_id``.
    """
    tarball = get_config().run_directories.snapshots / snapshot_id / "snapshot.tar.gz"
    if not tarball.is_file():
        raise RuntimeError(
            f"snapshot {snapshot_id} has no tarball at {tarball}.\n"
            "Snapshots live in the furu-data snapshots directory of the project "
            "that created the result. Run from that project, or copy its "
            "snapshot directory here."
        )
    manifest = read_snapshot_manifest(snapshot_id)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise RuntimeError(
            f"refusing to extract snapshot {snapshot_id} into {dest}: "
            "the destination already exists and is not an empty directory."
        )
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        tar.extractall(dest, filter="tar")
    if verify:
        _verify_extraction(snapshot_id, manifest, dest)
    marker = SnapshotMarker(snapshot_id=snapshot_id, git=manifest.git)
    _write_file(
        dest / SNAPSHOT_MARKER_NAME, marker.model_dump_json(indent=2).encode()
    )
    return dest


def _verify_extraction(
    snapshot_id: str, manifest: SnapshotManifest, dest: Path
) -> None:
    expected = {entry.path: entry.hash for entry in manifest.entries}
    if _hash_dict_deterministically(expected) != snapshot_id:
        raise RuntimeError(
            f"snapshot {snapshot_id} failed verification: manifest.json does not "
            "re-derive the snapshot id, so the snapshot directory was modified "
            "after it was written."
        )
    actual = _hash_extracted_tree(dest)
    if actual == expected:
        return
    problems = []
    for path in sorted(expected.keys() | actual.keys()):
        if path not in actual:
            problems.append(f"  missing    {path}")
        elif path not in expected:
            problems.append(f"  unexpected {path}")
        elif actual[path] != expected[path]:
            problems.append(f"  modified   {path}")
    raise RuntimeError(
        f"snapshot {snapshot_id} failed verification: the extracted tree does "
        f"not match the manifest:\n" + "\n".join(problems)
    )


def _hash_extracted_tree(root: Path) -> dict[str, str]:
    # Mirror of create_snapshot's dirty-file hashing: git hash-object semantics,
    # symlinks hash their target, regular files their bytes.
    blobs: dict[str, str] = {}
    regular_paths: list[str] = []
    for path in sorted(
        str(entry.relative_to(root))
        for entry in root.rglob("*")
        if entry.is_symlink() or entry.is_file()
    ):
        if (root / path).is_symlink():
            blobs[path] = _run_git(
                ["hash-object", "--stdin"], cwd=root, input=os.readlink(root / path)
            )
        else:
            regular_paths.append(path)
    if regular_paths:
        output = _run_git(
            ["hash-object", "--stdin-paths"],
            cwd=root,
            input="".join(f"{path}\n" for path in regular_paths),
        )
        blobs.update(zip(regular_paths, output.splitlines(), strict=True))
    return blobs


def _write_file(path: Path, data: bytes) -> None:
    with open(path, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _split_z(output: str) -> list[str]:
    return [token for token in output.split("\0") if token]
