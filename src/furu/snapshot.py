from __future__ import annotations

import gzip
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

from pydantic import BaseModel, ByteSize, ConfigDict

from furu.config import get_config
from furu.provenance import _run_git
from furu.utils import _hash_dict_deterministically, nfs_safe_unique_name


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    hash: str
    size: int


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    commit: str
    entries: tuple[SnapshotEntry, ...]
    total_bytes: int


def create_snapshot(worktree: Path) -> str:
    """Snapshot the git worktree containing ``worktree``; return its id."""
    try:
        repo_root = Path(_run_git(["rev-parse", "--show-toplevel"], cwd=worktree))
        commit = _run_git(["rev-parse", "HEAD"], cwd=worktree)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot snapshot {worktree}: not inside a git worktree with a commit.\n"
            "Snapshots use git's index as the manifest, so snapshotting requires "
            "a git repository."
        ) from exc

    # Map manifest paths to index blob hashes. ``None`` marks paths whose
    # worktree bytes need hashing because they differ from the index.
    index_blobs: dict[str, str | None] = {}
    for line in _split_z(_run_git(["ls-files", "-s", "-z"], cwd=repo_root)):
        meta, _, path = line.partition("\t")
        mode, blob, stage = meta.split()
        if mode == "160000":  # gitlink (submodule): no worktree bytes to archive
            continue
        index_blobs[path] = blob if stage == "0" else None
    tokens = _split_z(_run_git(["diff-files", "--name-status", "-z"], cwd=repo_root))
    for status, path in zip(tokens[::2], tokens[1::2], strict=True):
        if status == "D":
            index_blobs.pop(path, None)
        else:
            index_blobs[path] = None
    for path in _split_z(
        _run_git(["ls-files", "-o", "--exclude-standard", "-z"], cwd=repo_root)
    ):
        index_blobs[path] = None

    sizes = {path: os.lstat(repo_root / path).st_size for path in index_blobs}
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

    # Use ``git hash-object`` semantics so dirty files hash identically after
    # they are committed later.
    hashed: dict[str, str] = {}
    regular_paths: list[str] = []
    for path, blob in index_blobs.items():
        if blob is not None:
            continue
        if (repo_root / path).is_symlink():
            hashed[path] = _run_git(
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
        hashed.update(zip(regular_paths, output.splitlines(), strict=True))
    blobs = {path: blob or hashed[path] for path, blob in index_blobs.items()}
    snapshot_id = _hash_dict_deterministically(blobs)

    final_dir = get_config().run_directories.snapshots / snapshot_id
    if final_dir.is_dir():
        return snapshot_id

    manifest = SnapshotManifest(
        commit=commit,
        entries=tuple(
            SnapshotEntry(path=path, hash=blob, size=sizes[path])
            for path, blob in sorted(blobs.items())
        ),
        total_bytes=total_bytes,
    )
    # Raw subprocess, not _run_git: a patch can contain non-UTF-8 bytes.
    diff = subprocess.run(
        ["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, check=True
    ).stdout

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
        _write_file(tmp_dir / "diff.patch", diff)
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


def _write_file(path: Path, data: bytes) -> None:
    with open(path, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _split_z(output: str) -> list[str]:
    return [token for token in output.split("\0") if token]
