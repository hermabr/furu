"""Content-addressed worktree snapshots.

Git's index defines "the code": the manifest is every tracked file plus
untracked-but-not-ignored files, at worktree state. The snapshot id is a hash
of the (path -> blob hash) map, so identical worktree states dedup to a single
shared directory under the configured snapshots root.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

from pydantic import BaseModel, ByteSize, ConfigDict

from furu.config import get_config
from furu.provenance import _run_git
from furu.utils import nfs_safe_unique_name

_SNAPSHOT_ID_DIGEST_SIZE = 10
_LARGEST_OFFENDERS_SHOWN = 10
_GITLINK_MODE = "160000"


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


def snapshot_dir(snapshot_id: str) -> Path:
    return get_config().run_directories.snapshots / snapshot_id


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

    index_blobs = _manifest_blobs(repo_root)
    sizes = {path: os.lstat(repo_root / path).st_size for path in index_blobs}
    _guard_size(sizes)
    blobs = {path: blob for path, blob in index_blobs.items() if blob is not None}
    blobs |= _hash_worktree_files(
        repo_root, [path for path, blob in index_blobs.items() if blob is None]
    )
    snapshot_id = hashlib.blake2s(
        json.dumps(blobs, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=_SNAPSHOT_ID_DIGEST_SIZE,
    ).hexdigest()

    final_dir = snapshot_dir(snapshot_id)
    if final_dir.is_dir():
        return snapshot_id

    manifest = SnapshotManifest(
        commit=commit,
        entries=tuple(
            SnapshotEntry(path=path, hash=blob, size=sizes[path])
            for path, blob in sorted(blobs.items())
        ),
        total_bytes=sum(sizes.values()),
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, check=True
    ).stdout

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = nfs_safe_unique_name(final_dir, name="tmp")
    tmp_dir.mkdir()
    try:
        _write_tarball(tmp_dir / "snapshot.tar.gz", repo_root, sorted(blobs))
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


def _manifest_blobs(repo_root: Path) -> dict[str, str | None]:
    """Manifest paths mapped to index blob hashes.

    ``None`` marks paths whose worktree bytes differ from the index (modified,
    conflicted, or untracked) and therefore need hashing; clean tracked files
    reuse the index blob with zero file reads.
    """
    blobs: dict[str, str | None] = {}
    for line in _split_z(_run_git(["ls-files", "-s", "-z"], cwd=repo_root)):
        meta, _, path = line.partition("\t")
        mode, blob, stage = meta.split()
        if mode == _GITLINK_MODE:
            continue
        blobs[path] = blob if stage == "0" else None
    tokens = _split_z(_run_git(["diff-files", "--name-status", "-z"], cwd=repo_root))
    for status, path in zip(tokens[::2], tokens[1::2], strict=True):
        if status == "D":
            blobs.pop(path, None)
        else:
            blobs[path] = None
    for path in _split_z(
        _run_git(["ls-files", "-o", "--exclude-standard", "-z"], cwd=repo_root)
    ):
        blobs[path] = None
    return blobs


def _hash_worktree_files(repo_root: Path, paths: list[str]) -> dict[str, str]:
    """Hash worktree bytes with ``git hash-object`` semantics.

    Matching git's blob addressing means a file hashes identically whether it
    is dirty today or committed tomorrow, so the snapshot id is stable.
    """
    hashes: dict[str, str] = {}
    regular: list[str] = []
    for path in paths:
        if (repo_root / path).is_symlink():
            hashes[path] = _run_git(
                ["hash-object", "--stdin"],
                cwd=repo_root,
                input=os.readlink(repo_root / path),
            )
        else:
            regular.append(path)
    if regular:
        output = _run_git(
            ["hash-object", "--stdin-paths"],
            cwd=repo_root,
            input="".join(f"{path}\n" for path in regular),
        )
        hashes.update(zip(regular, output.splitlines(), strict=True))
    return hashes


def _guard_size(sizes: dict[str, int]) -> None:
    total = sum(sizes.values())
    limit = get_config().provenance.max_snapshot_bytes
    if total <= limit:
        return
    largest = sorted(sizes.items(), key=lambda item: item[1], reverse=True)
    offenders = "\n".join(
        f"  {_human_size(size):>10}  {path}"
        for path, size in largest[:_LARGEST_OFFENDERS_SHOWN]
    )
    raise RuntimeError(
        f"worktree snapshot would be {_human_size(total)} "
        f"(limit: {_human_size(limit)}). Largest files in the manifest:\n"
        f"{offenders}\n"
        "These files are tracked or not ignored. Either add them to .gitignore,\n"
        "or raise [tool.furu.provenance] max_snapshot_bytes if this is intentional."
    )


def _human_size(size: int) -> str:
    return ByteSize(size).human_readable(separator=" ")


def _write_tarball(path: Path, repo_root: Path, paths: list[str]) -> None:
    """Deterministic archive: sorted paths, zeroed times and owners,
    normalized modes — identical manifests produce byte-identical bytes."""
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for rel_path in paths:
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
                        info.mode = 0o755 if file_stat.st_mode & stat.S_IXUSR else 0o644
                        with open(full_path, "rb") as file:
                            tar.addfile(info, file)
        raw.flush()
        os.fsync(raw.fileno())


def _write_file(path: Path, data: bytes) -> None:
    with open(path, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _split_z(output: str) -> list[str]:
    return [token for token in output.split("\0") if token]
