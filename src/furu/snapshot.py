from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ByteSize

from furu.config import get_config
from furu.logging import get_logger
from furu.provenance import _run_git
from furu.utils import nfs_safe_unique_name

logger = get_logger()

# Fixed identity for synthetic snapshot commits: the sha then depends only on
# (tree, parent), so the same dirty tree snapshots to the same sha everywhere.
_COMMIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "furu",
    "GIT_AUTHOR_EMAIL": "furu@localhost",
    "GIT_COMMITTER_NAME": "furu",
    "GIT_COMMITTER_EMAIL": "furu@localhost",
}


def snapshot_ref(snapshot_id: str) -> str:
    return f"refs/furu/{snapshot_id}"


@contextmanager
def publish_dir_atomically(final_dir: Path) -> Iterator[Path]:
    """Yield a fresh temp dir to build in; on clean exit rename it to ``final_dir``.

    A concurrent builder of the same content may win the rename first, in which case
    the temp dir is discarded — the published directory is identical either way.
    """
    tmp_dir = nfs_safe_unique_name(final_dir, name="tmp")
    tmp_dir.mkdir(parents=True)
    try:
        yield tmp_dir
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    try:
        tmp_dir.rename(final_dir)
    except OSError:
        shutil.rmtree(tmp_dir)
        if not final_dir.is_dir():
            raise


def create_snapshot(worktree: Path) -> str:
    """Commit the working tree of the repo containing ``worktree``; return the sha.

    The commit is built in a scratch index, so HEAD, the index and the working
    files never move. Its parent is HEAD; a clean tree snapshots to HEAD itself.
    The sha is pinned under ``refs/furu/<sha>`` so it survives ``git gc``.
    """
    try:
        repo_root = Path(_run_git(["rev-parse", "--show-toplevel"], cwd=worktree))
        head = _run_git(["rev-parse", "HEAD"], cwd=worktree)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot snapshot {worktree}: not inside a git worktree with a commit.\n"
            "Snapshots are git commits, so snapshotting requires a git repository."
        ) from exc

    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(scratch, "index")}
        _run_git(["read-tree", "HEAD"], cwd=repo_root, env=env)
        _run_git(["add", "-A"], cwd=repo_root, env=env)
        tree = _run_git(["write-tree"], cwd=repo_root, env=env)

    _check_size(repo_root, tree)

    if tree == _run_git(["rev-parse", "HEAD^{tree}"], cwd=repo_root):
        sha = head
    else:
        date = _run_git(
            ["show", "-s", "--format=%cd", "--date=raw", head], cwd=repo_root
        )
        sha = _run_git(
            ["commit-tree", tree, "-p", head, "-m", "furu: working tree at submit"],
            cwd=repo_root,
            env={
                **os.environ,
                **_COMMIT_IDENTITY,
                "GIT_AUTHOR_DATE": date,
                "GIT_COMMITTER_DATE": date,
            },
        )

    # The local ref keeps the commit out of gc and memoizes "already pushed".
    ref = snapshot_ref(sha)
    if not _run_git(["for-each-ref", ref], cwd=repo_root):
        push = get_config().provenance.push
        if push:
            _push(repo_root, sha)
        _run_git(["update-ref", ref, sha], cwd=repo_root)
        untracked = _split_z(
            _run_git(["ls-files", "-o", "--exclude-standard", "-z"], cwd=repo_root)
        )
        logger.info(
            "snapshot %s %s %s%s",
            sha[:12],
            "pushed to origin as" if push else "pinned locally at",
            ref,
            f"; untracked files included: {' '.join(untracked)}" if untracked else "",
        )
    return sha


def _push(repo_root: Path, sha: str) -> None:
    result = subprocess.run(
        [
            "git",
            "push",
            "--quiet",
            "--no-verify",
            "origin",
            f"{sha}:{snapshot_ref(sha)}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not push snapshot {sha} to origin:\n"
            f"{textwrap.indent(result.stderr.strip(), '  ')}\n"
            "furu pushes every snapshot so results point at commits anyone with repo "
            "access can fetch.\n"
            "Fix the `origin` remote (or the connection), or keep snapshots local with\n"
            "  FURU_PROVENANCE__PUSH=false   # or push = false under [tool.furu.provenance]"
        )


def extract_snapshot(snapshot_id: str, *, repo: Path) -> Path:
    """Materialize snapshot ``snapshot_id`` from ``repo``'s objects; return the tree.

    Extraction is content-addressed like the snapshot itself: the tree lands in
    ``<snapshots>/<snapshot_id>/code`` once and is reused by every later caller.
    """
    code_dir = get_config().run_directories.snapshots / snapshot_id / "code"
    if code_dir.is_dir():
        return code_dir
    archive = subprocess.run(
        ["git", "archive", "--format=tar", snapshot_id],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    with (
        publish_dir_atomically(code_dir) as tmp_dir,
        tarfile.open(fileobj=io.BytesIO(archive)) as tar,
    ):
        tar.extractall(tmp_dir, filter="tar")
    return code_dir


def _check_size(repo_root: Path, tree: str) -> None:
    sizes: dict[str, int] = {}
    for line in _split_z(_run_git(["ls-tree", "-r", "-l", "-z", tree], cwd=repo_root)):
        meta, _, path = line.partition("\t")
        size = meta.split()[3]
        if size != "-":  # gitlinks (submodules) have no bytes of their own
            sizes[path] = int(size)
    total_bytes = sum(sizes.values())
    limit = get_config().provenance.max_snapshot_bytes
    if total_bytes <= limit:
        return
    largest = sorted(sizes.items(), key=lambda item: item[1], reverse=True)
    offenders = "\n".join(
        f"  {ByteSize(size).human_readable(separator=' '):>10}  {path}"
        for path, size in largest[:10]
    )
    total = ByteSize(total_bytes).human_readable(separator=" ")
    limit_text = ByteSize(limit).human_readable(separator=" ")
    raise RuntimeError(
        f"worktree snapshot would be {total} "
        f"(limit: {limit_text}). Largest files in the snapshot:\n"
        f"{offenders}\n"
        "These files are tracked or not ignored. Either add them to .gitignore,\n"
        "or raise [tool.furu.provenance] max_snapshot_bytes if this is intentional."
    )


def _split_z(output: str) -> list[str]:
    return [token for token in output.split("\0") if token]
