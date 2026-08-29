from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import ByteSize

from furu.config import get_config
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitProvenance,
    _run_git,
)
from furu.utils import nfs_safe_unique_name


@contextmanager
def publish_dir_atomically(final_dir: Path) -> Generator[Path]:
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


def require_pushed(git: GitIdentity) -> None:
    """Refuse unless the checkout is exactly ``git.commit`` and origin has it.

    Every result then points at a commit that anyone with access to origin can
    fetch, and Slurm workers can run that commit as the code snapshot.
    """
    repo_root = Path(git.repo_root)
    # Keep furu's own directories out of `git status`, as uv does for .venv.
    for _, directory in get_config().run_directories:
        if not (ignore := directory / ".gitignore").exists():
            directory.mkdir(parents=True, exist_ok=True)
            ignore.write_text("*\n")
    status = _run_git(["status", "--porcelain"], cwd=repo_root)
    if status:
        changes = status.splitlines()
        shown = "\n".join(f"  {line}" for line in changes[:10])
        more = f"\n  ... and {len(changes) - 10} more" if len(changes) > 10 else ""
        raise RuntimeError(
            f"uncommitted changes in {repo_root}:\n{shown}{more}\n"
            "furu only runs committed, pushed code, so every result points at a "
            "commit others can fetch.\n"
            "Commit and push them (or add them to .gitignore):\n"
            f"  git add -A && git commit -m '...' && {_push_command(git)}"
        )
    _check_size(repo_root, git.commit)
    if git.remote is None:
        raise RuntimeError(
            f"{repo_root} has no remote named origin.\n"
            "furu only runs pushed code, so every result points at a commit "
            "others can fetch. Add a remote and push:\n"
            "  git remote add origin <url>\n"
            f"  {_push_command(git)}"
        )
    pushed_to = _run_git(
        ["for-each-ref", "--contains", git.commit, "refs/remotes/origin"],
        cwd=repo_root,
    )
    if not pushed_to:
        raise RuntimeError(
            f"HEAD ({git.commit[:12]}) is not on any branch of origin "
            f"({git.remote}).\n"
            "furu only runs pushed code, so every result points at a commit "
            "others can fetch. Push it:\n"
            f"  {_push_command(git)}"
        )


def extract_snapshot(commit: str, *, repo: Path) -> Path:
    """Materialize ``commit`` from ``repo``'s objects; return the tree.

    The tree lands in ``<snapshots>/<commit>/code`` once and is reused by every
    later caller.
    """
    code_dir = get_config().run_directories.snapshots / commit / "code"
    if code_dir.is_dir():
        return code_dir
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
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


@dataclass(frozen=True, slots=True)
class CodeLocation:
    python: Path
    project_root: Path
    cwd: Path

    @classmethod
    def here(cls) -> CodeLocation:
        return cls(
            python=Path(sys.executable),
            project_root=Path(EnvironmentIdentity.capture().project_root),
            cwd=Path.cwd(),
        )

    @classmethod
    def from_snapshot(cls, provenance: SubmitProvenance) -> CodeLocation:
        repo_root = Path(provenance.git.repo_root)
        code_dir = extract_snapshot(provenance.git.commit, repo=repo_root).resolve()
        project_root = code_dir / Path(provenance.environment.project_root).relative_to(
            repo_root
        )
        return cls(
            python=project_root / ".venv" / "bin" / "python",
            project_root=project_root,
            cwd=code_dir / Path(provenance.submitted.cwd).relative_to(repo_root),
        )


def _push_command(git: GitIdentity) -> str:
    if git.branch is None:  # detached HEAD
        return "git push origin HEAD:refs/heads/<branch>"
    return f"git push -u origin {git.branch}"


def _check_size(repo_root: Path, commit: str) -> None:
    sizes: dict[str, int] = {}
    for line in _split_z(
        _run_git(["ls-tree", "-r", "-l", "-z", commit], cwd=repo_root)
    ):
        meta, _, path = line.partition("\t")
        mode, _, _, size = meta.split()
        if mode == "160000":  # gitlink: its files live in another repository
            raise RuntimeError(
                f"{path} is a git submodule, which furu does not support.\n"
                "Are you sure you put it in the correct place?"
            )
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
        f"code snapshot of {commit[:12]} would be {total} "
        f"(limit: {limit_text}). Largest committed files:\n"
        f"{offenders}\n"
        "Remove them from git (git rm --cached <path>, then add them to .gitignore),\n"
        "or raise [tool.furu.provenance] max_snapshot_bytes if this is intentional."
    )


def _split_z(output: str) -> list[str]:
    return [token for token in output.split("\0") if token]
