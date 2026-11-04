from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import tomllib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from furu import code_trace
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
    for _, directory in get_config().directories.anchored():
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
    pushed_to = _run_git(
        ["for-each-ref", "--contains", git.commit, "refs/remotes/origin"],
        cwd=repo_root,
    )
    if not pushed_to:
        add_remote = "" if git.remote else "  git remote add origin <url>\n"
        raise RuntimeError(
            f"HEAD ({git.commit[:12]}) is not on any branch of origin "
            f"({git.remote or 'no remote named origin'}).\n"
            "furu only runs pushed code, so every result points at a commit "
            "others can fetch. Push it:\n"
            f"{add_remote}  {_push_command(git)}"
        )


def require_path_dependencies_in_repo(git: GitIdentity, project_root: Path) -> None:
    """Refuse uv.lock path dependencies outside the repo: workers run a git
    archive of the commit, which would not contain them."""
    repo_root = Path(git.repo_root).resolve()
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    outside = [
        f"  {package['name']}: {path}"
        for package in lock.get("package", [])
        for kind in ("editable", "directory", "path", "virtual")
        if (path := package.get("source", {}).get(kind)) is not None
        and not (project_root / path).resolve().is_relative_to(repo_root)
    ]
    if outside:
        raise RuntimeError(
            f"path dependencies outside the git repository {repo_root}:\n"
            + "\n".join(outside)
            + "\nfuru runs workers from a snapshot of the commit, which would not "
            "contain them.\nMove them inside the repository, or depend on them "
            "through git or a package index."
        )


def extract_snapshot(commit: str, *, repo: Path) -> Path:
    """Materialize ``commit`` from ``repo``'s objects; return the tree.

    The tree lands in ``<snapshots>/<commit>`` once and is reused by every later
    caller.
    """
    code_dir = get_config().directories.anchored().snapshots / commit
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
    repo_root: Path  # traced code paths are relative to it

    @classmethod
    def here(cls) -> CodeLocation:
        return cls(
            python=Path(sys.executable),
            project_root=Path(EnvironmentIdentity.capture().project_root),
            cwd=Path.cwd(),
            repo_root=code_trace.repo_root(),
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
            repo_root=code_dir,
        )


def _push_command(git: GitIdentity) -> str:
    if git.branch is None:  # detached HEAD
        return "git push origin HEAD:refs/heads/<branch>"
    return f"git push -u origin {git.branch}"
