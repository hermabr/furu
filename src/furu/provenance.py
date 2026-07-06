from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import socket
import subprocess
import sys
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
from getpass import getuser
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_ACCELERATOR_PROBE_TIMEOUT_SECONDS = 2.0
_HASH_PREFIX = "blake2s:"


class GitIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    commit: str
    branch: str | None
    remote: str | None
    repo_root: str
    dirty: bool
    diff_stats: str | None

    @classmethod
    def capture(cls, cwd: Path) -> GitIdentity:
        try:
            repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
            commit = _run_git(["rev-parse", "HEAD"], cwd=cwd)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot capture git identity from {cwd}") from exc
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
        try:
            remote = _run_git(["remote", "get-url", "origin"], cwd=cwd)
        except subprocess.CalledProcessError:
            remote = None
        dirty = bool(_run_git(["status", "--porcelain"], cwd=cwd))
        diff_stats = (
            _run_git(["diff", "HEAD", "--shortstat"], cwd=cwd) or None
            if dirty
            else None
        )
        return cls(
            commit=commit,
            branch=None if branch == "HEAD" else branch,
            remote=remote,
            repo_root=repo_root,
            dirty=dirty,
            diff_stats=diff_stats,
        )


class EnvironmentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    python: str
    uv: str
    project_root: str
    uv_lock_hash: str
    pyproject_hash: str
    furu: str

    @classmethod
    @functools.cache
    def capture(cls) -> EnvironmentIdentity:
        start = Path.cwd().resolve()
        project_root = next(
            (
                directory
                for directory in (start, *start.parents)
                if (directory / "pyproject.toml").is_file()
            ),
            None,
        )
        if project_root is None:
            raise RuntimeError(
                f"no pyproject.toml found from {start} upward.\n"
                "furu requires a uv project so results are reproducible. Create one with:\n"
                "  uv init"
            )
        uv_lock = project_root / "uv.lock"
        if not uv_lock.is_file():
            raise RuntimeError(
                f"no uv.lock beside {project_root / 'pyproject.toml'}.\n"
                "furu requires a locked uv project so results are reproducible. "
                "Run:\n"
                "  uv sync"
            )

        def hash_file(path: Path) -> str:
            return (
                _HASH_PREFIX
                + hashlib.blake2s(path.read_bytes(), digest_size=10).hexdigest()
            )

        try:
            uv = next(
                parts[2].strip()
                for line in (Path(sys.prefix) / "pyvenv.cfg")
                .read_text(encoding="utf-8")
                .splitlines()
                if (parts := line.partition("="))[1] and parts[0].strip() == "uv"
            )
        except (OSError, StopIteration) as exc:
            raise RuntimeError(
                "furu must run under a uv-managed interpreter so results are "
                "reproducible. Run furu commands via:\n  uv run ..."
            ) from exc

        return cls(
            python="{}.{}.{}".format(*sys.version_info[:3]),
            uv=uv,
            project_root=str(project_root),
            uv_lock_hash=hash_file(uv_lock),
            pyproject_hash=hash_file(project_root / "pyproject.toml"),
            furu=version("furu"),
        )


class SubmitContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    hostname: str
    user: str
    cwd: str
    launch_command: tuple[str, ...]
    timestamp: datetime

    @classmethod
    def capture(cls) -> SubmitContext:
        return cls(
            hostname=socket.gethostname(),
            user=getuser(),
            cwd=str(Path.cwd()),
            launch_command=tuple(sys.orig_argv),
            timestamp=datetime.now(timezone.utc),
        )


class ExecuteContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    hostname: str
    cpu_count: int
    accelerators: tuple[str, ...]
    slurm_job_id: str | None
    worker_backend: str
    pid: int

    @classmethod
    def capture(cls) -> ExecuteContext:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)()
        return cls(
            hostname=socket.gethostname(),
            cpu_count=cpu_count or 0,
            accelerators=_probe_accelerators(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            worker_backend=_worker_backend.get(),
            pid=os.getpid(),
        )


class SubmitProvenance(BaseModel):
    """The submit-side half of a Provenance record; travels inside a Job."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    git: GitIdentity
    environment: EnvironmentIdentity
    snapshot_id: str | None
    submitted: SubmitContext


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    git: GitIdentity
    environment: EnvironmentIdentity
    snapshot_id: str | None
    submitted: SubmitContext
    executed: ExecuteContext

    @classmethod
    def merge(cls, submit: SubmitProvenance, executed: ExecuteContext) -> Provenance:
        return cls(
            git=submit.git,
            environment=submit.environment,
            snapshot_id=submit.snapshot_id,
            submitted=submit.submitted,
            executed=executed,
        )


_worker_backend: ContextVar[str] = ContextVar("_furu_worker_backend", default="local")


def _run_git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@functools.cache
def _probe_accelerators() -> tuple[str, ...]:
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_ACCELERATOR_PROBE_TIMEOUT_SECONDS,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0 and names:
            return tuple(
                name if count == 1 else f"{name} ×{count}"
                for name, count in Counter(names).items()
            )
    visible = [
        device
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    if visible:
        return (f"cuda ×{len(visible)}",)
    return ()
