"""The name of a run's working directory is its ownership token."""

import os
import re
import secrets
import shlex
import socket
import stat
import subprocess
from collections.abc import Callable
from contextlib import suppress
from enum import Enum, auto
from pathlib import Path

from furu.storage._layout import log_path_in
from furu.worker.backends.slurm.command import slurm


class AttemptBusy(RuntimeError):
    """Another owner holds the folder, or its state cannot be determined."""


class State(Enum):
    ADOPTABLE = auto()
    HELD = auto()
    UNKNOWN = auto()


_CLEAN = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "DEADLINE",
    "REVOKED",
}
_TOKEN = r"[0-9a-f]{16}"


def _slurm_state(cluster: str, job: str) -> State:
    if cluster != os.environ.get("SLURM_CLUSTER_NAME"):
        return State.UNKNOWN
    try:
        try:
            queued = slurm(
                "squeue", "--all", "--noheader", "-O", "JobIDRaw:32,State:32", "-j", job
            )
        except subprocess.CalledProcessError as exc:
            # Slurm can report a purged job as an error instead of an empty queue.
            if (exc.stdout or "").strip() or not re.fullmatch(
                r"(?:squeue: error: )?(?:slurm_load_jobs error: )?Invalid job id specified",
                (exc.stderr or "").strip(),
                re.IGNORECASE,
            ):
                return State.UNKNOWN
        else:
            if queued.strip():
                return State.HELD
        accounts = slurm(
            "sacct",
            "-D",
            "-X",
            "--noheader",
            "--parsable2",
            "-o",
            "JobIDRaw,State,Reason",
            "-j",
            job,
        )
    except (OSError, subprocess.SubprocessError):
        return State.UNKNOWN

    found = False
    for line in accounts.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            return State.UNKNOWN
        job_id, state, reason = parts
        found |= job_id == job
        if re.fullmatch(r"CANCELLED by [0-9]+", state):
            state = "CANCELLED"
        if state not in _CLEAN or reason == "Kill task failed":
            return State.UNKNOWN
    return State.ADOPTABLE if found else State.UNKNOWN


def classify(path: Path) -> State:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return State.UNKNOWN
    if not stat.S_ISDIR(mode):
        return State.UNKNOWN
    match path.name.split("__"):
        case ["free", tok] if tok in {"seed", "manual"} or re.fullmatch(_TOKEN, tok):
            return State.ADOPTABLE
        case ["local", host, pid, tok] if (
            host and re.fullmatch(r"[1-9][0-9]*", pid) and re.fullmatch(_TOKEN, tok)
        ):
            return State.HELD
        case ["slurm", cluster, job, restart, tok] if (
            cluster
            and re.fullmatch(r"[0-9]+", job)
            and re.fullmatch(r"[0-9]+", restart)
            and re.fullmatch(_TOKEN, tok)
        ):
            return _slurm_state(cluster, job)
    return State.UNKNOWN


def _owner() -> str:
    if job := os.environ.get("SLURM_JOB_ID"):
        cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        if (
            not cluster
            or "__" in cluster
            or "/" in cluster
            or not re.fullmatch(r"[0-9]+", job)
            or not re.fullmatch(r"[0-9]+", restart)
        ):
            raise RuntimeError(
                "Cannot identify this Slurm allocation from its environment"
            )
        return f"slurm__{cluster}__{job}__{restart}"
    host = socket.gethostname()
    if not host or "__" in host or "/" in host:
        raise RuntimeError(f"Cannot use hostname {host!r} in an attempt name")
    return f"local__{host}__{os.getpid()}"


def _names(parent: Path) -> list[str]:
    try:
        return os.listdir(parent)
    except FileNotFoundError:
        return []


def entries(parent: Path) -> list[Path]:
    return sorted(parent / name for name in _names(parent) if not name.startswith("."))


def rename_if_present(source: Path, unique_target: Path) -> bool:
    """Move to a fresh name, resolving a lost NFS reply by listing its parent."""
    try:
        os.rename(source, unique_target)
    except OSError as exc:
        if unique_target.name not in _names(unique_target.parent):
            if isinstance(exc, FileNotFoundError) and source.name not in _names(
                source.parent
            ):
                return False
            raise
    return True


def working_dir(parent: Path) -> Path:
    """For diagnostics, resolve today's owner; before the first claim show the parent."""
    return next(iter(entries(parent)), parent)


def log(parent: Path) -> Path:
    """Before a worker has a log, point diagnostics at its stable run folder."""
    return next((log_path_in(folder) for folder in entries(parent)), parent)


def require_adoptable(folders: list[Path]) -> None:
    for folder in folders:
        state = classify(folder)
        if state is State.HELD and folder.name.startswith("local__"):
            raise AttemptBusy(
                f"Local attempt held at {folder}. If the process is dead, "
                f"mv {shlex.quote(str(folder))} {shlex.quote(str(folder.with_name('free__manual')))} "
                "to resume its checkpoints, or remove the folder."
            )
        if state is not State.ADOPTABLE:
            raise AttemptBusy(f"Cannot adopt {folder}: owner state is {state.name}")


def acquire(parent: Path, published: Callable[[], bool]) -> Path | None:
    """Claim a unique path, or return None if a live version already has the run."""
    while not published():
        folders = entries(parent)
        seed = parent / "free__seed"
        if not folders:
            parent.mkdir(parents=True, exist_ok=True)
            try:
                seed.mkdir()
            except FileExistsError:
                pass
            except FileNotFoundError:
                continue  # another worker published and removed the empty parent
            else:
                # A delayed initializer may plant its empty seed beside an owner
                # that already claimed the first seed. Undo only our empty mkdir.
                if len(entries(parent)) > 1:
                    with suppress(FileNotFoundError):
                        seed.rmdir()
            folders = entries(parent)
        require_adoptable(folders)
        if not folders:
            continue
        if len(folders) > 1:
            if seed in folders:
                # The delayed initializer may have died just after mkdir.
                # rmdir preserves any checkpoint data and any renamed claim.
                with suppress(FileNotFoundError):
                    seed.rmdir()
                continue
            raise RuntimeError(f"Multiple working folders in {parent}: {folders}")
        source = folders[0]
        claim = parent / f"{_owner()}__{secrets.token_hex(8)}"
        try:
            if not rename_if_present(source, claim):
                continue
            if entries(parent) != [claim]:
                raise RuntimeError(f"Cannot verify exclusive ownership of {claim}")
            if not published():
                return claim
            # A previous worker published between our first listing and seed.
        except BaseException:
            _abandon(claim, source == seed)
            raise
        _abandon(claim, source == seed)
        return None
    return None


def _abandon(claim: Path, seeded: bool) -> None:
    if seeded:
        # Only remove empty seeds. A restored checkpoint is always released.
        with suppress(OSError):
            claim.rmdir()
            with suppress(OSError):
                claim.parent.rmdir()
            return
    release(claim)


def release(claim: Path) -> None:
    rename_if_present(claim, claim.with_name(f"free__{secrets.token_hex(8)}"))
