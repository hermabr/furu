"""Folder-name attempt locking under ``attempts/<run>/``.

Each run has at most one active non-dotted working directory inside
``attempts/<run>/``, whose name encodes the lock state and owner:

- ``free__<tok>`` — released and adoptable by any worker (``free__seed`` initially).
- ``slurm__<cluster>__<jobid>__<restart>__<tok>`` — owned by a Slurm job.
- ``local__<host>__<pid>__<tok>`` — owned by a local process.

Dotted entries (``.trash__*``, ``.deleting__*``) are ignored.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
import socket
import subprocess
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from furu.locking import LockError
from furu.storage._layout import revalidated_listdir
from furu.worker.backends.slurm.pool import _slurm

type SlurmRunner = Callable[..., str]

_CLEAN_SLURM_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "DEADLINE",
        "REVOKED",
    }
)
_SLURM_JOBID_RE = re.compile(r"[0-9]+(?:_[0-9]+)?")


class EntryStatus(Enum):
    ADOPTABLE = "adoptable"
    HELD = "held"
    UNKNOWN = "unknown"


def _non_dotted(entries: Sequence[str]) -> list[str]:
    return sorted(name for name in entries if not name.startswith("."))


def _current_host() -> str:
    return socket.gethostname().replace("__", "_")


def _current_cluster() -> str:
    return (os.environ.get("SLURM_CLUSTER_NAME") or "unknown").replace("__", "_")


def _owner_prefix() -> str:
    if job_id := os.environ.get("SLURM_JOB_ID"):
        cluster = _current_cluster()
        restart = (os.environ.get("SLURM_RESTART_COUNT") or "0").replace("__", "_")
        return f"slurm__{cluster}__{job_id}__{restart}"
    return f"local__{_current_host()}__{os.getpid()}"


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _normalize_sacct_state(raw_state: str) -> str:
    stripped = raw_state.strip()
    if stripped.startswith("CANCELLED by "):
        return "CANCELLED"
    return stripped


def _classify_slurm_owner(
    *,
    cluster: str,
    job_id: str,
    slurm_run: SlurmRunner | None = None,
    current_cluster: str | None = None,
) -> tuple[EntryStatus, str]:
    my_cluster = _current_cluster() if current_cluster is None else current_cluster
    if cluster != my_cluster:
        return (
            EntryStatus.UNKNOWN,
            f"slurm cluster mismatch ({cluster!r} != {my_cluster!r})",
        )

    runner = _slurm if slurm_run is None else slurm_run

    try:
        squeue_out = runner(
            "squeue",
            "--all",
            "--noheader",
            "-O",
            "JobIDRaw:32,State:32",
            "-j",
            job_id,
        )
    except subprocess.CalledProcessError as exc:
        if "invalid job id" in (exc.stderr or "").lower():
            squeue_out = ""
        else:
            return (EntryStatus.UNKNOWN, f"squeue failed: {exc}")
    except (subprocess.SubprocessError, OSError) as exc:
        return (EntryStatus.UNKNOWN, f"squeue unavailable: {exc}")

    if any(line.strip() for line in squeue_out.splitlines()):
        return (EntryStatus.HELD, f"slurm job {job_id} is active in squeue")

    try:
        sacct_out = runner(
            "sacct",
            "-D",
            "-X",
            "--noheader",
            "--parsable2",
            "-o",
            "JobIDRaw,State,Reason",
            "-j",
            job_id,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return (EntryStatus.UNKNOWN, f"sacct unavailable: {exc}")

    matched = False
    for line in sacct_out.splitlines():
        if not line.strip():
            continue
        cols = line.split("|", maxsplit=2)
        if len(cols) != 3:
            return (EntryStatus.UNKNOWN, f"malformed sacct row: {line!r}")
        row_job_id, raw_state, reason = cols[0].strip(), cols[1].strip(), cols[2].strip()
        if row_job_id == job_id:
            matched = True
        state = _normalize_sacct_state(raw_state)
        if state not in _CLEAN_SLURM_STATES:
            return (
                EntryStatus.UNKNOWN,
                f"slurm job {job_id} has state {raw_state!r}",
            )
        if reason == "Kill task failed":
            return (
                EntryStatus.UNKNOWN,
                f"slurm job {job_id} had Reason='Kill task failed'",
            )

    if not matched:
        return (EntryStatus.UNKNOWN, f"sacct returned no row matching {job_id}")
    return (EntryStatus.ADOPTABLE, f"slurm job {job_id} exited cleanly")


def classify_entry(
    name: str,
    *,
    slurm_run: SlurmRunner | None = None,
    current_host: str | None = None,
    current_cluster: str | None = None,
) -> tuple[EntryStatus, str]:
    """Classify one non-dotted entry inside ``attempts/<run>/``."""
    parts = name.split("__")
    if len(parts) == 2 and parts[0] == "free" and parts[1]:
        return (EntryStatus.ADOPTABLE, "free")

    if (
        len(parts) == 4
        and parts[0] == "local"
        and parts[1]
        and parts[2].isdigit()
        and int(parts[2]) > 0
        and parts[3]
    ):
        host, pid = parts[1], int(parts[2])
        my_host = _current_host() if current_host is None else current_host
        if host == my_host and not _is_pid_alive(pid):
            return (EntryStatus.ADOPTABLE, f"local process {pid} on {host} is gone")
        return (EntryStatus.HELD, f"local process {pid} on {host}")

    if (
        len(parts) == 5
        and parts[0] == "slurm"
        and parts[1]
        and _SLURM_JOBID_RE.fullmatch(parts[2])
        and parts[3].isdigit()
        and parts[4]
    ):
        return _classify_slurm_owner(
            cluster=parts[1],
            job_id=parts[2],
            slurm_run=slurm_run,
            current_cluster=current_cluster,
        )

    return (EntryStatus.UNKNOWN, f"malformed lock entry {name!r}")


def _blocked_error(
    attempts_dir: Path, entry: str, status: EntryStatus, reason: str
) -> LockError:
    folder_path = attempts_dir / entry
    if entry.startswith("local__") and status is EntryStatus.HELD:
        return LockError(
            f"attempt {attempts_dir} is locked by {folder_path}; if the process is "
            f"dead, mv {folder_path} {attempts_dir / 'free__manual'} to resume "
            "checkpoints or remove it"
        )
    return LockError(
        f"attempt {attempts_dir} is blocked by {status.value} folder {folder_path} ({reason})"
    )


@dataclass(slots=True)
class AttemptLock:
    """An active claimed working directory inside ``attempts/<run>/``."""

    attempts_dir: Path
    path: Path
    claimed_name: str
    _published: bool = False

    def is_held(self) -> bool:
        return self.claimed_name in _non_dotted(revalidated_listdir(self.attempts_dir))

    def mark_published(self) -> None:
        self._published = True

    def release(self) -> None:
        if self._published:
            return
        free_name = f"free__{secrets.token_hex(8)}"
        dst = self.attempts_dir / free_name
        try:
            os.rename(self.path, dst)
        except OSError as exc:
            if free_name in _non_dotted(revalidated_listdir(self.attempts_dir)):
                return
            if exc.errno in (errno.ENOENT, errno.ESTALE):
                return
            raise


def _is_published(run: str, runs_dirs: Sequence[Path]) -> bool:
    return any(run in revalidated_listdir(runs_dir) for runs_dir in runs_dirs)


def acquire(
    attempts_dir: Path,
    run: str,
    *,
    runs_dirs: Sequence[Path] = (),
    slurm_run: SlurmRunner | None = None,
) -> AttemptLock | None:
    """Acquire exclusive ownership of ``attempts/<run>/`` or return None if published."""
    if _is_published(run, runs_dirs):
        return None

    attempts_dir.mkdir(parents=True, exist_ok=True)
    entries = _non_dotted(revalidated_listdir(attempts_dir))
    if not entries:
        try:
            (attempts_dir / "free__seed").mkdir()
        except FileExistsError:
            pass
        entries = _non_dotted(revalidated_listdir(attempts_dir))
        if not entries:
            if _is_published(run, runs_dirs):
                return None
            raise LockError(f"attempt directory {attempts_dir} is empty after seeding")

    classified = [
        (entry, *classify_entry(entry, slurm_run=slurm_run)) for entry in entries
    ]
    for entry, status, reason in classified:
        if status is not EntryStatus.ADOPTABLE:
            raise _blocked_error(attempts_dir, entry, status, reason)
    if len(entries) != 1:
        raise LockError(f"multiple attempt folders in {attempts_dir}: {entries}")

    src_name = entries[0]
    src = attempts_dir / src_name
    dst_name = f"{_owner_prefix()}__{secrets.token_hex(8)}"
    dst = attempts_dir / dst_name
    try:
        os.rename(src, dst)
    except OSError as exc:
        after = _non_dotted(revalidated_listdir(attempts_dir))
        if dst_name in after:
            return AttemptLock(
                attempts_dir=attempts_dir, path=dst, claimed_name=dst_name
            )
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            if _is_published(run, runs_dirs):
                return None
            for entry in after:
                status, reason = classify_entry(entry, slurm_run=slurm_run)
                if status is not EntryStatus.ADOPTABLE:
                    raise _blocked_error(attempts_dir, entry, status, reason) from exc
        raise LockError(f"failed to claim {src} as {dst}: {exc}") from exc

    after = _non_dotted(revalidated_listdir(attempts_dir))
    if dst_name not in after:
        raise LockError(f"lost ownership of {dst} immediately after rename")
    return AttemptLock(attempts_dir=attempts_dir, path=dst, claimed_name=dst_name)


@contextmanager
def acquire_attempt(
    attempts_dir: Path,
    run: str,
    *,
    runs_dirs: Sequence[Path] = (),
    slurm_run: SlurmRunner | None = None,
) -> Generator[AttemptLock | None]:
    """Acquire an attempt lock and release it to ``free__<tok>`` unless published."""
    attempt_lock = acquire(
        attempts_dir, run, runs_dirs=runs_dirs, slurm_run=slurm_run
    )
    if attempt_lock is None:
        yield None
        return
    try:
        yield attempt_lock
    finally:
        attempt_lock.release()


def is_attempt_blocked(
    attempts_dir: Path, *, slurm_run: SlurmRunner | None = None
) -> bool:
    """Whether ``attempts/<run>/`` holds any non-adoptable (HELD or UNKNOWN) folder."""
    try:
        entries = _non_dotted(os.listdir(attempts_dir))
    except FileNotFoundError:
        return False
    return any(
        classify_entry(entry, slurm_run=slurm_run)[0] is not EntryStatus.ADOPTABLE
        for entry in entries
    )


def inspect_attempt_dir(
    attempts_dir: Path, *, slurm_run: SlurmRunner | None = None
) -> list[Path]:
    """Return adoptable folders in ``attempts/<run>/``, refusing if any is HELD/UNKNOWN."""
    if not attempts_dir.is_dir():
        return []
    entries = _non_dotted(revalidated_listdir(attempts_dir))
    for entry in entries:
        status, reason = classify_entry(entry, slurm_run=slurm_run)
        if status is not EntryStatus.ADOPTABLE:
            raise _blocked_error(attempts_dir, entry, status, reason)
    return [attempts_dir / entry for entry in entries]
