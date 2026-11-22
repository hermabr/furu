"""Folder-name attempt locking for runs in ``attempts/<run>/``.

Each run gets a parent folder ``attempts/<run>/`` that holds at most one active
non-dotted working directory whose name encodes ownership:

- ``free__<tok>`` — released, adoptable by any worker (``free__seed`` initially,
  fresh 64-bit random hex token on every release to prevent ABA races).
- ``slurm__<cluster>__<jobid>__<restart>__<tok>`` — claimed by a Slurm job.
- ``local__<host>__<pid>__<tok>`` — claimed by a local process.

Dotted entries (``.trash__*``, ``.deleting__*``) are ignored by readers and
lock acquisition.
"""

from __future__ import annotations

import errno
import os
import secrets
import shutil
import socket
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from furu.locking import LockError
from furu.storage._layout import log_path_in, scratch_dir_in
from furu.worker.backends.slurm._command import run_slurm

_MAX_CLAIM_RETRIES = 16

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


class AttemptState(Enum):
    ADOPTABLE = "adoptable"
    HELD = "held"
    UNKNOWN = "unknown"


class AttemptLockedError(LockError):
    """An attempt folder is held by another owner or in an unknown state."""


@dataclass(frozen=True, slots=True)
class Owner:
    """Identity prefix of the worker claiming an attempt folder."""

    prefix: str

    @classmethod
    def current(cls) -> Owner:
        job_id = os.environ.get("SLURM_JOB_ID", "").strip()
        cluster = os.environ.get("SLURM_CLUSTER_NAME", "").strip()
        if job_id and cluster:
            restart = os.environ.get("SLURM_RESTART_COUNT", "").strip() or "0"
            return cls(f"slurm__{cluster}__{job_id}__{restart}")
        return cls(f"local__{socket.gethostname()}__{os.getpid()}")


@dataclass(frozen=True, slots=True)
class SlurmQuery:
    """Injectable Slurm CLI runner for 2-gate Slurm owner liveness checks."""

    run: Callable[..., str] = run_slurm
    cluster: str | None = None

    def current_cluster(self) -> str | None:
        if self.cluster is not None:
            return self.cluster or None
        if env_cluster := os.environ.get("SLURM_CLUSTER_NAME", "").strip():
            return env_cluster
        try:
            output = self.run("scontrol", "show", "config")
        except (subprocess.SubprocessError, OSError):
            return None
        for line in output.splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "ClusterName":
                name = value.strip()
                return name or None
        return None


_DEFAULT_SLURM = SlurmQuery()


def revalidate_listdir(path: Path) -> list[str]:
    """List ``path``, revalidating parent directories on ENOENT to bypass NFS negative dcache."""
    try:
        return os.listdir(path)
    except FileNotFoundError:
        try:
            for parent in reversed(path.parents[:4]):
                os.listdir(parent)
            return os.listdir(path)
        except FileNotFoundError:
            return []


def list_attempt_entries(attempt_root: Path) -> list[str]:
    """Sorted non-dotted entries in ``attempts/<run>/``."""
    return sorted(
        name for name in revalidate_listdir(attempt_root) if not name.startswith(".")
    )


def active_attempt_dir(attempt_root: Path) -> Path | None:
    """The active working folder inside ``attempts/<run>/``, if any."""
    entries = list_attempt_entries(attempt_root)
    return attempt_root / entries[0] if entries else None


def attempt_log_path(attempt_root: Path) -> Path:
    """Path to the active attempt's log, or ``attempt_root`` before one exists."""
    if (active := active_attempt_dir(attempt_root)) is not None:
        return log_path_in(active)
    return attempt_root


def _default_is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_local(name: str) -> tuple[str, int, str] | None:
    parts = name.split("__")
    if (
        len(parts) == 4
        and parts[0] == "local"
        and parts[1]
        and parts[2].isdigit()
        and int(parts[2]) > 0
        and parts[3]
    ):
        return parts[1], int(parts[2]), parts[3]
    return None


def _parse_slurm(name: str) -> tuple[str, str, str, str] | None:
    parts = name.split("__")
    if (
        len(parts) == 5
        and parts[0] == "slurm"
        and parts[1]
        and parts[2].isdigit()
        and parts[3].isdigit()
        and parts[4]
    ):
        return parts[1], parts[2], parts[3], parts[4]
    return None


def _parse_free(name: str) -> str | None:
    parts = name.split("__")
    if len(parts) == 2 and parts[0] == "free" and parts[1]:
        return parts[1]
    return None


def _classify_slurm(cluster: str, job_id: str, *, slurm: SlurmQuery) -> AttemptState:
    """Conservative 2-gate Slurm owner check: cluster match, absent from squeue, clean sacct."""
    current = slurm.current_cluster()
    if current is None or cluster != current:
        return AttemptState.UNKNOWN

    # Gate 1: must be completely absent from squeue.
    try:
        squeue_out = slurm.run(
            "squeue",
            "--all",
            "--noheader",
            "-O",
            "JobIDRaw:32,State:32",
            "-j",
            job_id,
        )
    except subprocess.CalledProcessError as exc:
        if "invalid job id" not in (exc.stderr or "").lower():
            return AttemptState.UNKNOWN
    except (subprocess.SubprocessError, OSError):
        return AttemptState.UNKNOWN
    else:
        if any(line.strip() for line in squeue_out.splitlines()):
            return AttemptState.HELD

    # Gate 2: sacct must confirm a clean terminal exit with no kill failure.
    try:
        sacct_out = slurm.run(
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
    except (subprocess.SubprocessError, OSError):
        return AttemptState.UNKNOWN

    matched = 0
    for line in sacct_out.splitlines():
        if not line.strip():
            continue
        columns = line.split("|")
        if len(columns) != 3:
            return AttemptState.UNKNOWN
        row_job_id, raw_state, reason = (col.strip() for col in columns)
        if row_job_id != job_id:
            continue
        matched += 1
        state = "CANCELLED" if raw_state.startswith("CANCELLED by ") else raw_state
        if state not in _CLEAN_SLURM_STATES:
            return AttemptState.UNKNOWN
        if reason.lower() == "kill task failed":
            return AttemptState.UNKNOWN

    return AttemptState.ADOPTABLE if matched > 0 else AttemptState.UNKNOWN


def classify_entry(
    name: str,
    *,
    slurm: SlurmQuery = _DEFAULT_SLURM,
    local_host: str | None = None,
    is_pid_alive: Callable[[int], bool] | None = None,
) -> AttemptState:
    """Defensively classify one non-dotted entry inside ``attempts/<run>/``."""
    if _parse_free(name) is not None:
        return AttemptState.ADOPTABLE
    if (local := _parse_local(name)) is not None:
        host, pid, _ = local
        my_host = socket.gethostname() if local_host is None else local_host
        if host != my_host:
            return AttemptState.UNKNOWN
        alive = _default_is_pid_alive if is_pid_alive is None else is_pid_alive
        return AttemptState.HELD if alive(pid) else AttemptState.ADOPTABLE
    if (slurm_entry := _parse_slurm(name)) is not None:
        cluster, job_id, _, _ = slurm_entry
        return _classify_slurm(cluster, job_id, slurm=slurm)
    return AttemptState.UNKNOWN


def _blocked_error(
    attempt_root: Path, name: str, state: AttemptState
) -> AttemptLockedError:
    entry_path = attempt_root / name
    if (local := _parse_local(name)) is not None:
        host, pid, _ = local
        manual_target = attempt_root / "free__manual"
        return AttemptLockedError(
            f"attempt is blocked by local owner folder {entry_path} "
            f"(host={host}, pid={pid}, state={state.value}). "
            f"If that process is dead, run `mv {entry_path} {manual_target}` "
            f"to resume its checkpoints, or remove {entry_path}."
        )
    if (slurm_entry := _parse_slurm(name)) is not None:
        cluster, job_id, restart, _ = slurm_entry
        return AttemptLockedError(
            f"attempt is blocked by Slurm owner folder {entry_path} "
            f"(cluster={cluster}, job_id={job_id}, restart={restart}, "
            f"state={state.value})"
        )
    return AttemptLockedError(
        f"attempt is blocked by unrecognized folder {entry_path} "
        f"(state={state.value}); refusing to adopt or overwrite it"
    )


def _check_entries(
    attempt_root: Path,
    entries: list[str],
    *,
    slurm: SlurmQuery = _DEFAULT_SLURM,
    local_host: str | None = None,
    is_pid_alive: Callable[[int], bool] | None = None,
) -> None:
    """Raise ``AttemptLockedError`` if any entry is HELD or UNKNOWN."""
    for name in entries:
        state = classify_entry(
            name, slurm=slurm, local_host=local_host, is_pid_alive=is_pid_alive
        )
        if state is not AttemptState.ADOPTABLE:
            raise _blocked_error(attempt_root, name, state)


def is_attempt_blocked(
    attempt_root: Path,
    *,
    slurm: SlurmQuery = _DEFAULT_SLURM,
    local_host: str | None = None,
    is_pid_alive: Callable[[int], bool] | None = None,
) -> bool:
    """Whether ``attempts/<run>/`` holds any HELD or UNKNOWN attempt folder."""
    for name in list_attempt_entries(attempt_root):
        state = classify_entry(
            name, slurm=slurm, local_host=local_host, is_pid_alive=is_pid_alive
        )
        if state is not AttemptState.ADOPTABLE:
            return True
    return False


def inspect_deletable_attempts(
    attempt_root: Path,
    *,
    slurm: SlurmQuery = _DEFAULT_SLURM,
    local_host: str | None = None,
    is_pid_alive: Callable[[int], bool] | None = None,
) -> list[Path]:
    """Return adoptable attempt folders in ``attempt_root``, refusing if any are HELD/UNKNOWN."""
    entries = list_attempt_entries(attempt_root)
    _check_entries(
        attempt_root,
        entries,
        slurm=slurm,
        local_host=local_host,
        is_pid_alive=is_pid_alive,
    )
    return [attempt_root / name for name in entries]


@dataclass
class ClaimedAttempt:
    """An exclusively claimed attempt working folder inside ``attempts/<run>/``."""

    path: Path
    published: bool = False
    released: bool = False

    def still_owned(self) -> bool:
        """Whether this claimed folder still exists under its claimed name."""
        return self.path.name in revalidate_listdir(self.path.parent)

    def release(self) -> Path | None:
        """Rename the claimed folder to ``free__<fresh_tok>`` on clean exit without publish."""
        if self.published or self.released:
            return None
        self.released = True
        target = self.path.parent / f"free__{secrets.token_hex(8)}"
        try:
            os.rename(self.path, target)
            return target
        except FileNotFoundError:
            return None
        except OSError:
            if target.name in revalidate_listdir(self.path.parent):
                return target
            raise

    def discard(self) -> None:
        """Atomically move an unused claimed folder to ``.trash__<tok>`` and remove it."""
        if self.published or self.released:
            return
        self.released = True
        trash = self.path.parent / f".trash__{secrets.token_hex(8)}"
        try:
            os.rename(self.path, trash)
        except FileNotFoundError:
            return
        shutil.rmtree(trash, ignore_errors=True)
        with suppress(OSError):
            self.path.parent.rmdir()

    def publish(self, destination: Path) -> None:
        """Crash-safely publish this attempt folder to ``destination`` (``runs/<run_name>``).

        Moves ``scratch/`` aside atomically before publishing. On ``OSError`` from
        the publish rename, revalidates ``destination.parent``: if nothing is
        published, re-raises with this attempt intact; if a winner is already
        published, atomically trashes this loser copy.
        """
        scratch_trash: Path | None = None
        scratch = scratch_dir_in(self.path)
        if scratch.exists():
            scratch_trash = self.path.parent / f".trash__{secrets.token_hex(8)}"
            try:
                os.rename(scratch, scratch_trash)
            except FileNotFoundError:
                scratch_trash = None

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.rename(destination)
        except OSError:
            if destination.name not in revalidate_listdir(destination.parent):
                raise
            # A winner is already published (or our own rename succeeded on NFS).
            if self.path.name in revalidate_listdir(self.path.parent):
                loser_trash = self.path.parent / f".trash__{secrets.token_hex(8)}"
                try:
                    os.rename(self.path, loser_trash)
                except FileNotFoundError:
                    pass
                else:
                    shutil.rmtree(loser_trash, ignore_errors=True)

        self.published = True
        if scratch_trash is not None:
            shutil.rmtree(scratch_trash, ignore_errors=True)
        with suppress(OSError):
            self.path.parent.rmdir()


def acquire(
    attempt_root: Path,
    *,
    is_published: Callable[[], bool] | None = None,
    owner: Owner | None = None,
    slurm: SlurmQuery = _DEFAULT_SLURM,
    local_host: str | None = None,
    is_pid_alive: Callable[[int], bool] | None = None,
) -> ClaimedAttempt | None:
    """Acquire exclusive ownership of the single working folder inside ``attempt_root``.

    Returns ``None`` if ``is_published()`` reports the run is already published.
    Raises ``AttemptLockedError`` if any non-dotted entry is ``HELD`` or ``UNKNOWN``.
    """
    me = Owner.current() if owner is None else owner
    for _ in range(_MAX_CLAIM_RETRIES):
        # 1. Check if already published via revalidated directory listing.
        if is_published is not None and is_published():
            return None

        # 2. List non-dotted entries in attempts/<run>/.
        attempt_root.mkdir(parents=True, exist_ok=True)
        entries = list_attempt_entries(attempt_root)

        # 3. If empty, seed with free__seed (catching FileExistsError) and re-list.
        if not entries:
            try:
                (attempt_root / "free__seed").mkdir()
            except FileExistsError:
                pass
            except FileNotFoundError:
                # attempt_root was removed concurrently (e.g., publisher finished).
                continue
            entries = list_attempt_entries(attempt_root)
            if not entries:
                continue

        # 4 & 5. Defensively classify every entry; stop and refuse if any is HELD/UNKNOWN.
        _check_entries(
            attempt_root,
            entries,
            slurm=slurm,
            local_host=local_host,
            is_pid_alive=is_pid_alive,
        )

        # 6. Atomically claim entries[0] with a fresh 64-bit random hex token.
        src = attempt_root / entries[0]
        dst_name = f"{me.prefix}__{secrets.token_hex(8)}"
        dst = attempt_root / dst_name
        try:
            os.rename(src, dst)
        except OSError as exc:
            # Ambiguous NFS error: re-check whether our rename actually took effect.
            if dst_name in revalidate_listdir(attempt_root):
                return ClaimedAttempt(path=dst)
            if exc.errno in (
                errno.ENOENT,
                errno.EEXIST,
                errno.ENOTEMPTY,
                errno.ESTALE,
            ):
                continue
            raise

        if dst_name in revalidate_listdir(attempt_root):
            return ClaimedAttempt(path=dst)

    raise AttemptLockedError(
        f"could not claim an attempt folder in {attempt_root} after {_MAX_CLAIM_RETRIES} attempts"
    )


@contextmanager
def acquire_attempt(
    attempt_root: Path,
    *,
    is_published: Callable[[], bool] | None = None,
    owner: Owner | None = None,
    slurm: SlurmQuery = _DEFAULT_SLURM,
    local_host: str | None = None,
    is_pid_alive: Callable[[int], bool] | None = None,
) -> Generator[ClaimedAttempt | None]:
    """Claim an attempt folder, releasing it to ``free__<fresh_tok>`` if not published."""
    claimed = acquire(
        attempt_root,
        is_published=is_published,
        owner=owner,
        slurm=slurm,
        local_host=local_host,
        is_pid_alive=is_pid_alive,
    )
    try:
        yield claimed
    finally:
        if claimed is not None and not claimed.published and not claimed.released:
            claimed.release()
