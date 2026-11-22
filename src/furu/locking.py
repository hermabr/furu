"""NFS-safe locking: short O_EXCL locks for schema appends, and folder-name attempt locks.

An attempt's parent directory ``attempts/<run>/`` holds at most one non-dotted
working folder whose name encodes its current owner:

- ``free__<tok>`` (seeded via ``free__seed`` when empty)
- ``local__<host>__<pid>__<tok>``
- ``slurm__<cluster>__<jobid>__<restart>__<tok>`` (raw ``$SLURM_JOB_ID``)

Claiming an adoptable folder is an atomic ``os.rename()`` to our owner name with
a fresh 64-bit token, followed by a revalidated ``os.listdir()`` confirming it
is the sole non-dotted entry. Clean exit renames it back to ``free__<fresh_tok>``
unless it was published or deleted. Dotted entries (such as ``.trash__<tok>`` or
NFS ``.nfs*``) are ignored.
"""

from __future__ import annotations

import errno
import os
import random
import re
import secrets
import socket
import subprocess
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from furu.logging import get_logger
from furu.storage._layout import log_path_in

logger = get_logger()

DEFAULT_ACQUIRE_TIMEOUT_S = 45.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 0.05
DEFAULT_LOCK_WAIT_LOG_INTERVAL_S = 10.0
_SLURM_QUERY_TIMEOUT_S = 30.0

# Clean terminal Slurm states where the node confirmed job exit. NODE_FAIL,
# BOOT_FAIL, PREEMPTED, and any non-terminal or unknown state fail closed.
_CLEAN_SLURM_TERMINAL_STATES = frozenset(
    {
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "OUT_OF_MEMORY",
        "REVOKED",
        "TIMEOUT",
    }
)
_UNSAFE_SLURM_TERMINAL_STATES = frozenset({"NODE_FAIL", "BOOT_FAIL"})


class LockError(RuntimeError):
    """A furu lock could not be acquired, was lost, or is inconsistent on disk."""


def _is_missing_or_stale(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


@contextmanager
def short_lock(
    lock_path: Path,
    *,
    timeout_s: float = 30.0,
    stale_s: float = 30.0,
    poll_s: float = 0.05,
) -> Generator[None]:
    """Short-lived O_EXCL file lock for quick appends like ``schema.ndjson``."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if lock_path.stat().st_mtime + stale_s <= time.time():
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError as exc:
                if _is_missing_or_stale(exc):
                    continue
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockError(
                    f"could not acquire lock at {lock_path} within {timeout_s:g}s"
                ) from None
            time.sleep(min(poll_s, remaining))
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError as exc:
            if not _is_missing_or_stale(exc):
                raise


def revalidated_listdir(directory: Path, *, levels: int = 2) -> list[str] | None:
    """List ``directory``, revalidating ancestors on ENOENT to bypass NFS negative caches."""
    try:
        return os.listdir(directory)
    except OSError as exc:
        if not _is_missing_or_stale(exc):
            raise
    for ancestor in reversed(directory.parents[:levels]):
        try:
            os.listdir(ancestor)
        except OSError as exc:
            if _is_missing_or_stale(exc):
                return None
            raise
    try:
        return os.listdir(directory)
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


def _non_dotted(entries: list[str] | None) -> list[str]:
    if not entries:
        return []
    return sorted(name for name in entries if not name.startswith("."))


def active_attempt_dir(parent_dir: Path) -> Path | None:
    """The single active working folder inside ``attempts/<run>/``, or None."""
    active = _non_dotted(revalidated_listdir(parent_dir))
    return parent_dir / active[0] if len(active) == 1 else None


def attempt_log_path(parent_dir: Path) -> Path:
    """The log inside the active working folder, or ``parent_dir`` if none exists yet."""
    active = active_attempt_dir(parent_dir)
    return log_path_in(active) if active is not None else parent_dir


def owns_attempt(claimed: Path) -> bool:
    """Whether ``claimed`` is still the sole non-dotted entry in its attempt parent."""
    return _non_dotted(revalidated_listdir(claimed.parent)) == [claimed.name]


def _safe_field(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[/\s]+", "-", value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "-")
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _local_host() -> str:
    return _safe_field(socket.gethostname(), fallback="unknown")


def _fresh_token() -> str:
    return secrets.token_hex(8)


def _owner_folder_name() -> str:
    """Folder name encoding this process's ownership with a fresh 64-bit token."""
    tok = _fresh_token()
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if job_id:
        cluster = _safe_field(
            os.environ.get("SLURM_CLUSTER_NAME", ""), fallback="default"
        )
        raw_restart = os.environ.get("SLURM_RESTART_COUNT", "0").strip()
        restart = raw_restart if raw_restart.isdigit() else "0"
        return f"slurm__{cluster}__{_safe_field(job_id, fallback='0')}__{restart}__{tok}"
    return f"local__{_local_host()}__{os.getpid()}__{tok}"


@dataclass(frozen=True, slots=True)
class _FreeOwner:
    token: str


@dataclass(frozen=True, slots=True)
class _LocalOwner:
    host: str
    pid: int
    token: str


@dataclass(frozen=True, slots=True)
class _SlurmOwner:
    cluster: str
    job_id: str
    restart: int
    token: str


type _Owner = _FreeOwner | _LocalOwner | _SlurmOwner


def _parse_owner(name: str) -> _Owner | None:
    parts = name.split("__")
    match parts:
        case ["free", tok] if tok:
            return _FreeOwner(token=tok)
        case ["local", host, pid, tok] if (
            host and pid.isdigit() and int(pid) > 0 and tok
        ):
            return _LocalOwner(host=host, pid=int(pid), token=tok)
        case ["slurm", cluster, job_id, restart, tok] if (
            cluster and job_id and restart.isdigit() and tok
        ):
            return _SlurmOwner(
                cluster=cluster, job_id=job_id, restart=int(restart), token=tok
            )
        case _:
            return None


type _Status = Literal["adoptable", "held", "refuse"]


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _check_slurm_owner(path: Path, owner: _SlurmOwner) -> tuple[_Status, str]:
    """Conservative fail-closed 2-gate check for Slurm ownership.

    Gate 1: same cluster, and absent from ``squeue``.
    Gate 2: ``sacct -D -X`` returns >=1 row, every row is on the same cluster,
    has a state in ``_CLEAN_SLURM_TERMINAL_STATES`` (excluding ``NODE_FAIL`` and
    ``BOOT_FAIL``), and does not report ``"Kill task failed"``.
    """
    env_cluster = os.environ.get("SLURM_CLUSTER_NAME", "").strip()
    if env_cluster:
        current_cluster = _safe_field(env_cluster, fallback="default")
        if current_cluster != owner.cluster:
            return (
                "refuse",
                (
                    f"attempt folder {path} belongs to Slurm cluster {owner.cluster!r}, "
                    f"not current cluster {current_cluster!r}"
                ),
            )
    current_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if (
        current_job_id
        and _safe_field(env_cluster, fallback="default") == owner.cluster
        and _safe_field(current_job_id, fallback="0") == owner.job_id
    ):
        return (
            "held",
            f"attempt folder {path} is held by current Slurm job {owner.job_id}",
        )

    # Gate 1: absent from squeue.
    try:
        squeue = subprocess.run(
            ["squeue", "--noheader", "--jobs", owner.job_id, "--format=%A %T"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SLURM_QUERY_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            "held",
            f"cannot verify Slurm job {owner.job_id} for {path} via squeue: {exc}",
        )

    if squeue.returncode == 0:
        if squeue.stdout.strip():
            return (
                "held",
                f"attempt folder {path} is held by active Slurm job {owner.job_id}",
            )
    elif not (
        "Invalid job id specified" in squeue.stderr and not squeue.stdout.strip()
    ):
        detail = squeue.stderr.strip() or f"exit code {squeue.returncode}"
        return (
            "held",
            f"cannot verify Slurm job {owner.job_id} for {path} via squeue: {detail}",
        )

    # Gate 2: every sacct -D -X row is on the same cluster and cleanly terminal.
    try:
        sacct = subprocess.run(
            [
                "sacct",
                "-D",
                "-X",
                "--noheader",
                "--parsable2",
                "-o",
                "Cluster,State,Reason",
                "-j",
                owner.job_id,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_SLURM_QUERY_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            "held",
            f"cannot verify Slurm job {owner.job_id} for {path} via sacct: {exc}",
        )

    if sacct.returncode != 0:
        detail = sacct.stderr.strip() or f"exit code {sacct.returncode}"
        return (
            "held",
            f"cannot verify Slurm job {owner.job_id} for {path} via sacct: {detail}",
        )

    rows = [line for line in sacct.stdout.splitlines() if line.strip()]
    if not rows:
        return (
            "held",
            (
                f"sacct -D -X returned no rows for Slurm job {owner.job_id} ({path}); "
                "failing closed"
            ),
        )

    for row in rows:
        fields = row.split("|")
        if len(fields) != 3:
            return (
                "refuse",
                f"unexpected sacct output {row!r} for Slurm job {owner.job_id} ({path})",
            )
        row_cluster, row_state, row_reason = (f.strip() for f in fields)
        if _safe_field(row_cluster, fallback="") != owner.cluster:
            return (
                "refuse",
                (
                    f"attempt folder {path} belongs to Slurm cluster {owner.cluster!r}, "
                    f"but sacct reported cluster {row_cluster!r}"
                ),
            )
        base_state = row_state.partition(" ")[0]
        if "kill task failed" in row_reason.lower():
            return (
                "refuse",
                (
                    f"attempt folder {path} belongs to Slurm job {owner.job_id} with "
                    f"reason {row_reason!r}; refusing automatic takeover. Once you confirm "
                    f"the job is gone, rename it to {path.parent / 'free__manual'}."
                ),
            )
        if base_state in _UNSAFE_SLURM_TERMINAL_STATES:
            return (
                "refuse",
                (
                    f"attempt folder {path} belongs to Slurm job {owner.job_id} in unsafe "
                    f"state {row_state!r}; refusing automatic takeover. Once you confirm "
                    f"the job is gone, rename it to {path.parent / 'free__manual'}."
                ),
            )
        if base_state not in _CLEAN_SLURM_TERMINAL_STATES:
            return (
                "held",
                (
                    f"attempt folder {path} belongs to Slurm job {owner.job_id} in "
                    f"non-terminal state {row_state!r}"
                ),
            )

    return "adoptable", ""


def _inspect_entry(path: Path) -> tuple[_Status, str]:
    owner = _parse_owner(path.name)
    if owner is None:
        return (
            "refuse",
            (
                f"unexpected or malformed entry {path} in attempt directory {path.parent}; "
                "refusing to acquire"
            ),
        )
    if isinstance(owner, _FreeOwner):
        return "adoptable", ""
    if isinstance(owner, _LocalOwner):
        my_host = _local_host()
        if owner.host != my_host:
            return (
                "refuse",
                (
                    f"attempt folder {path} is held by local PID {owner.pid} on host "
                    f"{owner.host!r} (this host is {my_host!r}). If that process is no "
                    f"longer running, release it by renaming {path} to "
                    f"{path.parent / 'free__manual'} (or remove it to start over)."
                ),
            )
        if _pid_is_alive(owner.pid):
            return (
                "held",
                (
                    f"attempt folder {path} is held by live local PID {owner.pid} on "
                    f"{owner.host!r}"
                ),
            )
        return "adoptable", ""
    return _check_slurm_owner(path, owner)


def is_attempt_held(parent_dir: Path) -> bool:
    """Whether ``attempts/<run>/`` currently has an entry that cannot be adopted."""
    active = _non_dotted(revalidated_listdir(parent_dir))
    if not active:
        return False
    if len(active) > 1:
        return True
    status, _ = _inspect_entry(parent_dir / active[0])
    return status != "adoptable"


def _release_to_free(claimed: Path) -> None:
    free_path = claimed.with_name(f"free__{_fresh_token()}")
    try:
        os.rename(claimed, free_path)
    except OSError as exc:
        if not _is_missing_or_stale(exc):
            raise


def _release_claimed(claimed: Path, *, seeded: bool) -> None:
    """Release ``claimed`` on exit: remove if untouched seed, else rename to ``free__<tok>``."""
    if seeded:
        try:
            if not os.listdir(claimed):
                os.rmdir(claimed)
                with suppress(OSError):
                    claimed.parent.rmdir()
                return
        except OSError as exc:
            if _is_missing_or_stale(exc):
                with suppress(OSError):
                    claimed.parent.rmdir()
                return
    try:
        os.rename(claimed, claimed.with_name(f"free__{_fresh_token()}"))
    except OSError as exc:
        if _is_missing_or_stale(exc):
            with suppress(OSError):
                claimed.parent.rmdir()
            return
        raise


@contextmanager
def attempt_lock(
    parent_dir: Path,
    *,
    is_published: Callable[[], bool] | None = None,
    acquire_timeout_s: float = DEFAULT_ACQUIRE_TIMEOUT_S,
    acquire_poll_interval_s: float = DEFAULT_ACQUIRE_POLL_INTERVAL_S,
) -> Generator[Path | None]:
    """Claim exclusive ownership of the single working folder inside ``parent_dir``.

    Yields ``None`` without touching ``parent_dir`` if ``is_published()`` reports
    a published run already exists. Otherwise yields the claimed working folder
    path and releases it back to ``free__<fresh_tok>`` on clean exit unless it
    was published or deleted.
    """
    deadline = time.monotonic() + max(acquire_timeout_s, 0.0)
    next_wait_log_at = 0.0
    seed_race_retries = 0

    claimed: Path | None = None
    seeded = False

    while True:
        if is_published is not None and is_published():
            yield None
            return

        parent_dir.mkdir(parents=True, exist_ok=True)
        entries = revalidated_listdir(parent_dir)
        if entries is None:
            continue
        active = _non_dotted(entries)

        if not active:
            seed_path = parent_dir / "free__seed"
            try:
                seed_path.mkdir()
            except FileExistsError:
                pass
            except FileNotFoundError:
                continue
            candidate = "free__seed"
            seeded = True
        elif len(active) > 1:
            if "free__seed" in active and seed_race_retries < 3:
                seed_race_retries += 1
                time.sleep(0.02)
                continue
            raise LockError(
                f"multiple active entries in attempt directory {parent_dir}: "
                f"{', '.join(active)}; refusing to acquire"
            )
        else:
            candidate = active[0]
            seeded = False
            status, reason = _inspect_entry(parent_dir / candidate)
            if status == "refuse":
                raise LockError(reason)
            if status == "held":
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    raise LockError(reason)
                if now >= next_wait_log_at:
                    logger.info("waiting for attempt lock at %s (%s)", parent_dir, reason)
                    next_wait_log_at = now + DEFAULT_LOCK_WAIT_LOG_INTERVAL_S
                time.sleep(
                    min(
                        acquire_poll_interval_s,
                        remaining,
                        max(next_wait_log_at - now, 0.0),
                    )
                )
                continue

        my_name = _owner_folder_name()
        target = parent_dir / my_name
        try:
            os.rename(parent_dir / candidate, target)
        except OSError as exc:
            if exc.errno in (
                errno.ENOENT,
                errno.ESTALE,
                errno.EEXIST,
                errno.ENOTEMPTY,
            ):
                continue
            raise

        # Post-rename revalidated listdir verification: we must be the sole
        # non-dotted entry inside parent_dir.
        after = _non_dotted(revalidated_listdir(parent_dir))
        if after != [my_name]:
            if seeded:
                try:
                    os.rmdir(target)
                except OSError as exc:
                    if not _is_missing_or_stale(exc):
                        _release_to_free(target)
                time.sleep(random.uniform(0.005, 0.025))
                continue
            _release_to_free(target)
            raise LockError(
                f"attempt directory {parent_dir} has unexpected concurrent entries "
                f"after claim: {', '.join(after)}"
            )

        claimed = target
        break

    try:
        yield claimed
    finally:
        _release_claimed(claimed, seeded=seeded)
