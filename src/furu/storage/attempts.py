"""Folder-name-based attempt locks over NFS.

Each run's attempt parent directory ``attempts/<run>/`` holds at most one
non-dotted working folder whose name encodes ownership:

- ``free__<tok>`` (seeded via ``free__seed`` when empty; released to ``free__<fresh_tok>``)
- ``slurm__<cluster>__<jobid>__<restart>__<tok>`` (raw ``$SLURM_JOB_ID``)
- ``local__<host>__<pid>__<tok>``

Dotted entries (such as ``.trash__<tok>`` or ``.deleting__<tok>``) are ignored.
"""

from __future__ import annotations

import errno
import os
import secrets
import socket
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from furu.code_trace import versions
from furu.config import CodeVersion
from furu.locking import LockError
from furu.logging import get_logger
from furu.storage._layout import FIXED, attempt_parent_in, attempts_dir_in, runs_dir_in

DEFAULT_ACQUIRE_TIMEOUT_S = 45.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 0.05
DEFAULT_LOCK_WAIT_LOG_INTERVAL_S = 10.0
SLURM_COMMAND_TIMEOUT_S = 60.0

# Clean terminal Slurm job states: NODE_FAIL and BOOT_FAIL are intentionally
# excluded because the node or task may still be unhealthy or running.
CLEAN_TERMINAL_SLURM_STATES = frozenset(
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

logger = get_logger()


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
    restart: str
    token: str


type _Owner = _FreeOwner | _LocalOwner | _SlurmOwner


@dataclass(slots=True)
class ClaimedAttempt:
    """An attempt working folder claimed by this process."""

    parent: Path
    path: Path
    name: str
    was_created: bool
    _gone: bool = False

    def is_held(self) -> bool:
        """Return True while this claimed folder name still exists in ``parent``."""
        if self._gone:
            return False
        return self.name in revalidated_listdir(self.parent)

    def mark_gone(self) -> None:
        """Mark the folder as intentionally moved/deleted (e.g. published or deleted)."""
        self._gone = True


def revalidated_listdir(path: Path, *, root: Path | None = None) -> list[str]:
    """List ``path``, refreshing parent directory entries to avoid NFS negative caches.

    Returns ``[]`` if ``path`` does not exist or is stale.
    """
    ancestors: list[Path] = []
    if root is not None:
        cur = path.parent
        while True:
            ancestors.append(cur)
            if cur == root or cur == cur.parent:
                break
            cur = cur.parent
        ancestors.reverse()
    elif path.parent != path:
        ancestors = [path.parent]

    for ancestor in ancestors:
        with suppress(OSError):
            os.listdir(ancestor)

    try:
        return os.listdir(path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            if path.parent != path:
                with suppress(OSError):
                    os.listdir(path.parent)
            try:
                return os.listdir(path)
            except OSError as retry_exc:
                if retry_exc.errno in (errno.ENOENT, errno.ESTALE):
                    return []
                raise
        raise


def _active_entries(attempt_parent: Path, *, root: Path | None = None) -> list[str]:
    return [
        name
        for name in revalidated_listdir(attempt_parent, root=root)
        if not name.startswith(".")
    ]


def resolve_attempt_dir(attempt_parent: Path) -> Path | None:
    """Resolve the single active working folder inside ``attempts/<run>/``, if any."""
    active = _active_entries(attempt_parent)
    if len(active) != 1:
        return None
    return attempt_parent / active[0]


def has_published_run(
    class_dir: Path, run: str, *, code_version: CodeVersion = "fixed"
) -> bool:
    """Check whether ``run`` is already published in a live code version via revalidated listing."""
    if code_version == "fixed":
        runs = runs_dir_in(class_dir / FIXED)
        return any(
            name == run for name in revalidated_listdir(runs, root=class_dir) if "." not in name
        )

    revalidated_listdir(class_dir, root=class_dir.parent)
    for name in revalidated_listdir(class_dir):
        if name not in ("attempts", "remembered", FIXED) and "." not in name:
            revalidated_listdir(class_dir / name)
    for version, why in versions(class_dir):
        if why is not None:
            continue
        runs = runs_dir_in(version)
        if any(
            entry == run
            for entry in revalidated_listdir(runs, root=version)
            if "." not in entry
        ):
            return True
    return False


def _clean_part(value: str) -> str:
    cleaned = value.replace("__", "-").replace("/", "-").strip()
    return cleaned or "unknown"


def _parse_owner(name: str) -> _Owner | None:
    parts = name.split("__")
    match parts:
        case ["free", token] if token:
            return _FreeOwner(token)
        case ["local", host, pid_str, token] if (
            host and pid_str.isdigit() and int(pid_str) > 0 and token
        ):
            return _LocalOwner(host, int(pid_str), token)
        case ["slurm", cluster, job_id, restart, token] if (
            cluster and job_id and restart and token
        ):
            return _SlurmOwner(cluster, job_id, restart, token)
        case _:
            return None


def _new_owner_name() -> str:
    token = secrets.token_hex(8)
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "").strip()
    if job_id and cluster:
        restart = os.environ.get("SLURM_RESTART_COUNT", "").strip() or "0"
        return (
            f"slurm__{_clean_part(cluster)}__{job_id}__{_clean_part(restart)}__{token}"
        )
    host = _clean_part(socket.gethostname())
    return f"local__{host}__{os.getpid()}__{token}"


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return True
    rparen = stat_text.rfind(")")
    return not (rparen != -1 and stat_text[rparen + 2 :].startswith("Z"))


def _run_slurm(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=SLURM_COMMAND_TIMEOUT_S,
    )


def _is_slurm_owner_adoptable(owner: _SlurmOwner) -> bool:
    """Conservative fail-closed 2-gate check for Slurm attempt ownership."""
    raw_cluster = os.environ.get("SLURM_CLUSTER_NAME", "").strip()
    if not raw_cluster or _clean_part(raw_cluster) != owner.cluster:
        return False

    try:
        sq = _run_slurm(
            "squeue", "--noheader", "--jobs", owner.job_id, "--array", "--format=%i %T"
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if sq.returncode != 0:
        if "Invalid job id specified" not in sq.stderr:
            return False
    elif sq.stdout.strip():
        return False

    try:
        sa = _run_slurm(
            "sacct",
            "-D",
            "-X",
            "--noheader",
            "--parsable2",
            "-o",
            "JobID,State,Reason",
            "-j",
            owner.job_id,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if sa.returncode != 0:
        return False

    rows = [line for line in sa.stdout.splitlines() if line.strip()]
    if not rows:
        return False
    for row in rows:
        if "kill task failed" in row.lower():
            return False
        parts = row.split("|")
        if len(parts) < 2:
            return False
        state = parts[1].strip().partition(" ")[0]
        if state not in CLEAN_TERMINAL_SLURM_STATES:
            return False
    return True


def _inspect_entry(
    name: str,
) -> tuple[Literal["adoptable", "held", "stranded_local", "malformed"], _Owner | None]:
    owner = _parse_owner(name)
    if owner is None:
        return "malformed", None
    match owner:
        case _FreeOwner():
            return "adoptable", owner
        case _LocalOwner(host=host, pid=pid):
            if host != _clean_part(socket.gethostname()):
                return "stranded_local", owner
            return ("held" if _is_pid_alive(pid) else "adoptable"), owner
        case _SlurmOwner():
            return ("adoptable" if _is_slurm_owner_adoptable(owner) else "held"), owner


def is_attempt_held(attempt_parent: Path) -> bool:
    """Return True when ``attempts/<run>/`` holds an active, unadoptable, or malformed entry."""
    active = _active_entries(attempt_parent)
    if not active:
        return False
    if len(active) > 1:
        return True
    status, _ = _inspect_entry(active[0])
    return status != "adoptable"


@contextmanager
def acquire_attempt(
    class_dir: Path,
    run: str,
    *,
    code_version: CodeVersion = "fixed",
    check_published: bool = True,
    acquire_timeout_s: float = DEFAULT_ACQUIRE_TIMEOUT_S,
    acquire_poll_interval_s: float = DEFAULT_ACQUIRE_POLL_INTERVAL_S,
) -> Generator[ClaimedAttempt | None]:
    """Claim the working folder inside ``attempts/<run>/`` and release to ``free__<tok>``.

    When ``check_published`` is True, yields ``None`` without creating a folder
    if a revalidated listing already finds ``run`` published in ``runs/``.
    """
    if check_published and has_published_run(
        class_dir, run, code_version=code_version
    ):
        yield None
        return

    attempts_root = attempts_dir_in(class_dir)
    parent = attempt_parent_in(class_dir, run)
    deadline = time.monotonic() + max(acquire_timeout_s, 0.0)
    next_wait_log_at = 0.0
    claimed: ClaimedAttempt | None = None

    while claimed is None:
        if check_published and has_published_run(
            class_dir, run, code_version=code_version
        ):
            yield None
            return

        parent.mkdir(parents=True, exist_ok=True)
        active = _active_entries(parent, root=attempts_root)
        if len(active) > 1:
            raise LockError(
                f"cannot acquire attempt in {parent}: multiple active entries "
                f"{sorted(active)}; expected at most one"
            )

        if not active:
            seed_path = parent / "free__seed"
            try:
                os.mkdir(seed_path)
            except OSError as exc:
                if exc.errno in (errno.EEXIST, errno.ENOENT, errno.ESTALE):
                    continue
                raise
            candidate = "free__seed"
        else:
            candidate = active[0]
            status, owner = _inspect_entry(candidate)
            if status == "malformed":
                raise LockError(
                    f"cannot acquire attempt in {parent}: malformed entry {parent / candidate}; "
                    f"expected free__<tok>, slurm__..., or local__... "
                    f"(remove or rename it to {parent / 'free__manual'})"
                )
            if status == "stranded_local":
                assert isinstance(owner, _LocalOwner)
                raise LockError(
                    f"attempt at {parent / candidate} is held by a local process on "
                    f"another host ({owner.host}, pid {owner.pid}; current host is "
                    f"{socket.gethostname()}). If that process is no longer running, "
                    f"rename {parent / candidate} to {parent / 'free__manual'} or remove it"
                )
            if status == "held":
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    raise LockError(
                        f"could not acquire attempt at {parent / candidate} within "
                        f"{acquire_timeout_s:g} seconds; if its owner is no longer "
                        f"running, rename {parent / candidate} to {parent / 'free__manual'} "
                        f"or remove it"
                    )
                if now >= next_wait_log_at:
                    logger.info("waiting for attempt at %s", parent / candidate)
                    next_wait_log_at = now + DEFAULT_LOCK_WAIT_LOG_INTERVAL_S
                time.sleep(
                    min(
                        acquire_poll_interval_s,
                        remaining,
                        max(next_wait_log_at - now, 0.0),
                    )
                )
                continue

        claimed_name = _new_owner_name()
        claimed_path = parent / claimed_name
        try:
            os.rename(parent / candidate, claimed_path)
        except OSError as exc:
            if exc.errno in (
                errno.ENOENT,
                errno.ESTALE,
                errno.EEXIST,
                errno.ENOTEMPTY,
            ):
                continue
            raise

        verified = _active_entries(parent, root=attempts_root)
        if claimed_name not in verified:
            continue
        if verified != [claimed_name]:
            raise LockError(
                f"cannot acquire attempt in {parent}: multiple active entries "
                f"after claim {sorted(verified)}"
            )
        was_created = candidate == "free__seed" and not any(claimed_path.iterdir())
        claimed = ClaimedAttempt(
            parent=parent,
            path=claimed_path,
            name=claimed_name,
            was_created=was_created,
        )

    body_error: BaseException | None = None
    try:
        yield claimed
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if not claimed._gone:
            if claimed.is_held():
                released = parent / f"free__{secrets.token_hex(8)}"
                try:
                    os.rename(claimed.path, released)
                except OSError as exc:
                    if exc.errno in (errno.ENOENT, errno.ESTALE):
                        if body_error is None:
                            raise LockError(f"lost lock at {claimed.path}") from exc
                    else:
                        raise
            elif body_error is None:
                raise LockError(f"lost lock at {claimed.path}")
