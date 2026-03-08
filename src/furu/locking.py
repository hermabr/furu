import errno
import multiprocessing
import os
import signal
import socket
import time
import uuid
from contextlib import contextmanager, suppress
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Callable, Iterator

# TODO: move these to the general config rather than having these be hard coded here, so that the user can override them. also consider if we can remove some of these/simplify
CLOCK_SLOP_S = 10  # NFS systems can be slow, so we add a 10 second safety margin TODO: assume this is included in the 120 second lease time?
DEFAULT_LIFETIME_S = 120.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01
CHILD_SIGNAL_POLL_INTERVAL_S = 0.1

# TODO: currently, we don't handle SIGTERM/SIGINT in the parent. we need to add this (but this needs to be integrated with the general furu handling of SIGINT/SIGTERM)


class LockAcquireError(RuntimeError):
    pass


class NotLockedError(RuntimeError):
    pass


class LockLostError(RuntimeError):
    pass


class StaleLockRaceError(LockAcquireError):
    pass


def _touch_future(path: Path, *, lifetime_s: float) -> None:
    expiry = time.time() + lifetime_s
    os.utime(path, times=(expiry, expiry))


def _is_missing_or_stale(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


def _try_touch_future(path: Path, *, lifetime_s: float) -> bool:
    try:
        _touch_future(path, lifetime_s=lifetime_s)
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return False
        raise
    return True


def _safe_read_path(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


def _safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


def _safe_unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        if not _is_missing_or_stale(exc):
            raise


def _safe_rmdir_if_exists(path: Path) -> None:
    try:
        path.rmdir()
    except OSError as exc:
        if not _is_missing_or_stale(exc) and exc.errno != errno.ENOTEMPTY:
            raise


def _try_acquire_stale_break_dir(*, lock_path: Path, lifetime_s: float) -> Path | None:
    break_dir = lock_path.with_name(f"{lock_path.name}.break")
    for _ in range(2):
        try:
            break_dir.mkdir()
        except FileExistsError:
            break_stat = _safe_stat(break_dir)
            if (
                break_stat is None
                or break_stat.st_mtime + lifetime_s + CLOCK_SLOP_S > time.time()
            ):
                return None
            _safe_rmdir_if_exists(break_dir)
            if _safe_stat(break_dir) is not None:
                return None
            continue
        return break_dir
    return None


def _safe_read_breakable_claim_path(lock_path: Path) -> Path | None:
    claim_path_str = _safe_read_path(lock_path)
    if claim_path_str is None:
        return None

    claim_path = Path(claim_path_str)
    if not claim_path.is_absolute():
        claim_path = lock_path.parent / claim_path

    if not (
        claim_path.parent == lock_path.parent
        and claim_path.name.startswith(f"{lock_path.name}.")
        and claim_path.name.endswith(".claim")
    ):
        return None

    if _safe_read_path(claim_path) != str(claim_path):
        return None

    return claim_path


def _is_owner(*, lock_path: Path, owner_claim_path: Path) -> bool:
    lock_stat = _safe_stat(lock_path)
    owner_stat = _safe_stat(owner_claim_path)
    return (
        lock_stat is not None
        and owner_stat is not None
        and os.path.samestat(lock_stat, owner_stat)
    )


def _try_acquire_lock(*, lock_path: Path, owner_claim_path: Path) -> bool:
    try:
        os.link(owner_claim_path, lock_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return False
        raise

    lock_stat = _safe_stat(lock_path)
    if lock_stat is None or lock_stat.st_nlink != 2:
        _safe_unlink_if_exists(lock_path)
        return False

    return True


def _break_stale_lock(*, lock_path: Path, lifetime_s: float) -> None:
    lock_stat = _safe_stat(lock_path)
    if lock_stat is None or lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
        return

    owner_claim_path = _safe_read_breakable_claim_path(lock_path)
    if owner_claim_path is None:
        return

    break_dir = _try_acquire_stale_break_dir(lock_path=lock_path, lifetime_s=lifetime_s)
    if break_dir is None:
        return

    try:
        current_lock_stat = _safe_stat(lock_path)
        if current_lock_stat is None:
            return

        if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
            raise StaleLockRaceError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        if current_lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
            return

        _safe_unlink_if_exists(lock_path)
        _safe_unlink_if_exists(owner_claim_path)
    finally:
        _safe_rmdir_if_exists(break_dir)


def _release_lock(*, lock_path: Path, owner_claim_path: Path) -> None:
    if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
        raise NotLockedError(f"lock {lock_path} is owned by another process")

    try:
        lock_path.unlink()
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise NotLockedError(f"lock {lock_path} does not exist") from exc
        if exc.errno != errno.ESTALE:
            raise
    _safe_unlink_if_exists(owner_claim_path)


def _parent_is_alive(*, parent_pid: int) -> bool:
    if os.getppid() == parent_pid:
        return True

    try:
        os.kill(parent_pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _try_release_if_parent_dead(
    *, lock_path: Path, owner_claim_path: Path, parent_pid: int
) -> bool:
    if _parent_is_alive(parent_pid=parent_pid):
        return False

    with suppress(NotLockedError, OSError):
        _release_lock(lock_path=lock_path, owner_claim_path=owner_claim_path)
    with suppress(OSError):
        _safe_unlink_if_exists(owner_claim_path)
    return True


def _heartbeat_loop(
    *,
    lock_path: Path,
    owner_claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: ProcessEvent,
    parent_pid: int,
) -> None:
    def handle_shutdown_signal(_signum: int, _frame: object | None) -> None:
        _try_touch_future(owner_claim_path, lifetime_s=lifetime_s + CLOCK_SLOP_S)
        if not _parent_is_alive(parent_pid=parent_pid):
            raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    while True:
        if not _parent_is_alive(parent_pid=parent_pid):
            return
        timeout_s = min(
            max(next_heartbeat_at - time.monotonic(), 0.0),
            CHILD_SIGNAL_POLL_INTERVAL_S,
        )
        if stop_event.wait(timeout_s):
            return  # TODO: should i mark/log this in any way?
        if time.monotonic() < next_heartbeat_at:
            continue
        if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
            return  # TODO: should i mark/log this in any way?
        if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
            return  # TODO: should i mark/log this in any way?
        next_heartbeat_at = time.monotonic() + heartbeat_interval_s


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
) -> Iterator[Callable[[], bool]]:
    owner_claim_path = lock_path.with_name(
        f"{lock_path.name}.{socket.getfqdn()}.{os.getpid()}.{uuid.uuid4().hex}.claim"
    )

    fd = os.open(owner_claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(owner_claim_path))
        f.flush()
        os.fsync(f.fileno())

    _touch_future(owner_claim_path, lifetime_s=lifetime_s)

    try:
        if not _try_acquire_lock(
            lock_path=lock_path, owner_claim_path=owner_claim_path
        ):
            _break_stale_lock(lock_path=lock_path, lifetime_s=lifetime_s)
            if not _try_acquire_lock(
                lock_path=lock_path, owner_claim_path=owner_claim_path
            ):
                raise LockAcquireError(f"could not acquire lock at {lock_path}")

        _touch_future(owner_claim_path, lifetime_s=lifetime_s)

        multiprocessing_context = multiprocessing.get_context("spawn")
        stop_event = multiprocessing_context.Event()
        heartbeat = multiprocessing_context.Process(
            target=_heartbeat_loop,
            kwargs={
                "lock_path": lock_path,
                "owner_claim_path": owner_claim_path,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
                "parent_pid": os.getpid(),
            },
            name=f"lock-heartbeat:{lock_path.name}",
            daemon=True,
        )
        heartbeat.start()
        body_error: BaseException | None = None

        def has_lock() -> bool:
            return _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path)

        try:
            yield has_lock
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            if heartbeat.is_alive():
                heartbeat.terminate()
                heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            try:
                _release_lock(lock_path=lock_path, owner_claim_path=owner_claim_path)
            except NotLockedError:
                if body_error is None:
                    raise LockLostError(f"lost lock at {lock_path}")
    finally:
        _safe_unlink_if_exists(owner_claim_path)
