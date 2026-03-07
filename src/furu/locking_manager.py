import errno
import multiprocessing as mp
import os
import signal
import socket
import time
import uuid
from contextlib import contextmanager, suppress
from enum import StrEnum
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Iterator

CLOCK_SLOP_S = 10
DEFAULT_HEARTBEAT_INTERVAL_S = 15


class _HeartbeatMessage(StrEnum): # TODO: should this be literal/do i need this at all?
    STOP = "stop"
    LOST_LOCK = "lost-lock"


class NotLockedError(RuntimeError):
    pass


class TimeOutError(TimeoutError):
    pass


def _touch_future(path: Path, *, lifetime_s: float) -> None:
    expiry = time.time() + lifetime_s
    os.utime(path, times=(expiry, expiry))


def _read_path(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return None
        raise


def _stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return None
        raise


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        if exc.errno not in (errno.ENOENT, errno.ESTALE):
            raise


def _is_related_claim_path(*, lock_path: Path, claim_path: Path) -> bool:
    return (
        claim_path.parent == lock_path.parent
        and claim_path.name.startswith(f"{lock_path.name}.")
        and claim_path.name.endswith(".claim")
    )


def _read_owner_claim_path(lock_path: Path) -> Path | None:
    claim_path_str = _read_path(lock_path)
    if claim_path_str is None:
        return None
    claim_path = Path(claim_path_str)
    if not claim_path.is_absolute():
        claim_path = lock_path.parent / claim_path
    return claim_path


def _assert_is_owner(*, lock_path: Path, owner_claim_path: Path) -> None:
    lock_stat = _stat(lock_path)
    owner_stat = _stat(owner_claim_path)
    if lock_stat is None or owner_stat is None:
        raise NotLockedError(f"lock {lock_path} is owned by another process")
    if not os.path.samestat(lock_stat, owner_stat):
        raise NotLockedError(f"lock {lock_path} is owned by another process")


def _try_acquire_lock(*, lock_path: Path, owner_claim_path: Path) -> bool:
    try:
        os.link(owner_claim_path, lock_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return False
        raise

    try:
        lock_stat = _stat(lock_path)
        if lock_stat is None:
            return False
        if lock_stat.st_nlink != 2:
            _unlink_if_exists(lock_path)
            return False
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return False
        raise
    return True


def _break_stale_lock(*, lock_path: Path, lifetime_s: float) -> None:
    def _safe_to_break() -> Path | None:
        claim_path = _read_owner_claim_path(lock_path)
        if claim_path is None:
            return None
        if not _is_related_claim_path(lock_path=lock_path, claim_path=claim_path):
            return None
        claim_contents = _read_path(claim_path)
        if claim_contents != str(claim_path):
            return None
        return claim_path

    def _is_stale() -> bool:
        try:
            return lock_path.stat().st_mtime + CLOCK_SLOP_S <= time.time()
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESTALE):
                return False
            raise

    if not _is_stale():
        return

    owner_claim_path = _safe_to_break()
    if owner_claim_path is None:
        return

    with suppress(PermissionError):
        try:
            _touch_future(lock_path, lifetime_s=lifetime_s)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ESTALE):
                raise
            return

    try:
        _unlink_if_exists(lock_path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return
        raise

    try:
        _unlink_if_exists(owner_claim_path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return
        raise


def _release_lock(*, lock_path: Path, owner_claim_path: Path) -> None:
    _assert_is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path)
    try:
        _unlink_if_exists(lock_path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            raise NotLockedError(f"lock {lock_path} does not exist") from exc
        raise
    _unlink_if_exists(owner_claim_path)


def _heartbeat_loop(
    *,
    lock_path: Path,
    owner_claim_path: Path,
    lifetime_s: float,
    interval_s: float,
    parent_pid: int,
    conn: Connection,
) -> None:
    try:
        while True:
            if conn.poll(interval_s):
                if conn.recv() == _HeartbeatMessage.STOP:
                    return

            if os.getppid() != parent_pid:
                return

            try:
                _assert_is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path)
                _touch_future(owner_claim_path, lifetime_s=lifetime_s)
            except (NotLockedError, OSError):
                with suppress(BrokenPipeError):
                    conn.send(_HeartbeatMessage.LOST_LOCK)
                return
    finally:
        conn.close()


def _start_heartbeat(
    *,
    lock_path: Path,
    owner_claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
) -> tuple[BaseProcess, Connection]:
    ctx = mp.get_context("spawn")
    effective_interval_s = min(heartbeat_interval_s, max(lifetime_s / 3, 0.01))
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_heartbeat_loop,
        kwargs={ # TODO: make this a dataclass?
            "lock_path": lock_path,
            "owner_claim_path": owner_claim_path,
            "lifetime_s": lifetime_s,
            "interval_s": effective_interval_s,
            "parent_pid": os.getpid(),
            "conn": child_conn,
        },
        daemon=True,
    )
    process.start()
    child_conn.close()
    return process, parent_conn


def _raise_if_heartbeat_failed(conn: Connection) -> None:
    if not conn.poll():
        return

    try:
        message = conn.recv()
    except EOFError:
        return

    if message == _HeartbeatMessage.LOST_LOCK:
        raise NotLockedError("lock heartbeat lost ownership")


def _stop_heartbeat(
    *,
    process: BaseProcess,
    conn: Connection,
    timeout_s: float,
) -> NotLockedError | None:
    error: NotLockedError | None = None

    with suppress(BrokenPipeError, EOFError, OSError):
        conn.send(_HeartbeatMessage.STOP)

    process.join(timeout=timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout_s)

    if process.exitcode not in (0, -signal.SIGTERM):
        error = NotLockedError("lock heartbeat exited unexpectedly")

    if error is None and process.exitcode == 0:
        try:
            _raise_if_heartbeat_failed(conn)
        except NotLockedError as exc:
            error = exc

    conn.close()

    return error


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float,
    timeout_s: float,
    poll_interval_s: float = 0.05,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
) -> Iterator[None]:
    owner_claim_path = lock_path.with_name(
        f"{lock_path.name}.{socket.getfqdn()}.{os.getpid()}.{uuid.uuid4().hex}.claim"
    )
    deadline = time.monotonic() + timeout_s

    fd = os.open(owner_claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(owner_claim_path))
        f.flush()
        os.fsync(f.fileno())

    _touch_future(owner_claim_path, lifetime_s=lifetime_s)

    heartbeat_process: BaseProcess | None = None
    heartbeat_conn: Connection | None = None
    body_failed = True

    try:
        while True:  # TODO: do i want/need this while loop?
            if _try_acquire_lock(
                lock_path=lock_path, owner_claim_path=owner_claim_path
            ):
                _touch_future(owner_claim_path, lifetime_s=lifetime_s)
                break

            _break_stale_lock(lock_path=lock_path, lifetime_s=lifetime_s)

            if timeout_s <= 0 or time.monotonic() >= deadline:
                raise TimeOutError(f"timed out acquiring lock at {lock_path}")
            time.sleep(poll_interval_s)  # TODO: do i want a poll interval here?

        heartbeat_process, heartbeat_conn = _start_heartbeat(
            lock_path=lock_path,
            owner_claim_path=owner_claim_path,
            lifetime_s=lifetime_s,
            heartbeat_interval_s=heartbeat_interval_s,
        )

        try:
            yield
            body_failed = False
        finally:
            heartbeat_error: NotLockedError | None = None
            if heartbeat_process is not None and heartbeat_conn is not None:
                heartbeat_error = _stop_heartbeat(
                    process=heartbeat_process,
                    conn=heartbeat_conn,
                    timeout_s=max(poll_interval_s, 0.05),
                )
                heartbeat_process = None
                heartbeat_conn = None

            release_error: Exception | None = None
            try:
                _release_lock(lock_path=lock_path, owner_claim_path=owner_claim_path)
            except Exception as exc:
                release_error = exc

            if not body_failed:
                if heartbeat_error is not None:
                    raise heartbeat_error
                if release_error is not None:
                    raise release_error
    finally:
        if heartbeat_process is not None and heartbeat_conn is not None:
            _stop_heartbeat(
                process=heartbeat_process,
                conn=heartbeat_conn,
                timeout_s=max(poll_interval_s, 0.05),
            )
        _unlink_if_exists(owner_claim_path)
