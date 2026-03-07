import errno
import multiprocessing
import os
import socket
import time
import uuid
from contextlib import contextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Iterator

CLOCK_SLOP_S = 10  # NFS systems can be slow, so we add a 10 second safety margin TODO: assume this is included in the 120 second lease time?
DEFAULT_LIFETIME_S = 120.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01


class LockAcquireError(RuntimeError):
    pass


class NotLockedError(RuntimeError):
    pass


class LockLostError(RuntimeError):
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

    with suppress(PermissionError):
        if not _try_touch_future(lock_path, lifetime_s=lifetime_s):
            return

    _safe_unlink_if_exists(lock_path)
    _safe_unlink_if_exists(owner_claim_path)


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


class _LockHandle:
    def __init__(
        self,
        *,
        lock_path: Path,
        owner_claim_path: Path,
        heartbeat: multiprocessing.Process,
        stop_event: ProcessEvent,
        failure_rx: Connection,
    ):
        self._lock_path = lock_path
        self._owner_claim_path = owner_claim_path
        self._heartbeat = heartbeat
        self._stop_event = stop_event
        self._failure_rx = failure_rx
        self._failure: BaseException | None = None

    def __call__(self) -> bool:
        return self.held()

    def held(self) -> bool:
        owned = _is_owner(
            lock_path=self._lock_path, owner_claim_path=self._owner_claim_path
        )
        if not owned:
            self._record_failure(self._new_lock_lost_error())
        return owned

    def ensure(self) -> None:
        failure = self.failure()
        if failure is not None:
            raise failure
        if not _is_owner(
            lock_path=self._lock_path, owner_claim_path=self._owner_claim_path
        ):
            raise self._record_failure(self._new_lock_lost_error())

    def failure(self) -> BaseException | None:
        self._poll_failure()
        return self._failure

    def _new_lock_lost_error(self) -> LockLostError:
        return LockLostError(f"lost lock at {self._lock_path}")

    def _record_failure(self, failure: BaseException) -> BaseException:
        if self._failure is None:
            self._failure = failure
        return self._failure

    def _poll_failure(self) -> None:
        if self._failure is None:
            with suppress(EOFError, OSError):
                while self._failure_rx.poll():
                    self._record_failure(self._failure_rx.recv())
        if (
            self._failure is None
            and not self._stop_event.is_set()
            and not self._heartbeat.is_alive()
        ):
            exitcode = self._heartbeat.exitcode
            if exitcode not in (None, 0):
                self._record_failure(
                    RuntimeError(
                        f"heartbeat process exited unexpectedly for {self._lock_path}"
                    )
                )


def _heartbeat_loop(
    *,
    lock_path: Path,
    owner_claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: ProcessEvent,
    parent_pid: int,
    failure_tx: Connection,
) -> None:
    next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    try:
        while True:
            if os.getppid() != parent_pid:
                return
            if stop_event.wait(max(next_heartbeat_at - time.monotonic(), 0.0)):
                return
            if os.getppid() != parent_pid:
                return
            if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
                failure_tx.send(LockLostError(f"lost lock at {lock_path}"))
                return
            if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
                failure_tx.send(LockLostError(f"lost lock at {lock_path}"))
                return
            next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    except BaseException as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            failure_tx.send(exc)
    finally:
        failure_tx.close()


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
) -> Iterator[_LockHandle]:
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

        stop_event = multiprocessing.Event()
        failure_rx, failure_tx = multiprocessing.Pipe(duplex=False)
        heartbeat = multiprocessing.Process(
            target=_heartbeat_loop,
            kwargs={
                "lock_path": lock_path,
                "owner_claim_path": owner_claim_path,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
                "parent_pid": os.getpid(),
                "failure_tx": failure_tx,
            },
            name=f"lock-heartbeat:{lock_path.name}",
            daemon=True,
        )
        heartbeat.start()
        failure_tx.close()
        lock_handle = _LockHandle(
            lock_path=lock_path,
            owner_claim_path=owner_claim_path,
            heartbeat=heartbeat,
            stop_event=stop_event,
            failure_rx=failure_rx,
        )
        body_error: BaseException | None = None

        try:
            yield lock_handle
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            if heartbeat.is_alive():
                heartbeat.terminate()
                heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            failure = lock_handle.failure()
            failure_rx.close()
            try:
                _release_lock(lock_path=lock_path, owner_claim_path=owner_claim_path)
            except NotLockedError:
                failure = lock_handle._record_failure(
                    lock_handle._new_lock_lost_error()
                )
            if body_error is None:
                if failure is not None:
                    raise failure
    finally:
        _safe_unlink_if_exists(owner_claim_path)
