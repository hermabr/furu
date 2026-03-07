import errno
import os
import socket
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Callable, Iterator

CLOCK_SLOP_S = 10  # NFS systems can be slow, so we add a 10 second safety margin TODO: assume this is included in the 120 second lease time?


class NotLockedError(RuntimeError):
    pass


class TimeOutError(TimeoutError):
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


def _assert_is_owner(*, lock_path: Path, owner_claim_path: Path) -> None:
    lock_stat = _safe_stat(lock_path)
    owner_stat = _safe_stat(owner_claim_path)
    same_owner = (
        lock_stat is not None
        and owner_stat is not None
        and os.path.samestat(lock_stat, owner_stat)
    )
    if not same_owner:
        raise NotLockedError(f"lock {lock_path} is owned by another process")


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
    _assert_is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path)
    try:
        lock_path.unlink()
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise NotLockedError(f"lock {lock_path} does not exist") from exc
        if exc.errno != errno.ESTALE:
            raise
    _safe_unlink_if_exists(owner_claim_path)


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> Iterator[Callable[[], None]]:
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
        del timeout_s, poll_interval_s

        if not _try_acquire_lock(
            lock_path=lock_path, owner_claim_path=owner_claim_path
        ):
            _break_stale_lock(lock_path=lock_path, lifetime_s=lifetime_s)

            if not _try_acquire_lock(
                lock_path=lock_path, owner_claim_path=owner_claim_path
            ):
                raise TimeOutError(f"could not acquire lock at {lock_path}")

        _touch_future(owner_claim_path, lifetime_s=lifetime_s)

        def refresh() -> None:
            _assert_is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path)
            if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
                raise NotLockedError(f"claim {owner_claim_path} does not exist")

        try:
            yield refresh
        finally:
            _release_lock(lock_path=lock_path, owner_claim_path=owner_claim_path)
    finally:
        _safe_unlink_if_exists(owner_claim_path)
