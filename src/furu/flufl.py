import errno
import os
import socket
import time
import uuid
from contextlib import suppress
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


CLOCK_SLOP_S = 10


class NotLockedError(RuntimeError):
    pass


class TimeOutError(TimeoutError):
    pass


def _claim_path_for(lock_path: Path) -> Path:
    return lock_path.with_name(
        f"{lock_path.name}.{socket.getfqdn()}.{os.getpid()}.{uuid.uuid4().hex}.claim"
    )


def _expiry_ts(lifetime_s: float) -> float:
    return time.time() + lifetime_s


def _touch_future(path: Path, *, lifetime_s: float) -> None:
    expiry = _expiry_ts(lifetime_s)
    os.utime(path, times=(expiry, expiry))


def _is_stale(path: Path) -> bool:
    try:
        return path.stat().st_mtime + CLOCK_SLOP_S <= time.time()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            return False
        raise


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


def _owner_claim_path_from_lock(lock_path: Path) -> Path:
    claim_path = _read_owner_claim_path(lock_path)
    if claim_path is None:
        raise NotLockedError(f"lock {lock_path} does not exist")
    return claim_path


def _write_claim_file(claim_path: Path) -> None:
    fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(claim_path))
        f.flush()
        os.fsync(f.fileno())


def _holds_lock(*, lock_path: Path, owner_claim_path: Path) -> bool:
    lock_stat = _stat(lock_path)
    owner_stat = _stat(owner_claim_path)
    if lock_stat is None or owner_stat is None:
        return False
    return os.path.samestat(lock_stat, owner_stat)


def _assert_is_owner(*, lock_path: Path, owner_claim_path: Path) -> None:
    if not _holds_lock(lock_path=lock_path, owner_claim_path=owner_claim_path):
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


def _safe_to_break(lock_path: Path) -> Path | None:
    claim_path = _read_owner_claim_path(lock_path)
    if claim_path is None:
        return None
    if not _is_related_claim_path(lock_path=lock_path, claim_path=claim_path):
        return None
    claim_contents = _read_path(claim_path)
    if claim_contents != str(claim_path):
        return None
    return claim_path


def _break_stale_lock(*, lock_path: Path, lifetime_s: float) -> None:
    if not _is_stale(lock_path):
        return

    owner_claim_path = _safe_to_break(lock_path)
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


def _refresh_lock(
    *, lock_path: Path, owner_claim_path: Path, lifetime_s: float
) -> None:
    _assert_is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path)
    try:
        _touch_future(owner_claim_path, lifetime_s=lifetime_s)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            raise NotLockedError(f"claim {owner_claim_path} does not exist") from exc
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


@contextmanager
def lock(
    lock_path: str | Path,
    *,
    lifetime_s: float,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> Iterator[Callable[[], None]]:
    path = Path(lock_path)
    owner_claim_path = _claim_path_for(path)
    deadline = time.monotonic() + timeout_s

    _write_claim_file(owner_claim_path)
    _touch_future(owner_claim_path, lifetime_s=lifetime_s)

    try:
        while True:
            if _try_acquire_lock(lock_path=path, owner_claim_path=owner_claim_path):
                _touch_future(owner_claim_path, lifetime_s=lifetime_s)
                break

            _break_stale_lock(lock_path=path, lifetime_s=lifetime_s)

            if timeout_s <= 0 or time.monotonic() >= deadline:
                raise TimeOutError(f"timed out acquiring lock at {path}")
            time.sleep(poll_interval_s)

        def refresh() -> None:
            _refresh_lock(
                lock_path=path,
                owner_claim_path=owner_claim_path,
                lifetime_s=lifetime_s,
            )

        try:
            yield refresh
        finally:
            _release_lock(lock_path=path, owner_claim_path=owner_claim_path)
    finally:
        _unlink_if_exists(owner_claim_path)
