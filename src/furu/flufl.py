import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


class NotLockedError(RuntimeError):
    pass


class TimeOutError(TimeoutError):
    pass


def _is_stale(*, lock_path: Path, lifetime_s: float) -> bool:
    try:
        age_s = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age_s > lifetime_s


def _read_owner_token(lock_path: Path) -> str:
    try:
        return lock_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise NotLockedError(f"lock {lock_path} does not exist") from exc


def _try_acquire_lock(*, lock_path: Path, owner_token: str, lifetime_s: float) -> bool:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _is_stale(lock_path=lock_path, lifetime_s=lifetime_s):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(owner_token)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return True


def _refresh_lock(*, lock_path: Path, owner_token: str) -> None:
    if _read_owner_token(lock_path) != owner_token:
        raise NotLockedError(f"lock {lock_path} is owned by another process")
    try:
        os.utime(lock_path)
    except FileNotFoundError as exc:
        raise NotLockedError(f"lock {lock_path} does not exist") from exc


def _release_lock(*, lock_path: Path, owner_token: str) -> None:
    if _read_owner_token(lock_path) != owner_token:
        raise NotLockedError(f"lock {lock_path} is owned by another process")
    try:
        lock_path.unlink()
    except FileNotFoundError as exc:
        raise NotLockedError(f"lock {lock_path} does not exist") from exc


@contextmanager
def lock(
    lock_path: str | Path,
    *,
    lifetime_s: float,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> Iterator[Callable[[], None]]:
    path = Path(lock_path)
    owner_token = f"{os.getpid()}-{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout_s

    while True:
        if _try_acquire_lock(
            lock_path=path,
            owner_token=owner_token,
            lifetime_s=lifetime_s,
        ):
            break
        if timeout_s <= 0 or time.monotonic() >= deadline:
            raise TimeOutError(f"timed out acquiring lock at {path}")
        time.sleep(poll_interval_s)

    def refresh() -> None:
        _refresh_lock(lock_path=path, owner_token=owner_token)

    try:
        yield refresh
    finally:
        _release_lock(lock_path=path, owner_token=owner_token)
