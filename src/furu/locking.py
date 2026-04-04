from __future__ import annotations

import errno
import os
import socket
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from furu.metadata import LockClaim
from furu.utils import _nfs_safe_unique_name

# TODO: move these to the general config rather than having these be hard coded here, so that the user can override them. also consider if we can remove some of these/simplify
CLOCK_SLOP_S = 10
DEFAULT_LIFETIME_S = 35.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 5.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01


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


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
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


def _normalize_lock_paths(lock_paths: Iterable[Path]) -> list[Path]:
    normalized = sorted(
        {path.resolve(strict=False) for path in lock_paths},
        key=lambda path: str(path),
    )
    if not normalized:
        raise ValueError("lock_many() requires at least one lock path")
    return normalized


def _try_acquire_break_dir(*, lock_path: Path, lifetime_s: float) -> Path | None:
    break_dir = lock_path.with_name(f"{lock_path.name}.break")
    for _ in range(2):
        try:
            break_dir.mkdir()
        except FileExistsError:
            break_stat = _safe_stat(break_dir)
            if break_stat is None:
                continue
            if break_stat.st_mtime + lifetime_s + CLOCK_SLOP_S > time.time():
                return None
            _safe_rmdir_if_exists(break_dir)
            if _safe_stat(break_dir) is not None:
                return None
        else:
            return break_dir
    return None


@dataclass(frozen=True, slots=True)
class LockEntry:
    lock_path: Path
    claim_path: Path

    @classmethod
    def for_path(cls, path: Path) -> "LockEntry":
        lock_path = path.resolve(strict=False)
        claim_path = _nfs_safe_unique_name(lock_path, name="claim").resolve(
            strict=False
        )
        return cls(lock_path=lock_path, claim_path=claim_path)

    def write_claim(self, *, lifetime_s: float) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        claim = LockClaim(
            lock_path=self.lock_path,
            claim_path=self.claim_path,
            pid=os.getpid(),
            hostname=socket.gethostname(),
        )
        fd = os.open(self.claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(claim.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        _touch_future(self.claim_path, lifetime_s=lifetime_s)

    def is_owned(self) -> bool:
        lock_stat = _safe_stat(self.lock_path)
        claim_stat = _safe_stat(self.claim_path)
        return (
            lock_stat is not None
            and claim_stat is not None
            and os.path.samestat(lock_stat, claim_stat)
        )

    def try_acquire(self) -> bool:
        try:
            os.link(self.claim_path, self.lock_path)
        except FileExistsError:
            return False
        except OSError as exc:
            if _is_missing_or_stale(exc):
                return False
            raise

        if not self.is_owned():
            _safe_unlink_if_exists(self.lock_path)
            return False
        return True

    def try_break_stale(self, *, lifetime_s: float) -> bool:
        lock_stat = _safe_stat(self.lock_path)
        if lock_stat is None:
            return False
        if lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
            return False

        raw_claim = _safe_read_text(self.lock_path)
        if raw_claim is None:
            return False

        try:
            claim = LockClaim.model_validate_json(raw_claim)
        except ValidationError as exc:
            raise LockAcquireError(
                f"cannot safely break stale lock at {self.lock_path}: malformed "
                "lock claim"
            ) from exc

        if claim.lock_path != self.lock_path:
            raise LockAcquireError(
                f"cannot safely break stale lock at {self.lock_path}: lock claim "
                "describes a different lock path"
            )

        break_dir = _try_acquire_break_dir(
            lock_path=self.lock_path,
            lifetime_s=lifetime_s,
        )
        if break_dir is None:
            return False

        try:
            current_lock_stat = _safe_stat(self.lock_path)
            if current_lock_stat is None:
                return True
            if current_lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
                return False

            claim_stat = _safe_stat(claim.claim_path)
            if claim_stat is not None and not os.path.samestat(
                current_lock_stat, claim_stat
            ):
                raise StaleLockRaceError(
                    f"lock {self.lock_path} changed owners while breaking a stale lock"
                )

            _safe_unlink_if_exists(self.lock_path)
            _safe_unlink_if_exists(claim.claim_path)
            return True
        finally:
            _safe_rmdir_if_exists(break_dir)

    def release(self) -> None:
        if not self.is_owned():
            raise NotLockedError(f"lock at {self.lock_path} is no longer owned")
        try:
            self.lock_path.unlink()
        except OSError as exc:
            if _is_missing_or_stale(exc):
                raise NotLockedError(
                    f"lock at {self.lock_path} is no longer owned"
                ) from exc
            raise


@dataclass(frozen=True, slots=True)
class Lease:
    entries: tuple[LockEntry, ...]

    def held(self) -> bool:
        return all(entry.is_owned() for entry in self.entries)

    def assert_held(self, message: str) -> None:
        if not self.held():
            raise LockLostError(message)


def _release_subset(entries: Iterable[LockEntry]) -> None:
    for entry in entries:
        if entry.is_owned():
            entry.release()


def _release_all(entries: Iterable[LockEntry]) -> None:
    lost_lock = False
    for entry in entries:
        try:
            entry.release()
        except NotLockedError:
            lost_lock = True
    if lost_lock:
        raise NotLockedError("lock set is no longer fully owned")


def _acquire_retry_sleep_s(
    *,
    lock_path: Path,
    acquire_deadline_monotonic_s: float,
    acquire_poll_interval_s: float,
) -> float:
    remaining_wait_s = max(acquire_deadline_monotonic_s - time.monotonic(), 0.0)
    if remaining_wait_s == 0.0:
        return 0.0

    lock_stat = _safe_stat(lock_path)
    if lock_stat is None:
        return min(acquire_poll_interval_s, remaining_wait_s)

    remaining_until_stale_s = lock_stat.st_mtime + CLOCK_SLOP_S - time.time()
    if remaining_until_stale_s <= 0.0:
        return min(acquire_poll_interval_s, remaining_wait_s)

    return min(acquire_poll_interval_s, remaining_wait_s, remaining_until_stale_s)


def _heartbeat_loop(
    *,
    entries: tuple[LockEntry, ...],
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: threading.Event,
) -> None:
    if not all(entry.is_owned() for entry in entries):
        return
    for entry in entries:
        if not _try_touch_future(entry.claim_path, lifetime_s=lifetime_s):
            return

    next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    while True:
        timeout_s = max(next_heartbeat_at - time.monotonic(), 0.0)
        if stop_event.wait(timeout_s):
            return
        if not all(entry.is_owned() for entry in entries):
            return
        for entry in entries:
            if not _try_touch_future(entry.claim_path, lifetime_s=lifetime_s):
                return
        next_heartbeat_at = time.monotonic() + heartbeat_interval_s


def _lock_timeout_message(lock_paths: list[Path], *, acquire_timeout_s: float) -> str:
    if len(lock_paths) == 1:
        return (
            f"could not acquire lock at {lock_paths[0]} within "
            f"{acquire_timeout_s:g} seconds"
        )
    return (
        "could not acquire lock set within "
        f"{acquire_timeout_s:g} seconds: " + ", ".join(str(path) for path in lock_paths)
    )


def _lock_lost_message(lock_paths: list[Path]) -> str:
    if len(lock_paths) == 1:
        return f"lost lock at {lock_paths[0]}"
    return "lost lock for paths: " + ", ".join(str(path) for path in lock_paths)


@contextmanager
def lock_many(
    lock_paths: Iterable[Path],
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Iterator[Lease]:
    entries = tuple(
        LockEntry.for_path(path) for path in _normalize_lock_paths(lock_paths)
    )

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S,
            heartbeat_interval_s / 3,
        )

    body_error: BaseException | None = None
    try:
        for entry in entries:
            entry.write_claim(lifetime_s=lifetime_s)

        acquire_deadline_monotonic_s = time.monotonic() + max(acquire_timeout_s, 0.0)
        while True:
            acquired: list[LockEntry] = []
            blocked: LockEntry | None = None
            try:
                for entry in entries:
                    if entry.try_acquire():
                        acquired.append(entry)
                    else:
                        blocked = entry
                        break
            except BaseException:
                _release_subset(acquired)
                raise

            if blocked is None:
                break

            _release_subset(acquired)

            if blocked.try_break_stale(lifetime_s=lifetime_s):
                continue

            sleep_s = _acquire_retry_sleep_s(
                lock_path=blocked.lock_path,
                acquire_deadline_monotonic_s=acquire_deadline_monotonic_s,
                acquire_poll_interval_s=acquire_poll_interval_s,
            )
            if sleep_s <= 0.0:
                raise LockAcquireError(
                    _lock_timeout_message(
                        [entry.lock_path for entry in entries],
                        acquire_timeout_s=acquire_timeout_s,
                    )
                )
            time.sleep(sleep_s)

        lease = Lease(entries)
        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "entries": entries,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{entries[0].lock_path.name}",
            daemon=True,
        )
        heartbeat.start()

        try:
            yield lease
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            try:
                _release_all(entries)
            except NotLockedError:
                if body_error is None:
                    raise LockLostError(
                        _lock_lost_message([entry.lock_path for entry in entries])
                    )
    finally:
        for entry in entries:
            _safe_unlink_if_exists(entry.claim_path)


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Iterator[Lease]:
    with lock_many(
        [lock_path],
        lifetime_s=lifetime_s,
        heartbeat_interval_s=heartbeat_interval_s,
        acquire_timeout_s=acquire_timeout_s,
        acquire_poll_interval_s=acquire_poll_interval_s,
    ) as lease:
        yield lease
