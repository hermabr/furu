from __future__ import annotations

import errno
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from furu.utils import _nfs_safe_unique_name

CLOCK_SLOP_S = 10
DEFAULT_LIFETIME_S = 35.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 5.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01


class LockAcquireError(RuntimeError):
    pass


class LockLostError(RuntimeError):
    pass


class StaleLockRaceError(LockAcquireError):
    pass


class LockManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1] = 1
    claim_path: Path
    lock_paths: tuple[Path, ...]

    @field_validator("claim_path")
    @classmethod
    def _claim_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("claim_path must be absolute")
        return value

    @field_validator("lock_paths")
    @classmethod
    def _lock_paths_must_be_nonempty_abs_unique(
        cls,
        value: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        if not value:
            raise ValueError("lock_paths may not be empty")
        if len(value) != len(set(value)):
            raise ValueError("lock_paths must be unique")
        if any(not path.is_absolute() for path in value):
            raise ValueError("lock_paths must be absolute")
        return value

    @classmethod
    def read(cls, lock_path: Path) -> LockManifest | None:
        raw_manifest = _safe_read_text(lock_path)
        if raw_manifest is None:
            return None

        try:
            manifest = cls.model_validate_json(raw_manifest)
        except ValidationError as exc:
            raise LockAcquireError(
                f"cannot safely break stale lock at {lock_path}: malformed lock manifest"
            ) from exc

        if lock_path not in manifest.lock_paths:
            raise LockAcquireError(
                f"cannot safely break stale lock at {lock_path}: manifest does not "
                "include contested lock path"
            )
        return manifest


def _is_missing_or_stale(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return os.path.samestat(left, right)


def _touch_future(path: Path, *, lifetime_s: float) -> None:
    expiry = time.time() + lifetime_s
    os.utime(path, times=(expiry, expiry))


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


def _normalize_lock_paths(lock_paths: Iterable[Path]) -> tuple[Path, ...]:
    resolved_paths: set[Path] = set()
    for path in lock_paths:
        resolved_paths.add(path.resolve(strict=False))

    normalized = tuple(sorted(resolved_paths, key=lambda path: str(path)))
    if not normalized:
        raise ValueError("lock_many() requires at least one lock path")
    return normalized


def _assert_same_filesystem(lock_paths: tuple[Path, ...]) -> None:
    first_device = lock_paths[0].parent.stat().st_dev
    for lock_path in lock_paths[1:]:
        if lock_path.parent.stat().st_dev != first_device:
            raise LockAcquireError(
                "hardlink-based locking requires every lock path to be on the same "
                "filesystem device"
            )


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


def _try_acquire_stale_break_dir(*, claim_path: Path, lifetime_s: float) -> Path | None:
    break_dir = Path(f"{claim_path}.break")
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


@dataclass(slots=True, frozen=True)
class _Lease:
    manifest: LockManifest

    @property
    def claim_path(self) -> Path:
        return self.manifest.claim_path

    @property
    def lock_paths(self) -> tuple[Path, ...]:
        return self.manifest.lock_paths

    @classmethod
    def create(cls, lock_paths: Iterable[Path], *, lifetime_s: float) -> _Lease:
        normalized_lock_paths = _normalize_lock_paths(lock_paths)
        _assert_same_filesystem(normalized_lock_paths)

        claim_path = _nfs_safe_unique_name(
            normalized_lock_paths[0],
            name="claim",
        ).resolve(strict=False)
        manifest = LockManifest(
            claim_path=claim_path,
            lock_paths=normalized_lock_paths,
        )

        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        _touch_future(claim_path, lifetime_s=lifetime_s)
        return cls(manifest=manifest)

    def try_acquire(self, lock_path: Path) -> bool:
        try:
            os.link(self.claim_path, lock_path)
        except FileExistsError:
            return False
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise LockAcquireError(
                    f"hardlink-based locking cannot link {self.claim_path} to {lock_path}"
                ) from exc
            if _is_missing_or_stale(exc):
                return False
            raise

        claim_stat = _safe_stat(self.claim_path)
        lock_stat = _safe_stat(lock_path)
        if (
            claim_stat is None
            or lock_stat is None
            or not _same_inode(lock_stat, claim_stat)
        ):
            _safe_unlink_if_exists(lock_path)
            return False
        return True

    def has_lock(self) -> bool:
        claim_stat = _safe_stat(self.claim_path)
        if claim_stat is None:
            return False

        for lock_path in self.lock_paths:
            lock_stat = _safe_stat(lock_path)
            if lock_stat is None or not _same_inode(lock_stat, claim_stat):
                return False
        return True

    def release_subset(self, lock_paths: Iterable[Path]) -> None:
        claim_stat = _safe_stat(self.claim_path)
        if claim_stat is None:
            return

        for lock_path in lock_paths:
            lock_stat = _safe_stat(lock_path)
            if lock_stat is None or not _same_inode(lock_stat, claim_stat):
                continue
            _safe_unlink_if_exists(lock_path)

    def release(self) -> bool:
        claim_stat = _safe_stat(self.claim_path)
        owned_at_release = claim_stat is not None

        for lock_path in self.lock_paths:
            lock_stat = _safe_stat(lock_path)
            if (
                claim_stat is None
                or lock_stat is None
                or not _same_inode(lock_stat, claim_stat)
            ):
                owned_at_release = False
                continue
            try:
                lock_path.unlink()
            except OSError as exc:
                if _is_missing_or_stale(exc):
                    owned_at_release = False
                    continue
                raise

        if claim_stat is not None:
            try:
                self.claim_path.unlink()
            except OSError as exc:
                if _is_missing_or_stale(exc):
                    owned_at_release = False
                else:
                    raise

        return owned_at_release

    def heartbeat_loop(
        self,
        *,
        lifetime_s: float,
        heartbeat_interval_s: float,
        stop_event: threading.Event,
    ) -> None:
        if not self.has_lock():
            return
        if not _try_touch_future(self.claim_path, lifetime_s=lifetime_s):
            return

        next_heartbeat_at = time.monotonic() + heartbeat_interval_s
        while True:
            timeout_s = max(next_heartbeat_at - time.monotonic(), 0.0)
            if stop_event.wait(timeout_s):
                return
            if not self.has_lock():
                return
            if not _try_touch_future(self.claim_path, lifetime_s=lifetime_s):
                return
            next_heartbeat_at = time.monotonic() + heartbeat_interval_s

    @classmethod
    def break_stale(cls, lock_path: Path, *, lifetime_s: float) -> bool:
        lock_stat = _safe_stat(lock_path)
        if lock_stat is None or lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
            return False

        manifest = LockManifest.read(lock_path)
        if manifest is None:
            return False

        break_dir = _try_acquire_stale_break_dir(
            claim_path=manifest.claim_path,
            lifetime_s=lifetime_s,
        )
        if break_dir is None:
            return False

        try:
            current_lock_stat = _safe_stat(lock_path)
            if current_lock_stat is None:
                return True

            if current_lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
                return False

            claim_stat = _safe_stat(manifest.claim_path)
            if claim_stat is None:
                raise LockAcquireError(
                    f"cannot safely break stale lock at {lock_path}: missing claim file "
                    f"{manifest.claim_path}"
                )

            if not _same_inode(current_lock_stat, claim_stat):
                raise StaleLockRaceError(
                    f"lock {lock_path} changed owners while breaking a stale lock"
                )

            for member_lock_path in manifest.lock_paths:
                member_lock_stat = _safe_stat(member_lock_path)
                if member_lock_stat is not None and _same_inode(
                    member_lock_stat, claim_stat
                ):
                    _safe_unlink_if_exists(member_lock_path)

            _safe_unlink_if_exists(manifest.claim_path)
            return True
        finally:
            _safe_rmdir_if_exists(break_dir)


@contextmanager
def lock_many(
    lock_paths: Iterable[Path],
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Iterator[Callable[[], bool]]:
    lease = _Lease.create(lock_paths, lifetime_s=lifetime_s)

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S,
            heartbeat_interval_s / 3,
        )

    try:
        acquire_deadline_monotonic_s = time.monotonic() + max(acquire_timeout_s, 0.0)
        while True:
            acquired_paths: list[Path] = []
            blocked_path: Path | None = None
            try:
                for lock_path in lease.lock_paths:
                    if lease.try_acquire(lock_path):
                        acquired_paths.append(lock_path)
                    else:
                        blocked_path = lock_path
                        break
            except BaseException:
                lease.release_subset(acquired_paths)
                raise

            if blocked_path is None:
                break

            try:
                broke_stale = _Lease.break_stale(blocked_path, lifetime_s=lifetime_s)
            finally:
                lease.release_subset(acquired_paths)

            if broke_stale:
                continue

            sleep_s = _acquire_retry_sleep_s(
                lock_path=blocked_path,
                acquire_deadline_monotonic_s=acquire_deadline_monotonic_s,
                acquire_poll_interval_s=acquire_poll_interval_s,
            )
            if sleep_s <= 0.0:
                if len(lease.lock_paths) == 1:
                    raise LockAcquireError(
                        f"could not acquire lock at {lease.lock_paths[0]} within "
                        f"{acquire_timeout_s:g} seconds"
                    )
                raise LockAcquireError(
                    "could not acquire lock set within "
                    f"{acquire_timeout_s:g} seconds: "
                    + ", ".join(str(path) for path in lease.lock_paths)
                )
            time.sleep(sleep_s)

        _touch_future(lease.claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=lease.heartbeat_loop,
            kwargs={
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{lease.lock_paths[0].name}",
            daemon=True,
        )
        heartbeat.start()

        body_error: BaseException | None = None
        try:
            yield lease.has_lock
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            owned_at_release = lease.release()
            if body_error is None and not owned_at_release:
                if len(lease.lock_paths) == 1:
                    raise LockLostError(f"lost lock at {lease.lock_paths[0]}")
                raise LockLostError(
                    "lost lock for paths: "
                    + ", ".join(str(path) for path in lease.lock_paths)
                )
    finally:
        _safe_unlink_if_exists(lease.claim_path)


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Iterator[Callable[[], bool]]:
    with lock_many(
        [lock_path],
        lifetime_s=lifetime_s,
        heartbeat_interval_s=heartbeat_interval_s,
        acquire_timeout_s=acquire_timeout_s,
        acquire_poll_interval_s=acquire_poll_interval_s,
    ) as has_lock:
        yield has_lock
