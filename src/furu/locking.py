from __future__ import annotations

import errno
import os
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from furu.logging import get_logger
from furu.utils import nfs_safe_unique_name

# TODO: move these to the general config rather than having these be hard coded here, so that the user can override them. also consider if we can remove some of these/simplify
CLOCK_SLOP_S = 10
DEFAULT_LIFETIME_S = 35.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 0.05
DEFAULT_LOCK_WAIT_LOG_INTERVAL_S = 10.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01

logger = get_logger()


class LockError(RuntimeError):
    """A furu lock could not be acquired, was lost, or is inconsistent on disk."""


class LockManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim_path: Path
    lock_path: Path

    @field_validator("claim_path", "lock_path")
    @classmethod
    def _validate_absolute(cls, path: Path) -> Path:
        if not path.is_absolute():
            raise ValueError("lock manifest paths must be absolute")
        return path


def _is_missing_or_stale(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return os.path.samestat(left, right)


def _is_stale(stat_result: os.stat_result) -> bool:
    return stat_result.st_mtime + CLOCK_SLOP_S <= time.time()


def _write_claim_file(manifest: LockManifest, *, lifetime_s: float) -> None:
    fd = os.open(manifest.claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
        f.flush()
        os.fsync(f.fileno())
    touch_future(manifest.claim_path, lifetime_s=lifetime_s)


def touch_future(path: Path, *, lifetime_s: float) -> None:
    expiry = time.time() + lifetime_s
    os.utime(path, times=(expiry, expiry))


def stat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return None
        raise


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        if not _is_missing_or_stale(exc):
            raise


def read_manifest(path: Path) -> LockManifest | None:
    raw = read_text_or_none(path)
    if raw is None:
        return None

    contested_path = path.resolve(strict=False)
    try:
        manifest = LockManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise LockError(
            f"cannot safely break stale lock at {path}: malformed lock manifest"
        ) from exc

    if contested_path != manifest.lock_path:
        raise LockError(
            f"cannot safely break stale lock at {path}: manifest names another lock path"
        )
    return manifest


def owns(lock_path: Path, *, claim_path: Path) -> bool:
    claim_stat = stat_or_none(claim_path)
    lock_stat = stat_or_none(lock_path)
    return (
        claim_stat is not None
        and lock_stat is not None
        and _same_inode(lock_stat, claim_stat)
    )


def try_link(*, lock_path: Path, claim_path: Path) -> bool:
    try:
        os.link(claim_path, lock_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise LockError(
                f"hardlink-based locking cannot link {claim_path} to {lock_path}"
            ) from exc
        if _is_missing_or_stale(exc):
            return False
        raise

    if not owns(lock_path, claim_path=claim_path):
        unlink_if_exists(lock_path)
        return False
    return True


def _unlink_if_same(lock_path: Path, *, reference_stat: os.stat_result) -> bool:
    """Remove the lock if it is still the reference inode; False when it was lost."""
    lock_stat = stat_or_none(lock_path)
    if lock_stat is None or not _same_inode(lock_stat, reference_stat):
        return False
    try:
        lock_path.unlink()
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return False
        raise
    return True


def release(lock_path: Path, *, claim_path: Path, owner_stat: os.stat_result) -> bool:
    """Remove the lock and its claim; True when the lock was lost before that."""
    claim_stat = stat_or_none(claim_path)
    removed = _unlink_if_same(lock_path, reference_stat=claim_stat or owner_stat)
    unlink_if_exists(claim_path)
    return claim_stat is None or not removed


def acquire_break_file(*, claim_path: Path, lifetime_s: float) -> Path | None:
    break_path = Path(f"{claim_path}.break")
    for _ in range(2):
        try:
            fd = os.open(
                break_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            break_stat = stat_or_none(break_path)
            if (
                break_stat is not None
                and break_stat.st_mtime + lifetime_s + CLOCK_SLOP_S > time.time()
            ):
                return None
            unlink_if_exists(break_path)
            if stat_or_none(break_path) is not None:
                return None
            continue
        else:
            os.close(fd)
            return break_path
    return None


def break_stale(lock_path: Path, *, lifetime_s: float) -> bool:
    lock_stat = stat_or_none(lock_path)
    if lock_stat is None or not _is_stale(lock_stat):
        return False

    manifest = read_manifest(lock_path)
    if manifest is None:
        return False

    break_path = acquire_break_file(
        claim_path=manifest.claim_path,
        lifetime_s=lifetime_s,
    )
    if break_path is None:
        return False

    try:
        current_lock_stat = stat_or_none(lock_path)
        if current_lock_stat is None:
            return True
        if not _is_stale(current_lock_stat):
            return False

        claim_stat = stat_or_none(manifest.claim_path)
        reference_stat = claim_stat or current_lock_stat
        if claim_stat is not None and not _same_inode(current_lock_stat, claim_stat):
            raise LockError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        _unlink_if_same(lock_path, reference_stat=reference_stat)
        unlink_if_exists(manifest.claim_path)
        return True
    finally:
        unlink_if_exists(break_path)


def is_active_lock(lock_path: Path) -> bool:
    lock_stat = stat_or_none(lock_path)
    if lock_stat is None or _is_stale(lock_stat):
        return False

    try:
        manifest = read_manifest(lock_path)
    except LockError:
        return False
    if manifest is None:
        return False

    return owns(lock_path, claim_path=manifest.claim_path)


def _try_touch_future(path: Path, *, lifetime_s: float) -> bool:
    try:
        touch_future(path, lifetime_s=lifetime_s)
    except OSError as exc:
        if _is_missing_or_stale(exc):
            return False
        raise
    return True


def heartbeat_loop(
    *,
    lock_path: Path,
    claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: threading.Event,
) -> None:
    if not owns(lock_path, claim_path=claim_path):
        return
    if not _try_touch_future(claim_path, lifetime_s=lifetime_s):
        return

    while not stop_event.wait(heartbeat_interval_s):
        if not owns(lock_path, claim_path=claim_path):
            return
        if not _try_touch_future(claim_path, lifetime_s=lifetime_s):
            return


@contextmanager
def lock(
    lock_path: Path,
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Generator[Callable[[], bool]]:
    lock_path = lock_path.resolve()

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S,
            heartbeat_interval_s / 3,
        )

    claim_path = nfs_safe_unique_name(lock_path, name="claim").resolve(strict=False)
    manifest = LockManifest(claim_path=claim_path, lock_path=lock_path)
    _write_claim_file(manifest, lifetime_s=lifetime_s)
    owner_stat = claim_path.stat()

    try:
        deadline = time.monotonic() + max(acquire_timeout_s, 0.0)
        next_wait_log_at = 0.0

        while not try_link(lock_path=lock_path, claim_path=claim_path):
            if break_stale(lock_path, lifetime_s=lifetime_s):
                logger.info("broke stale lock at %s", lock_path)
                continue

            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise LockError(
                    f"could not acquire lock at {lock_path} within {acquire_timeout_s:g} seconds"
                )
            if now >= next_wait_log_at:
                logger.info("waiting for lock at %s", lock_path)
                next_wait_log_at = now + DEFAULT_LOCK_WAIT_LOG_INTERVAL_S
            time.sleep(
                min(
                    acquire_poll_interval_s,
                    remaining,
                    max(next_wait_log_at - now, 0.0),
                )
            )

        touch_future(claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=heartbeat_loop,
            kwargs={
                "lock_path": lock_path,
                "claim_path": claim_path,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{lock_path.name}",
            daemon=True,
        )
        heartbeat.start()
        body_error: BaseException | None = None

        def has_lock() -> bool:
            return owns(lock_path, claim_path=claim_path)

        try:
            yield has_lock
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            lost_lock = release(lock_path, claim_path=claim_path, owner_stat=owner_stat)
            if lost_lock and body_error is None:
                raise LockError(f"lost lock at {lock_path}")
    finally:
        unlink_if_exists(claim_path)
