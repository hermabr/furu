from __future__ import annotations

import errno
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from furu.utils import _nfs_safe_unique_name

# TODO: move these to the general config rather than having these be hard coded here, so that the user can override them. also consider if we can remove some of these/simplify
CLOCK_SLOP_S = 10
DEFAULT_LIFETIME_S = 35.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 5.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01


class LockAcquireError(RuntimeError):
    pass


class LockLostError(RuntimeError):
    pass


class LockManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim_path: Path
    lock_paths: tuple[Path, ...]

    @model_validator(mode="after")
    def _validate_paths(self) -> LockManifest:
        if not self.claim_path.is_absolute():
            raise ValueError("claim_path must be absolute")
        if not self.lock_paths:
            raise ValueError("lock_paths must not be empty")
        if any(not path.is_absolute() for path in self.lock_paths):
            raise ValueError("lock_paths must be absolute")
        if len(self.lock_paths) != len(set(self.lock_paths)):
            raise ValueError("lock_paths must be unique")
        return self


def _is_missing_or_stale(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return os.path.samestat(left, right)


def _is_stale(stat_result: os.stat_result) -> bool:
    return stat_result.st_mtime + CLOCK_SLOP_S <= time.time()


def _lock_timeout_message(lock_paths: list[Path], acquire_timeout_s: float) -> str:
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


def normalize_lock_paths(lock_paths: Iterable[Path]) -> list[Path]:
    normalized = sorted(
        {path.resolve(strict=False) for path in lock_paths},
        key=lambda path: os.fspath(path),
    )
    if not normalized:
        raise ValueError("lock_many() requires at least one lock path")
    return normalized


def assert_same_filesystem(lock_paths: Iterable[Path]) -> None:
    iterator = iter(lock_paths)
    first_path = next(iterator)
    first_device = first_path.parent.stat().st_dev
    for lock_path in iterator:
        if lock_path.parent.stat().st_dev != first_device:
            raise LockAcquireError(
                "hardlink-based locking requires every lock path to be on the same filesystem device"
            )


def read_manifest(path: Path) -> LockManifest | None:
    raw = read_text_or_none(path)
    if raw is None:
        return None

    contested_path = path.resolve(strict=False)
    try:
        manifest = LockManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise LockAcquireError(
            f"cannot safely break stale lock at {path}: malformed lock manifest"
        ) from exc

    if contested_path not in manifest.lock_paths:
        raise LockAcquireError(
            f"cannot safely break stale lock at {path}: manifest does not include contested lock path"
        )
    return manifest


def owns(lock_paths: Iterable[Path], *, claim_path: Path) -> bool:
    claim_stat = stat_or_none(claim_path)
    if claim_stat is None:
        return False
    for lock_path in lock_paths:
        lock_stat = stat_or_none(lock_path)
        if lock_stat is None or not _same_inode(lock_stat, claim_stat):
            return False
    return True


def try_link(*, lock_path: Path, claim_path: Path) -> bool:
    try:
        os.link(claim_path, lock_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise LockAcquireError(
                f"hardlink-based locking cannot link {claim_path} to {lock_path}"
            ) from exc
        if _is_missing_or_stale(exc):
            return False
        raise

    if not owns([lock_path], claim_path=claim_path):
        unlink_if_exists(lock_path)
        return False
    return True


def release_acquired_subset(lock_paths: Iterable[Path], *, claim_path: Path) -> None:
    for lock_path in lock_paths:
        if owns([lock_path], claim_path=claim_path):
            unlink_if_exists(lock_path)


def _unlink_matching_group_members(
    lock_paths: Iterable[Path], *, reference_stat: os.stat_result
) -> bool:
    lost_lock = False
    for lock_path in lock_paths:
        lock_stat = stat_or_none(lock_path)
        if lock_stat is None or not _same_inode(lock_stat, reference_stat):
            lost_lock = True
            continue
        try:
            lock_path.unlink()
        except OSError as exc:
            if _is_missing_or_stale(exc):
                lost_lock = True
                continue
            raise

    return lost_lock


def release_owned_group(
    lock_paths: Iterable[Path],
    *,
    claim_path: Path,
    owner_stat: os.stat_result,
) -> bool:
    lock_paths = tuple(lock_paths)
    claim_stat = stat_or_none(claim_path)
    reference_stat = claim_stat
    lost_lock = claim_stat is None

    if reference_stat is None:
        for lock_path in lock_paths:
            lock_stat = stat_or_none(lock_path)
            if lock_stat is not None and _same_inode(lock_stat, owner_stat):
                reference_stat = lock_stat
                break

    if reference_stat is not None:
        lost_lock = (
            _unlink_matching_group_members(lock_paths, reference_stat=reference_stat)
            or lost_lock
        )

    unlink_if_exists(claim_path)
    return lost_lock


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
            raise LockAcquireError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        _unlink_matching_group_members(
            manifest.lock_paths,
            reference_stat=reference_stat,
        )
        unlink_if_exists(manifest.claim_path)
        return True
    finally:
        unlink_if_exists(break_path)


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
    lock_paths: list[Path],
    claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: threading.Event,
) -> None:
    if not owns(lock_paths, claim_path=claim_path):
        return
    if not _try_touch_future(claim_path, lifetime_s=lifetime_s):
        return

    while not stop_event.wait(heartbeat_interval_s):
        if not owns(lock_paths, claim_path=claim_path):
            return
        if not _try_touch_future(claim_path, lifetime_s=lifetime_s):
            return


@contextmanager
def lock_many(
    lock_paths: Iterable[Path],
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Iterator[Callable[[], bool]]:
    lock_paths = normalize_lock_paths(lock_paths)
    assert_same_filesystem(lock_paths)

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S,
            heartbeat_interval_s / 3,
        )

    claim_path = _nfs_safe_unique_name(lock_paths[0], name="claim").resolve(
        strict=False
    )
    manifest = LockManifest(
        claim_path=claim_path,
        lock_paths=tuple(lock_paths),
    )
    _write_claim_file(manifest, lifetime_s=lifetime_s)
    owner_stat = claim_path.stat()

    try:
        deadline = time.monotonic() + max(acquire_timeout_s, 0.0)

        while True:
            acquired: list[Path] = []
            blocked: Path | None = None
            try:
                for lock_path in lock_paths:
                    if try_link(lock_path=lock_path, claim_path=claim_path):
                        acquired.append(lock_path)
                    else:
                        blocked = lock_path
                        break
            except BaseException:
                release_acquired_subset(acquired, claim_path=claim_path)
                raise

            if blocked is None:
                break

            try:
                broke_stale = break_stale(blocked, lifetime_s=lifetime_s)
            finally:
                release_acquired_subset(acquired, claim_path=claim_path)

            if broke_stale:
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockAcquireError(
                    _lock_timeout_message(lock_paths, acquire_timeout_s)
                )
            time.sleep(min(acquire_poll_interval_s, remaining))

        touch_future(claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=heartbeat_loop,
            kwargs={
                "lock_paths": lock_paths,
                "claim_path": claim_path,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{lock_paths[0].name}",
            daemon=True,
        )
        heartbeat.start()
        body_error: BaseException | None = None

        def has_lock() -> bool:
            return owns(lock_paths, claim_path=claim_path)

        try:
            yield has_lock
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            lost_lock = release_owned_group(
                lock_paths,
                claim_path=claim_path,
                owner_stat=owner_stat,
            )
            if lost_lock and body_error is None:
                raise LockLostError(_lock_lost_message(lock_paths))
    finally:
        unlink_if_exists(claim_path)


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
