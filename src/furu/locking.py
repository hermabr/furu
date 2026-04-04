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
            raise ValueError("lock_paths cannot be empty")
        if any(not path.is_absolute() for path in self.lock_paths):
            raise ValueError("every lock path must be absolute")
        if len(self.lock_paths) != len(set(self.lock_paths)):
            raise ValueError("lock_paths cannot contain duplicates")
        return self

    @classmethod
    def new(cls, lock_paths: Iterable[Path]) -> LockManifest:
        normalized_lock_paths = sorted(
            {path.resolve(strict=False) for path in lock_paths},
            key=lambda path: os.fspath(path),
        )
        if not normalized_lock_paths:
            raise ValueError("lock_many() requires at least one lock path")

        first_device = normalized_lock_paths[0].parent.stat().st_dev
        for lock_path in normalized_lock_paths[1:]:
            if lock_path.parent.stat().st_dev != first_device:
                raise LockAcquireError(
                    "hardlink-based locking requires every lock path to be on the same "
                    "filesystem device"
                )

        claim_path = _nfs_safe_unique_name(
            normalized_lock_paths[0],
            name="claim",
        ).resolve(strict=False)
        return cls(
            claim_path=claim_path,
            lock_paths=tuple(normalized_lock_paths),
        )

    @classmethod
    def read_from(cls, path: Path) -> LockManifest | None:
        source_path = path.resolve(strict=False)
        raw_manifest = _safe_read_text(source_path)
        if raw_manifest is None:
            return None

        try:
            manifest = cls.model_validate_json(raw_manifest)
        except ValidationError as exc:
            raise LockAcquireError(
                f"cannot safely break stale lock at {source_path}: malformed lock manifest"
            ) from exc

        if source_path not in manifest.lock_paths:
            raise LockAcquireError(
                f"cannot safely break stale lock at {source_path}: manifest does not "
                "include contested lock path"
            )
        return manifest

    def write_claim_file(self, *, lifetime_s: float) -> None:
        fd = os.open(self.claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        _touch_future(self.claim_path, lifetime_s=lifetime_s)

    def owns(self, lock_path: Path) -> bool:
        claim_stat = _safe_stat(self.claim_path)
        lock_stat = _safe_stat(lock_path.resolve(strict=False))
        return (
            claim_stat is not None
            and lock_stat is not None
            and os.path.samestat(lock_stat, claim_stat)
        )

    def owns_all(self) -> bool:
        claim_stat = _safe_stat(self.claim_path)
        if claim_stat is None:
            return False

        for lock_path in self.lock_paths:
            lock_stat = _safe_stat(lock_path)
            if lock_stat is None or not os.path.samestat(lock_stat, claim_stat):
                return False
        return True

    def release_subset(self, lock_paths: Iterable[Path]) -> None:
        for lock_path in lock_paths:
            normalized_path = lock_path.resolve(strict=False)
            if self.owns(normalized_path):
                _safe_unlink_if_exists(normalized_path)

    def release(self) -> bool:
        lost_lock = not self.owns_all()
        claim_stat = _safe_stat(self.claim_path)
        if claim_stat is None:
            lost_lock = True

        for lock_path in self.lock_paths:
            lock_stat = _safe_stat(lock_path)
            if lock_stat is None:
                lost_lock = True
                continue
            if claim_stat is None or not os.path.samestat(lock_stat, claim_stat):
                lost_lock = True
                continue

            try:
                lock_path.unlink()
            except OSError as exc:
                if _is_missing_or_stale(exc):
                    lost_lock = True
                    continue
                raise

        try:
            self.claim_path.unlink()
        except OSError as exc:
            if _is_missing_or_stale(exc):
                lost_lock = True
            else:
                raise

        return lost_lock


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


def _try_link(*, lock_path: Path, owner_claim_path: Path) -> bool:
    try:
        os.link(owner_claim_path, lock_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise LockAcquireError(
                f"hardlink-based locking cannot link {owner_claim_path} to {lock_path}"
            ) from exc
        if _is_missing_or_stale(exc):
            return False
        raise

    owner_stat = _safe_stat(owner_claim_path)
    lock_stat = _safe_stat(lock_path)
    if (
        owner_stat is None
        or lock_stat is None
        or not os.path.samestat(lock_stat, owner_stat)
    ):
        _safe_unlink_if_exists(lock_path)
        return False
    return True


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


def _break_stale_lock_group(*, lock_path: Path, lifetime_s: float) -> bool:
    contested_path = lock_path.resolve(strict=False)
    contested_stat = _safe_stat(contested_path)
    if contested_stat is None or contested_stat.st_mtime + CLOCK_SLOP_S > time.time():
        return False

    manifest = LockManifest.read_from(contested_path)
    if manifest is None:
        return False

    break_dir = _try_acquire_stale_break_dir(
        claim_path=manifest.claim_path,
        lifetime_s=lifetime_s,
    )
    if break_dir is None:
        return False

    try:
        current_stat = _safe_stat(contested_path)
        if current_stat is None:
            return True
        if current_stat.st_mtime + CLOCK_SLOP_S > time.time():
            return False

        claim_stat = _safe_stat(manifest.claim_path)
        if claim_stat is None:
            raise LockAcquireError(
                f"cannot safely break stale lock at {contested_path}: missing claim "
                f"file {manifest.claim_path}"
            )
        if not os.path.samestat(current_stat, claim_stat):
            raise LockAcquireError(
                f"lock {contested_path} changed owners while breaking a stale lock"
            )

        for member_lock_path in manifest.lock_paths:
            member_stat = _safe_stat(member_lock_path)
            if member_stat is not None and os.path.samestat(member_stat, claim_stat):
                _safe_unlink_if_exists(member_lock_path)
        _safe_unlink_if_exists(manifest.claim_path)
        return True
    finally:
        _safe_rmdir_if_exists(break_dir)


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
    manifest: LockManifest,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: threading.Event,
) -> None:
    if not manifest.owns_all():
        return
    if not _try_touch_future(manifest.claim_path, lifetime_s=lifetime_s):
        return

    next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    while True:
        timeout_s = max(next_heartbeat_at - time.monotonic(), 0.0)
        if stop_event.wait(timeout_s):
            return
        if not manifest.owns_all():
            return
        if not _try_touch_future(manifest.claim_path, lifetime_s=lifetime_s):
            return
        next_heartbeat_at = time.monotonic() + heartbeat_interval_s


def _lock_timeout_message(
    lock_paths: tuple[Path, ...], *, acquire_timeout_s: float
) -> str:
    if len(lock_paths) == 1:
        return (
            f"could not acquire lock at {lock_paths[0]} within "
            f"{acquire_timeout_s:g} seconds"
        )
    return (
        "could not acquire lock set within "
        f"{acquire_timeout_s:g} seconds: " + ", ".join(str(path) for path in lock_paths)
    )


def _lock_lost_message(lock_paths: tuple[Path, ...]) -> str:
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
) -> Iterator[Callable[[], bool]]:
    manifest = LockManifest.new(lock_paths)

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S,
            heartbeat_interval_s / 3,
        )

    manifest.write_claim_file(lifetime_s=lifetime_s)

    try:
        acquire_deadline_monotonic_s = time.monotonic() + max(acquire_timeout_s, 0.0)
        while True:
            acquired: list[Path] = []
            blocked: Path | None = None

            try:
                for lock_path in manifest.lock_paths:
                    if _try_link(
                        lock_path=lock_path,
                        owner_claim_path=manifest.claim_path,
                    ):
                        acquired.append(lock_path)
                    else:
                        blocked = lock_path
                        break
            except BaseException:
                manifest.release_subset(acquired)
                raise

            if blocked is None:
                break

            try:
                broke_stale = _break_stale_lock_group(
                    lock_path=blocked,
                    lifetime_s=lifetime_s,
                )
            finally:
                manifest.release_subset(acquired)

            if broke_stale:
                continue

            sleep_s = _acquire_retry_sleep_s(
                lock_path=blocked,
                acquire_deadline_monotonic_s=acquire_deadline_monotonic_s,
                acquire_poll_interval_s=acquire_poll_interval_s,
            )
            if sleep_s <= 0.0:
                raise LockAcquireError(
                    _lock_timeout_message(
                        manifest.lock_paths,
                        acquire_timeout_s=acquire_timeout_s,
                    )
                )
            time.sleep(sleep_s)

        _touch_future(manifest.claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "manifest": manifest,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{manifest.lock_paths[0].name}",
            daemon=True,
        )
        heartbeat.start()

        body_error: BaseException | None = None
        try:
            yield manifest.owns_all
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            lost_lock = manifest.release()
            if lost_lock and body_error is None:
                raise LockLostError(_lock_lost_message(manifest.lock_paths))
    finally:
        _safe_unlink_if_exists(manifest.claim_path)
