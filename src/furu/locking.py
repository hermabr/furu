from __future__ import annotations

import errno
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

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


@dataclass(frozen=True, slots=True)
class _LockManifest:
    claim_path: Path
    lock_paths: tuple[Path, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 2,
                "claim_path": str(self.claim_path),
                "lock_paths": [str(path) for path in self.lock_paths],
            },
            indent=2,
            sort_keys=True,
        )


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
        key=lambda path: os.fspath(path),
    )
    if not normalized:
        raise ValueError("lock_many() requires at least one lock path")
    return normalized


def _assert_same_filesystem(lock_paths: Iterable[Path]) -> None:
    iterator = iter(lock_paths)
    first_path = next(iterator)
    first_device = first_path.parent.stat().st_dev
    for lock_path in iterator:
        if lock_path.parent.stat().st_dev != first_device:
            raise LockAcquireError(
                "hardlink-based locking requires every lock path to be on the same "
                "filesystem device"
            )


def _create_manifest(lock_paths: list[Path]) -> _LockManifest:
    claim_path = _nfs_safe_unique_name(lock_paths[0], name="claim").resolve(
        strict=False
    )
    return _LockManifest(claim_path=claim_path, lock_paths=tuple(lock_paths))


def _write_claim_file(manifest: _LockManifest, *, lifetime_s: float) -> None:
    fd = os.open(manifest.claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
        f.flush()
        os.fsync(f.fileno())
    _touch_future(manifest.claim_path, lifetime_s=lifetime_s)


def _parse_manifest(raw_manifest: str, *, source_path: Path) -> _LockManifest:
    try:
        payload = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: malformed lock manifest"
        ) from exc

    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: unsupported lock "
            "manifest"
        )

    claim_path_value = payload.get("claim_path")
    lock_paths_value = payload.get("lock_paths")
    if not isinstance(claim_path_value, str):
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: missing claim_path"
        )
    if not isinstance(lock_paths_value, list) or not lock_paths_value:
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: missing lock_paths"
        )
    if not all(isinstance(path, str) for path in lock_paths_value):
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: invalid lock_paths"
        )

    claim_path = Path(claim_path_value)
    lock_paths = tuple(Path(path) for path in lock_paths_value)
    if not claim_path.is_absolute() or any(not path.is_absolute() for path in lock_paths):
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: non-absolute manifest"
        )
    if len(lock_paths) != len(set(lock_paths)):
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: duplicate lock paths"
        )
    if source_path not in lock_paths:
        raise LockAcquireError(
            f"cannot safely break stale lock at {source_path}: manifest does not "
            "include contested lock path"
        )

    return _LockManifest(claim_path=claim_path, lock_paths=lock_paths)


def _read_manifest(path: Path) -> _LockManifest | None:
    raw_manifest = _safe_read_text(path)
    if raw_manifest is None:
        return None
    return _parse_manifest(raw_manifest, source_path=path)


def _is_owner(*, lock_path: Path, owner_claim_path: Path) -> bool:
    lock_stat = _safe_stat(lock_path)
    owner_stat = _safe_stat(owner_claim_path)
    return (
        lock_stat is not None
        and owner_stat is not None
        and os.path.samestat(lock_stat, owner_stat)
    )


def _owns_all_locks(*, lock_paths: Iterable[Path], owner_claim_path: Path) -> bool:
    owner_stat = _safe_stat(owner_claim_path)
    if owner_stat is None:
        return False
    for lock_path in lock_paths:
        lock_stat = _safe_stat(lock_path)
        if lock_stat is None or not os.path.samestat(lock_stat, owner_stat):
            return False
    return True


def _try_acquire_lock(*, lock_path: Path, owner_claim_path: Path) -> bool:
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

    if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
        _safe_unlink_if_exists(lock_path)
        return False
    return True


def _release_acquired_subset(
    *, lock_paths: Iterable[Path], owner_claim_path: Path
) -> None:
    for lock_path in lock_paths:
        if _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
            _safe_unlink_if_exists(lock_path)


def _release_lock_group(*, lock_paths: list[Path], owner_claim_path: Path) -> None:
    lost_lock = not _owns_all_locks(
        lock_paths=lock_paths,
        owner_claim_path=owner_claim_path,
    )

    for lock_path in lock_paths:
        if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
            lost_lock = True
            continue
        try:
            lock_path.unlink()
        except OSError as exc:
            if _is_missing_or_stale(exc):
                lost_lock = True
                continue
            raise

    _safe_unlink_if_exists(owner_claim_path)
    if lost_lock:
        raise NotLockedError(
            f"lock group is no longer fully owned by {owner_claim_path}"
        )


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
    lock_stat = _safe_stat(lock_path)
    if lock_stat is None or lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
        return False

    manifest = _read_manifest(lock_path)
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
        if not os.path.samestat(current_lock_stat, claim_stat):
            raise StaleLockRaceError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        for member_lock_path in manifest.lock_paths:
            member_lock_stat = _safe_stat(member_lock_path)
            if member_lock_stat is not None and os.path.samestat(
                member_lock_stat, claim_stat
            ):
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
    lock_paths: list[Path],
    owner_claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: threading.Event,
) -> None:
    if not _owns_all_locks(lock_paths=lock_paths, owner_claim_path=owner_claim_path):
        return
    if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
        return

    next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    while True:
        timeout_s = max(next_heartbeat_at - time.monotonic(), 0.0)
        if stop_event.wait(timeout_s):
            return
        if not _owns_all_locks(
            lock_paths=lock_paths,
            owner_claim_path=owner_claim_path,
        ):
            return
        if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
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
        f"{acquire_timeout_s:g} seconds: "
        + ", ".join(str(path) for path in lock_paths)
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
) -> Iterator[Callable[[], bool]]:
    normalized_lock_paths = _normalize_lock_paths(lock_paths)
    _assert_same_filesystem(normalized_lock_paths)
    manifest = _create_manifest(normalized_lock_paths)

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S,
            heartbeat_interval_s / 3,
        )

    _write_claim_file(manifest, lifetime_s=lifetime_s)

    try:
        acquire_deadline_monotonic_s = time.monotonic() + max(acquire_timeout_s, 0.0)
        while True:
            acquired_paths: list[Path] = []
            blocked_path: Path | None = None
            try:
                for lock_path in normalized_lock_paths:
                    if _try_acquire_lock(
                        lock_path=lock_path,
                        owner_claim_path=manifest.claim_path,
                    ):
                        acquired_paths.append(lock_path)
                        continue
                    blocked_path = lock_path
                    break
            except BaseException:
                _release_acquired_subset(
                    lock_paths=acquired_paths,
                    owner_claim_path=manifest.claim_path,
                )
                raise

            if blocked_path is None:
                break

            try:
                broke_stale_lock = _break_stale_lock_group(
                    lock_path=blocked_path,
                    lifetime_s=lifetime_s,
                )
            finally:
                _release_acquired_subset(
                    lock_paths=acquired_paths,
                    owner_claim_path=manifest.claim_path,
                )
            if broke_stale_lock:
                continue

            sleep_s = _acquire_retry_sleep_s(
                lock_path=blocked_path,
                acquire_deadline_monotonic_s=acquire_deadline_monotonic_s,
                acquire_poll_interval_s=acquire_poll_interval_s,
            )
            if sleep_s <= 0.0:
                raise LockAcquireError(
                    _lock_timeout_message(
                        normalized_lock_paths,
                        acquire_timeout_s=acquire_timeout_s,
                    )
                )
            time.sleep(sleep_s)

        _touch_future(manifest.claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "lock_paths": normalized_lock_paths,
                "owner_claim_path": manifest.claim_path,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{normalized_lock_paths[0].name}",
            daemon=True,
        )
        heartbeat.start()
        body_error: BaseException | None = None

        def has_lock() -> bool:
            return _owns_all_locks(
                lock_paths=normalized_lock_paths,
                owner_claim_path=manifest.claim_path,
            )

        try:
            yield has_lock
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop_event.set()
            heartbeat.join(timeout=HEARTBEAT_SHUTDOWN_GRACE_S)
            try:
                _release_lock_group(
                    lock_paths=normalized_lock_paths,
                    owner_claim_path=manifest.claim_path,
                )
            except NotLockedError:
                if body_error is None:
                    raise LockLostError(_lock_lost_message(normalized_lock_paths))
    finally:
        _safe_unlink_if_exists(manifest.claim_path)


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
