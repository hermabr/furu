import errno
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from furu.utils import _nfs_safe_unique_name

# TODO: i think it should be possible to make this code shorter and cleaner while still keeping it NFS safe, so will rewrite at some point

# TODO: move these to the general config rather than having these be hard coded here, so that the user can override them. also consider if we can remove some of these/simplify
CLOCK_SLOP_S = 10
DEFAULT_LIFETIME_S = 35.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 5.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01
LOCK_MANIFEST_VERSION = 2


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
    version: int
    claim_path: Path
    lock_paths: tuple[Path, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "claim_path": os.fspath(self.claim_path),
                "lock_paths": [os.fspath(path) for path in self.lock_paths],
            },
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


def _safe_rmdir_if_exists(path: Path) -> None:
    try:
        path.rmdir()
    except OSError as exc:
        if not _is_missing_or_stale(exc) and exc.errno != errno.ENOTEMPTY:
            raise


def _normalize_lock_paths(lock_paths: Iterable[Path]) -> tuple[Path, ...]:
    resolved_paths: set[Path] = set()
    for path in lock_paths:
        resolved_paths.add(path.resolve())

    normalized_list: list[Path] = list(resolved_paths)
    normalized_list.sort(key=str)
    normalized = tuple(normalized_list)
    if not normalized:
        raise ValueError("lock_many() requires at least one lock path")
    return normalized


def _assert_same_device(lock_paths: Sequence[Path]) -> None:
    devices = {path.parent.resolve().stat().st_dev for path in lock_paths}
    if len(devices) != 1:
        raise ValueError(
            "lock_many() requires every lock path to live on the same filesystem device"
        )


def _is_owner(*, lock_path: Path, owner_claim_path: Path) -> bool:
    lock_stat = _safe_stat(lock_path)
    owner_stat = _safe_stat(owner_claim_path)
    return (
        lock_stat is not None
        and owner_stat is not None
        and os.path.samestat(lock_stat, owner_stat)
    )


def _owns_all_locks(*, lock_paths: Sequence[Path], owner_claim_path: Path) -> bool:
    owner_stat = _safe_stat(owner_claim_path)
    if owner_stat is None:
        return False

    for lock_path in lock_paths:
        lock_stat = _safe_stat(lock_path)
        if lock_stat is None or not os.path.samestat(lock_stat, owner_stat):
            return False
    return True


def _write_manifest(owner_claim_path: Path, lock_paths: Sequence[Path]) -> _LockManifest:
    manifest = _LockManifest(
        version=LOCK_MANIFEST_VERSION,
        claim_path=owner_claim_path.resolve(),
        lock_paths=tuple(lock_paths),
    )

    fd = os.open(owner_claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
        f.flush()
        os.fsync(f.fileno())

    return manifest


def _legacy_manifest_from_claim_path(
    *, lock_path: Path, claim_path: Path
) -> _LockManifest | None:
    if not (
        claim_path.parent == lock_path.parent
        and claim_path.name.startswith(f"{lock_path.name}.")
        and claim_path.name.endswith(".claim")
    ):
        return None

    if _safe_read_path(claim_path) != os.fspath(claim_path):
        return None

    return _LockManifest(
        version=1,
        claim_path=claim_path.resolve(),
        lock_paths=(lock_path.resolve(),),
    )


def _safe_read_lock_manifest(lock_path: Path) -> _LockManifest | None:
    raw = _safe_read_path(lock_path)
    if raw is None:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _legacy_manifest_from_claim_path(
            lock_path=lock_path.resolve(), claim_path=Path(raw)
        )

    if not isinstance(payload, dict):
        return None

    if payload.get("version") != LOCK_MANIFEST_VERSION:
        return None

    claim_path_raw = payload.get("claim_path")
    lock_paths_raw = payload.get("lock_paths")
    if not isinstance(claim_path_raw, str) or not isinstance(lock_paths_raw, list):
        return None
    if not all(isinstance(path, str) for path in lock_paths_raw):
        return None

    resolved_lock_paths: set[Path] = set()
    for path in lock_paths_raw:
        resolved_lock_paths.add(Path(path).resolve())

    normalized_lock_paths_list: list[Path] = list(resolved_lock_paths)
    normalized_lock_paths_list.sort(key=str)
    normalized_lock_paths = tuple(normalized_lock_paths_list)
    normalized_lock_path = lock_path.resolve()
    if normalized_lock_path not in normalized_lock_paths:
        return None

    return _LockManifest(
        version=LOCK_MANIFEST_VERSION,
        claim_path=Path(claim_path_raw).resolve(),
        lock_paths=normalized_lock_paths,
    )


def _try_acquire_stale_break_dir(*, claim_path: Path, lifetime_s: float) -> Path | None:
    break_dir = claim_path.with_name(f"{claim_path.name}.break")
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
    owner_stat = _safe_stat(owner_claim_path)
    if (
        lock_stat is None
        or owner_stat is None
        or lock_stat.st_nlink < 2
        or not os.path.samestat(lock_stat, owner_stat)
    ):
        _safe_unlink_if_exists(lock_path)
        return False

    return True


def _release_attempted_locks(
    *, acquired_paths: Sequence[Path], owner_claim_path: Path
) -> None:
    for lock_path in reversed(acquired_paths):
        if _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
            _safe_unlink_if_exists(lock_path)


def _break_stale_lock(*, lock_path: Path, lifetime_s: float) -> None:
    lock_stat = _safe_stat(lock_path)
    if lock_stat is None or lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
        return

    manifest = _safe_read_lock_manifest(lock_path)
    if manifest is None:
        return

    break_dir = _try_acquire_stale_break_dir(
        claim_path=manifest.claim_path, lifetime_s=lifetime_s
    )
    if break_dir is None:
        return

    try:
        current_lock_stat = _safe_stat(lock_path)
        if current_lock_stat is None:
            return

        if not _is_owner(lock_path=lock_path, owner_claim_path=manifest.claim_path):
            raise StaleLockRaceError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        if current_lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
            return

        for member_path in manifest.lock_paths:
            if _is_owner(lock_path=member_path, owner_claim_path=manifest.claim_path):
                _safe_unlink_if_exists(member_path)

        _safe_unlink_if_exists(manifest.claim_path)
    finally:
        _safe_rmdir_if_exists(break_dir)


def _release_locks(*, lock_paths: Sequence[Path], owner_claim_path: Path) -> None:
    lost_path: Path | None = None

    for lock_path in lock_paths:
        if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
            if lost_path is None:
                lost_path = lock_path
            continue

        try:
            lock_path.unlink()
        except OSError as exc:
            if _is_missing_or_stale(exc):
                if lost_path is None:
                    lost_path = lock_path
                continue
            raise

    _safe_unlink_if_exists(owner_claim_path)

    if lost_path is not None:
        raise NotLockedError(f"lock {lost_path} is owned by another process")


def _parent_is_alive(*, parent_pid: int) -> bool:
    if os.getppid() == parent_pid:
        return True

    try:
        os.kill(parent_pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _try_release_if_parent_dead(
    *, lock_path: Path, owner_claim_path: Path, parent_pid: int
) -> bool:
    if _parent_is_alive(parent_pid=parent_pid):
        return False

    with suppress(NotLockedError, OSError):
        _release_locks(lock_paths=(lock_path,), owner_claim_path=owner_claim_path)
    with suppress(OSError):
        _safe_unlink_if_exists(owner_claim_path)
    return True


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
    lock_paths: Sequence[Path],
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
        if not _owns_all_locks(lock_paths=lock_paths, owner_claim_path=owner_claim_path):
            return
        if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
            return
        next_heartbeat_at = time.monotonic() + heartbeat_interval_s


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
    _assert_same_device(normalized_lock_paths)

    owner_claim_path = _nfs_safe_unique_name(normalized_lock_paths[0], name="claim")

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S, heartbeat_interval_s / 3
        )

    _write_manifest(owner_claim_path, normalized_lock_paths)
    _touch_future(owner_claim_path, lifetime_s=lifetime_s)

    try:
        acquire_deadline_monotonic_s = time.monotonic() + max(acquire_timeout_s, 0.0)
        while True:
            acquired_paths: list[Path] = []
            failed_path: Path | None = None

            try:
                for lock_path in normalized_lock_paths:
                    if _try_acquire_lock(
                        lock_path=lock_path, owner_claim_path=owner_claim_path
                    ):
                        acquired_paths.append(lock_path)
                        continue
                    failed_path = lock_path
                    break
            except BaseException:
                _release_attempted_locks(
                    acquired_paths=acquired_paths,
                    owner_claim_path=owner_claim_path,
                )
                raise

            if failed_path is None:
                break

            _release_attempted_locks(
                acquired_paths=acquired_paths,
                owner_claim_path=owner_claim_path,
            )
            _break_stale_lock(lock_path=failed_path, lifetime_s=lifetime_s)

            sleep_s = _acquire_retry_sleep_s(
                lock_path=failed_path,
                acquire_deadline_monotonic_s=acquire_deadline_monotonic_s,
                acquire_poll_interval_s=acquire_poll_interval_s,
            )
            if sleep_s <= 0.0:
                if len(normalized_lock_paths) == 1:
                    raise LockAcquireError(
                        f"could not acquire lock at {normalized_lock_paths[0]} "
                        f"within {acquire_timeout_s:g} seconds"
                    )
                raise LockAcquireError(
                    "could not acquire lock set "
                    f"{[os.fspath(path) for path in normalized_lock_paths]} "
                    f"within {acquire_timeout_s:g} seconds"
                )
            time.sleep(sleep_s)

        _touch_future(owner_claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "lock_paths": normalized_lock_paths,
                "owner_claim_path": owner_claim_path,
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
                owner_claim_path=owner_claim_path,
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
                _release_locks(
                    lock_paths=normalized_lock_paths,
                    owner_claim_path=owner_claim_path,
                )
            except NotLockedError:
                if body_error is None:
                    raise LockLostError(f"lost lock at {normalized_lock_paths[0]}")
    finally:
        _safe_unlink_if_exists(owner_claim_path)


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
