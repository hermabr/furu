import errno
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from furu.utils import _nfs_safe_unique_name

# TODO: i think it should be possible to make this code shorter and cleaner while still keeping it NFS safe, so will rewrite at some point

# TODO: move these to the general config rather than having these be hard coded here, so that the user can override them. also consider if we can remove some of these/simplify
CLOCK_SLOP_S = 10  # NFS systems can be slow, so we add a 10 second safety margin TODO: assume this is included in the 120 second lease time?
DEFAULT_LIFETIME_S = 35.0
DEFAULT_HEARTBEAT_INTERVAL_S = 15.0
DEFAULT_ACQUIRE_POLL_INTERVAL_S = 5.0
HEARTBEAT_SHUTDOWN_GRACE_S = 0.01

# TODO: currently, we don't handle SIGTERM/SIGINT in the parent. we need to add this (but this needs to be integrated with the general furu handling of SIGINT/SIGTERM)


class LockAcquireError(RuntimeError):
    pass


class NotLockedError(RuntimeError):
    pass


class LockLostError(RuntimeError):
    pass


class StaleLockRaceError(LockAcquireError):
    pass


@dataclass(frozen=True, slots=True)
class _ClaimManifest:
    claim_path: Path
    member_lock_paths: tuple[Path, ...]
    batch_id: str
    hostname: str
    pid: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "batch_id": self.batch_id,
                "claim_path": str(self.claim_path),
                "hostname": self.hostname,
                "member_lock_paths": [str(path) for path in self.member_lock_paths],
                "pid": self.pid,
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


def _try_acquire_stale_break_dir(*, lock_path: Path, lifetime_s: float) -> Path | None:
    break_dir = lock_path.with_name(f"{lock_path.name}.break")
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


def _canonicalize_lock_paths(lock_paths: Iterable[Path]) -> tuple[Path, ...]:
    canonical_lock_paths = {path.resolve() for path in lock_paths}
    if not canonical_lock_paths:
        raise ValueError("lock_many() requires at least one lock path")
    return tuple(sorted(canonical_lock_paths, key=lambda path: os.fspath(path)))


def _build_claim_manifest(
    owner_claim_path: Path,
    member_lock_paths: tuple[Path, ...],
) -> _ClaimManifest:
    return _ClaimManifest(
        claim_path=owner_claim_path,
        member_lock_paths=member_lock_paths,
        batch_id=uuid.uuid4().hex,
        hostname=socket.getfqdn(),
        pid=os.getpid(),
    )


def _write_claim_manifest(
    owner_claim_path: Path,
    member_lock_paths: tuple[Path, ...],
) -> None:
    manifest = _build_claim_manifest(owner_claim_path, member_lock_paths)
    fd = os.open(owner_claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
        f.flush()
        os.fsync(f.fileno())


def _looks_like_legacy_claim_path(lock_path: Path, claim_path: Path) -> bool:
    return (
        claim_path.parent == lock_path.parent
        and claim_path.name.startswith(f"{lock_path.name}.")
        and claim_path.name.endswith(".claim")
    )


def _parse_claim_manifest_text(
    text: str,
    *,
    lock_path: Path,
) -> _ClaimManifest | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        claim_path = Path(text)
        if not _looks_like_legacy_claim_path(lock_path, claim_path):
            return None
        return _ClaimManifest(
            claim_path=claim_path,
            member_lock_paths=(lock_path,),
            batch_id=claim_path.name,
            hostname="",
            pid=0,
        )

    if not isinstance(payload, dict):
        return None

    claim_path_raw = payload.get("claim_path")
    member_lock_paths_raw = payload.get("member_lock_paths")
    batch_id = payload.get("batch_id")
    hostname = payload.get("hostname")
    pid = payload.get("pid")
    if not (
        isinstance(claim_path_raw, str)
        and isinstance(member_lock_paths_raw, list)
        and isinstance(batch_id, str)
        and isinstance(hostname, str)
        and isinstance(pid, int)
    ):
        return None

    member_lock_paths: list[Path] = []
    for raw_path in member_lock_paths_raw:
        if not isinstance(raw_path, str):
            return None
        member_lock_paths.append(Path(raw_path))

    if not member_lock_paths or lock_path not in member_lock_paths:
        return None

    return _ClaimManifest(
        claim_path=Path(claim_path_raw),
        member_lock_paths=tuple(member_lock_paths),
        batch_id=batch_id,
        hostname=hostname,
        pid=pid,
    )


def _safe_read_claim_manifest(lock_path: Path) -> _ClaimManifest | None:
    text = _safe_read_path(lock_path)
    if text is None:
        return None

    manifest = _parse_claim_manifest_text(text, lock_path=lock_path)
    if manifest is None:
        return None

    if _safe_read_path(manifest.claim_path) != text:
        return None

    return manifest


def _safe_read_breakable_claim_path(lock_path: Path) -> Path | None:
    manifest = _safe_read_claim_manifest(lock_path)
    if manifest is None:
        return None
    return manifest.claim_path


def _path_points_to_owner(*, path: Path, owner_claim_path: Path) -> bool:
    path_stat = _safe_stat(path)
    owner_stat = _safe_stat(owner_claim_path)
    return (
        path_stat is not None
        and owner_stat is not None
        and os.path.samestat(path_stat, owner_stat)
    )


def _is_owner_of_paths(
    *,
    lock_paths: tuple[Path, ...],
    owner_claim_path: Path,
) -> bool:
    return all(
        _path_points_to_owner(path=lock_path, owner_claim_path=owner_claim_path)
        for lock_path in lock_paths
    )


def _is_owner(*, lock_path: Path, owner_claim_path: Path) -> bool:
    return _is_owner_of_paths(
        lock_paths=(lock_path,),
        owner_claim_path=owner_claim_path,
    )


def _cleanup_acquired_member_links(
    acquired_lock_paths: list[Path],
    *,
    owner_claim_path: Path,
) -> None:
    for lock_path in reversed(acquired_lock_paths):
        if _path_points_to_owner(path=lock_path, owner_claim_path=owner_claim_path):
            _safe_unlink_if_exists(lock_path)


def _try_acquire_lock_paths(
    *,
    lock_paths: tuple[Path, ...],
    owner_claim_path: Path,
) -> bool:
    acquired_lock_paths: list[Path] = []

    try:
        for lock_path in lock_paths:
            try:
                os.link(owner_claim_path, lock_path)
            except FileExistsError:
                _cleanup_acquired_member_links(
                    acquired_lock_paths,
                    owner_claim_path=owner_claim_path,
                )
                return False
            except OSError as exc:
                _cleanup_acquired_member_links(
                    acquired_lock_paths,
                    owner_claim_path=owner_claim_path,
                )
                if exc.errno == errno.EXDEV:
                    raise LockAcquireError(
                        "cannot acquire grouped lock across devices; "
                        "all lock paths must be on the same filesystem"
                    ) from exc
                if _is_missing_or_stale(exc):
                    return False
                raise

            acquired_lock_paths.append(lock_path)
            owner_stat = _safe_stat(owner_claim_path)
            lock_stat = _safe_stat(lock_path)
            expected_nlink = 1 + len(acquired_lock_paths)
            if (
                owner_stat is None
                or lock_stat is None
                or not os.path.samestat(owner_stat, lock_stat)
                or owner_stat.st_nlink != expected_nlink
            ):
                _cleanup_acquired_member_links(
                    acquired_lock_paths,
                    owner_claim_path=owner_claim_path,
                )
                return False

        return True
    except BaseException:
        _cleanup_acquired_member_links(
            acquired_lock_paths,
            owner_claim_path=owner_claim_path,
        )
        raise


def _try_acquire_lock(*, lock_path: Path, owner_claim_path: Path) -> bool:
    return _try_acquire_lock_paths(
        lock_paths=(lock_path,),
        owner_claim_path=owner_claim_path,
    )


def _try_acquire_lock_paths_with_stale_break(
    *,
    lock_paths: tuple[Path, ...],
    owner_claim_path: Path,
    lifetime_s: float,
) -> bool:
    if _try_acquire_lock_paths(
        lock_paths=lock_paths,
        owner_claim_path=owner_claim_path,
    ):
        return True

    for lock_path in lock_paths:
        _break_stale_lock(lock_path=lock_path, lifetime_s=lifetime_s)

    return _try_acquire_lock_paths(
        lock_paths=lock_paths,
        owner_claim_path=owner_claim_path,
    )


def _try_acquire_lock_with_stale_break(
    *,
    lock_path: Path,
    owner_claim_path: Path,
    lifetime_s: float,
) -> bool:
    return _try_acquire_lock_paths_with_stale_break(
        lock_paths=(lock_path,),
        owner_claim_path=owner_claim_path,
        lifetime_s=lifetime_s,
    )


def _break_stale_lock(*, lock_path: Path, lifetime_s: float) -> None:
    lock_stat = _safe_stat(lock_path)
    if lock_stat is None or lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
        return

    manifest = _safe_read_claim_manifest(lock_path)
    if manifest is None:
        return

    break_dir = _try_acquire_stale_break_dir(lock_path=lock_path, lifetime_s=lifetime_s)
    if break_dir is None:
        return

    try:
        current_lock_stat = _safe_stat(lock_path)
        if current_lock_stat is None:
            return

        if not _path_points_to_owner(path=lock_path, owner_claim_path=manifest.claim_path):
            raise StaleLockRaceError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        if current_lock_stat.st_mtime + CLOCK_SLOP_S > time.time():
            return

        stale_owner_stat = _safe_stat(manifest.claim_path)
        if stale_owner_stat is None:
            raise StaleLockRaceError(
                f"lock {lock_path} changed owners while breaking a stale lock"
            )

        for member_lock_path in manifest.member_lock_paths:
            member_lock_stat = _safe_stat(member_lock_path)
            if member_lock_stat is not None and os.path.samestat(
                member_lock_stat, stale_owner_stat
            ):
                _safe_unlink_if_exists(member_lock_path)

        claim_stat = _safe_stat(manifest.claim_path)
        if claim_stat is not None and os.path.samestat(claim_stat, stale_owner_stat):
            _safe_unlink_if_exists(manifest.claim_path)
    finally:
        _safe_rmdir_if_exists(break_dir)


def _release_lock_paths(
    *,
    lock_paths: tuple[Path, ...],
    owner_claim_path: Path,
) -> None:
    if _safe_stat(owner_claim_path) is None:
        raise NotLockedError(f"lock {lock_paths[0]} does not exist")

    lost_lock = False
    for lock_path in lock_paths:
        if not _path_points_to_owner(path=lock_path, owner_claim_path=owner_claim_path):
            lost_lock = True
            continue

        try:
            lock_path.unlink()
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESTALE):
                lost_lock = True
                continue
            raise

    _safe_unlink_if_exists(owner_claim_path)

    if lost_lock:
        raise NotLockedError(f"lock {lock_paths[0]} is owned by another process")


def _release_lock(*, lock_path: Path, owner_claim_path: Path) -> None:
    if not _is_owner(lock_path=lock_path, owner_claim_path=owner_claim_path):
        raise NotLockedError(f"lock {lock_path} is owned by another process")

    try:
        lock_path.unlink()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESTALE):
            raise NotLockedError(f"lock {lock_path} does not exist") from exc
        raise
    _safe_unlink_if_exists(owner_claim_path)


def _parent_is_alive(*, parent_pid: int) -> bool:
    if os.getppid() == parent_pid:
        return True

    try:
        os.kill(parent_pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _member_lock_paths_for_owner(
    *,
    lock_path: Path,
    owner_claim_path: Path,
) -> tuple[Path, ...]:
    text = _safe_read_path(owner_claim_path)
    if text is None:
        return (lock_path,)

    manifest = _parse_claim_manifest_text(text, lock_path=lock_path)
    if manifest is None or manifest.claim_path != owner_claim_path:
        return (lock_path,)

    return manifest.member_lock_paths


def _try_release_if_parent_dead(
    *,
    lock_path: Path,
    owner_claim_path: Path,
    parent_pid: int,
) -> bool:
    if _parent_is_alive(parent_pid=parent_pid):
        return False

    member_lock_paths = _member_lock_paths_for_owner(
        lock_path=lock_path,
        owner_claim_path=owner_claim_path,
    )
    with suppress(NotLockedError, OSError):
        _release_lock_paths(
            lock_paths=member_lock_paths,
            owner_claim_path=owner_claim_path,
        )
    with suppress(OSError):
        _safe_unlink_if_exists(owner_claim_path)
    return True


def _acquire_retry_sleep_s(
    *,
    lock_paths: tuple[Path, ...],
    acquire_deadline_monotonic_s: float,
    acquire_poll_interval_s: float,
) -> float:
    remaining_wait_s = max(acquire_deadline_monotonic_s - time.monotonic(), 0.0)
    if remaining_wait_s == 0.0:
        return 0.0

    remaining_until_stale_s: list[float] = []
    for lock_path in lock_paths:
        lock_stat = _safe_stat(lock_path)
        if lock_stat is None:
            return min(acquire_poll_interval_s, remaining_wait_s)
        remaining_until_stale_s.append(lock_stat.st_mtime + CLOCK_SLOP_S - time.time())

    if not remaining_until_stale_s or min(remaining_until_stale_s) <= 0.0:
        return min(acquire_poll_interval_s, remaining_wait_s)

    return min(acquire_poll_interval_s, remaining_wait_s, *remaining_until_stale_s)


def _heartbeat_loop(
    *,
    lock_paths: tuple[Path, ...],
    owner_claim_path: Path,
    lifetime_s: float,
    heartbeat_interval_s: float,
    stop_event: threading.Event,
) -> None:
    if not _is_owner_of_paths(lock_paths=lock_paths, owner_claim_path=owner_claim_path):
        return
    if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
        return

    next_heartbeat_at = time.monotonic() + heartbeat_interval_s
    while True:
        timeout_s = max(next_heartbeat_at - time.monotonic(), 0.0)
        if stop_event.wait(timeout_s):
            return  # TODO: should i mark/log this in any way?
        if not _is_owner_of_paths(
            lock_paths=lock_paths,
            owner_claim_path=owner_claim_path,
        ):
            return  # TODO: should i mark/log this in any way?
        if not _try_touch_future(owner_claim_path, lifetime_s=lifetime_s):
            return  # TODO: should i mark/log this in any way?
        next_heartbeat_at = time.monotonic() + heartbeat_interval_s


def _lock_timeout_message(
    lock_paths: tuple[Path, ...],
    *,
    acquire_timeout_s: float,
) -> str:
    if len(lock_paths) == 1:
        return (
            f"could not acquire lock at {lock_paths[0]} "
            f"within {acquire_timeout_s:g} seconds"
        )

    rendered_paths = ", ".join(os.fspath(path) for path in lock_paths)
    return (
        f"could not acquire grouped lock at [{rendered_paths}] "
        f"within {acquire_timeout_s:g} seconds"
    )


@contextmanager
def lock_many(
    lock_paths: Iterable[Path],
    *,
    lifetime_s: float = DEFAULT_LIFETIME_S,
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    acquire_timeout_s: float | None = None,
    acquire_poll_interval_s: float | None = None,
) -> Iterator[Callable[[], bool]]:
    member_lock_paths = _canonicalize_lock_paths(lock_paths)
    owner_claim_path = _nfs_safe_unique_name(member_lock_paths[0], name="claim")

    if acquire_timeout_s is None:
        acquire_timeout_s = lifetime_s + CLOCK_SLOP_S
    if acquire_poll_interval_s is None:
        acquire_poll_interval_s = min(
            DEFAULT_ACQUIRE_POLL_INTERVAL_S, heartbeat_interval_s / 3
        )

    _write_claim_manifest(owner_claim_path, member_lock_paths)
    _touch_future(owner_claim_path, lifetime_s=lifetime_s)

    try:
        acquire_deadline_monotonic_s = time.monotonic() + max(acquire_timeout_s, 0.0)
        while not _try_acquire_lock_paths_with_stale_break(
            lock_paths=member_lock_paths,
            owner_claim_path=owner_claim_path,
            lifetime_s=lifetime_s,
        ):
            sleep_s = _acquire_retry_sleep_s(
                lock_paths=member_lock_paths,
                acquire_deadline_monotonic_s=acquire_deadline_monotonic_s,
                acquire_poll_interval_s=acquire_poll_interval_s,
            )
            if sleep_s <= 0.0:
                raise LockAcquireError(
                    _lock_timeout_message(
                        member_lock_paths,
                        acquire_timeout_s=acquire_timeout_s,
                    )
                )
            time.sleep(sleep_s)

        _touch_future(owner_claim_path, lifetime_s=lifetime_s)

        stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "lock_paths": member_lock_paths,
                "owner_claim_path": owner_claim_path,
                "lifetime_s": lifetime_s,
                "heartbeat_interval_s": heartbeat_interval_s,
                "stop_event": stop_event,
            },
            name=f"lock-heartbeat:{member_lock_paths[0].name}",
            daemon=True,
        )
        heartbeat.start()
        body_error: BaseException | None = None

        def has_lock() -> bool:
            return _is_owner_of_paths(
                lock_paths=member_lock_paths,
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
                _release_lock_paths(
                    lock_paths=member_lock_paths,
                    owner_claim_path=owner_claim_path,
                )
            except NotLockedError:
                if body_error is None:
                    raise LockLostError(f"lost lock at {member_lock_paths[0]}")
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
