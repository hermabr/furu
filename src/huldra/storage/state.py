import contextlib
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class StateManager:
    """Manages state file operations with proper locking."""

    STATE_FILE = "state.json"
    COMPUTE_LOCK = ".compute.lock"
    SUBMIT_LOCK = ".submit.lock"
    STATE_LOCK = ".state.lock"

    @classmethod
    def get_state_path(cls, directory: Path) -> Path:
        return directory / cls.STATE_FILE

    @classmethod
    def read_state(cls, directory: Path) -> Dict[str, Any]:
        """Read state file, return {"status": "missing"} if not found."""
        try:
            return json.loads(cls.get_state_path(directory).read_text())
        except Exception:
            return {"status": "missing"}

    @classmethod
    def _pid_alive(cls, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    @classmethod
    def _read_lock_pid(cls, lock_path: Path) -> Optional[int]:
        try:
            first = lock_path.read_text().strip().splitlines()[0]
            return int(first)
        except Exception:
            return None

    @classmethod
    def _acquire_lock_blocking(
        cls,
        lock_path: Path,
        *,
        timeout_sec: float = 5.0,
        stale_after_sec: float = 60.0,
    ) -> int:
        deadline = time.time() + timeout_sec
        while True:
            fd = cls.try_lock(lock_path)
            if fd is not None:
                return fd

            should_break = False
            pid = cls._read_lock_pid(lock_path)
            if pid is not None and not cls._pid_alive(pid):
                should_break = True
            else:
                with contextlib.suppress(Exception):
                    age = time.time() - lock_path.stat().st_mtime
                    if age > stale_after_sec:
                        should_break = True

            if should_break:
                with contextlib.suppress(Exception):
                    lock_path.unlink(missing_ok=True)
                continue

            if time.time() >= deadline:
                raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
            time.sleep(0.05)

    @classmethod
    def update_state(
        cls,
        directory: Path,
        *,
        updates: Dict[str, Any],
        expected: Optional[Dict[str, Any]] = None,
    ) -> bool:
        lock_path = directory / cls.STATE_LOCK
        fd: Optional[int] = None
        try:
            fd = cls._acquire_lock_blocking(lock_path)
            current = cls.read_state(directory)
            if expected:
                for key, expected_value in expected.items():
                    if current.get(key) != expected_value:
                        return False

            current.update(updates)
            current["updated_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")

            state_path = cls.get_state_path(directory)
            tmp_path = state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(current, indent=2))
            os.replace(tmp_path, state_path)
            return True
        finally:
            cls.release_lock(fd, lock_path)

    @classmethod
    def write_state(cls, directory: Path, **updates: Any) -> None:
        """Update state file atomically."""
        cls.update_state(directory, updates=dict(updates))

    @classmethod
    def _utcnow(cls) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    @classmethod
    def _parse_time(cls, value: Any) -> Optional[datetime.datetime]:
        if not isinstance(value, str) or not value:
            return None
        try:
            dt = datetime.datetime.fromisoformat(value)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)

    @classmethod
    def new_lease(cls, *, lease_duration_sec: float) -> Dict[str, Any]:
        now = cls._utcnow()
        expires = now + datetime.timedelta(seconds=float(lease_duration_sec))
        return {
            "lease_duration_sec": float(lease_duration_sec),
            "last_heartbeat_at": now.isoformat(timespec="seconds"),
            "lease_expires_at": expires.isoformat(timespec="seconds"),
        }

    @classmethod
    def lease_time_remaining_sec(cls, state: Dict[str, Any]) -> Optional[float]:
        expires = cls._parse_time(state.get("lease_expires_at"))
        if expires is None:
            return None
        return (expires - cls._utcnow()).total_seconds()

    @classmethod
    def is_lease_expired(cls, state: Dict[str, Any]) -> bool:
        expires = cls._parse_time(state.get("lease_expires_at"))
        if expires is None:
            return False
        return cls._utcnow() >= expires

    @classmethod
    def renew_lease(
        cls, directory: Path, *, owner_id: str, lease_duration_sec: float
    ) -> bool:
        updates = cls.new_lease(lease_duration_sec=lease_duration_sec)
        updates["owner_id"] = owner_id
        return cls.update_state(
            directory,
            expected={"status": "running", "owner_id": owner_id},
            updates=updates,
        )

    @classmethod
    def reconcile_expired_running(
        cls,
        directory: Path,
        *,
        cancelled_status: str = "cancelled",
        reason: str = "lease_expired",
    ) -> bool:
        lock_path = directory / cls.STATE_LOCK
        fd: Optional[int] = None
        try:
            fd = cls._acquire_lock_blocking(lock_path)
            state = cls.read_state(directory)
            if state.get("status") != "running":
                return False

            if not cls.is_lease_expired(state):
                return False

            now = cls._utcnow().isoformat(timespec="seconds")
            state.update(
                {
                    "status": cancelled_status,
                    "reason": reason,
                    "ended_at": now,
                    "updated_at": now,
                }
            )

            state_path = cls.get_state_path(directory)
            tmp_path = state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state, indent=2))
            os.replace(tmp_path, state_path)
            return True
        finally:
            cls.release_lock(fd, lock_path)

    @classmethod
    def is_stale(cls, directory: Path, timeout: float) -> bool:
        """Check if state file is stale based on modification time."""
        try:
            mtime = cls.get_state_path(directory).stat().st_mtime
            return (time.time() - mtime) > timeout
        except FileNotFoundError:
            return True

    @classmethod
    def try_lock(cls, lock_path: Path) -> Optional[int]:
        """Try to acquire lock, return file descriptor or None."""
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except FileExistsError:
            return None

    @classmethod
    def release_lock(cls, fd: Optional[int], lock_path: Path) -> None:
        """Release lock and clean up lock file."""
        with contextlib.suppress(Exception):
            if fd is not None:
                os.close(fd)
        with contextlib.suppress(Exception):
            lock_path.unlink(missing_ok=True)
