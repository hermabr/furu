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
    def write_state(cls, directory: Path, **updates: Any) -> None:
        """Update state file atomically."""
        current = cls.read_state(directory)
        current.update(updates)
        current["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )

        state_path = cls.get_state_path(directory)
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(current, indent=2))
        os.replace(tmp_path, state_path)

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

