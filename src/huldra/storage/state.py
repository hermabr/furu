import contextlib
import datetime as _dt
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, cast


class StateManager:
    """
    Crash-safe state and liveness management for a single Huldra artifact directory.

    Design principles:
    - Only `result.status == "success"` is treated as loadable by default.
    - `attempt.status == "running"` is a lease-based claim that must be reconcilable.
    - Writes are atomic (`os.replace`) and serialized via a state lock.
    """

    SCHEMA_VERSION = 1

    STATE_FILE = "state.json"
    EVENTS_FILE = "events.jsonl"
    SUCCESS_MARKER = "SUCCESS.json"

    COMPUTE_LOCK = ".compute.lock"
    SUBMIT_LOCK = ".submit.lock"
    STATE_LOCK = ".state.lock"

    RUNNING_STATUSES = {"queued", "running"}
    TERMINAL_STATUSES = {
        "success",
        "failed",
        "cancelled",
        "preempted",
        "crashed",
    }

    @classmethod
    def get_state_path(cls, directory: Path) -> Path:
        return directory / cls.STATE_FILE

    @classmethod
    def _utcnow(cls) -> _dt.datetime:
        return _dt.datetime.now(_dt.timezone.utc)

    @classmethod
    def _iso_now(cls) -> str:
        return cls._utcnow().isoformat(timespec="seconds")

    @classmethod
    def _parse_time(cls, value: Any) -> Optional[_dt.datetime]:
        if not isinstance(value, str) or not value:
            return None
        try:
            dt = _dt.datetime.fromisoformat(value)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc)

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "result": {"status": "absent"},
            "attempt": None,
        }

    @classmethod
    def read_state(cls, directory: Path) -> dict[str, Any]:
        try:
            data = json.loads(cls.get_state_path(directory).read_text())
        except Exception:
            return cls.default_state()

        if (
            not isinstance(data, dict)
            or data.get("schema_version") != cls.SCHEMA_VERSION
        ):
            return cls.default_state()

        if "result" not in data or not isinstance(data["result"], dict):
            data["result"] = {"status": "absent"}
        if "attempt" not in data:
            data["attempt"] = None
        return data

    @classmethod
    def _write_state_unlocked(cls, directory: Path, state: dict[str, Any]) -> None:
        state_path = cls.get_state_path(directory)
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2))
        os.replace(tmp_path, state_path)

    @classmethod
    def _pid_alive(cls, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    @classmethod
    def try_lock(cls, lock_path: Path) -> Optional[int]:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": cls._iso_now(),
                "lock_id": uuid.uuid4().hex,
            }
            os.write(fd, (json.dumps(payload) + "\n").encode())
            return fd
        except FileExistsError:
            return None

    @classmethod
    def release_lock(cls, fd: Optional[int], lock_path: Path) -> None:
        with contextlib.suppress(Exception):
            if fd is not None:
                os.close(fd)
        with contextlib.suppress(Exception):
            lock_path.unlink(missing_ok=True)

    @classmethod
    def _read_lock_info(cls, lock_path: Path) -> dict[str, Any] | None:
        try:
            first = lock_path.read_text().strip().splitlines()[0]
            data = json.loads(first)
            return data if isinstance(data, dict) else None
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
            info = cls._read_lock_info(lock_path)
            if info and info.get("host") == socket.gethostname():
                pid = info.get("pid")
                if isinstance(pid, int) and not cls._pid_alive(pid):
                    should_break = True
            if not should_break:
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
        cls, directory: Path, mutator: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        lock_path = directory / cls.STATE_LOCK
        fd: Optional[int] = None
        try:
            fd = cls._acquire_lock_blocking(lock_path)
            state = cls.read_state(directory)
            mutator(state)
            state["updated_at"] = cls._iso_now()
            cls._write_state_unlocked(directory, state)
            return state
        finally:
            cls.release_lock(fd, lock_path)

    @classmethod
    def append_event(cls, directory: Path, event: dict[str, Any]) -> None:
        path = directory / cls.EVENTS_FILE
        enriched = {
            "ts": cls._iso_now(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **event,
        }
        with contextlib.suppress(Exception):
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(enriched) + "\n")

    @classmethod
    def write_success_marker(cls, directory: Path, *, attempt_id: str) -> None:
        marker = directory / cls.SUCCESS_MARKER
        payload = {"attempt_id": attempt_id, "created_at": cls._iso_now()}
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, marker)

    @classmethod
    def success_marker_exists(cls, directory: Path) -> bool:
        return (directory / cls.SUCCESS_MARKER).is_file()

    @classmethod
    def is_success(cls, state: dict[str, Any]) -> bool:
        result = state.get("result")
        return isinstance(result, dict) and result.get("status") == "success"

    @classmethod
    def _lease_expired(cls, attempt: dict[str, Any]) -> bool:
        expires = cls._parse_time(attempt.get("lease_expires_at"))
        if expires is None:
            return True
        return cls._utcnow() >= expires

    @classmethod
    def start_attempt(
        cls,
        directory: Path,
        *,
        backend: str,
        status: str,
        lease_duration_sec: float,
        owner: dict[str, Any],
        scheduler: dict[str, Any] | None = None,
    ) -> str:
        attempt_id = uuid.uuid4().hex
        now = cls._utcnow()
        expires = now + _dt.timedelta(seconds=float(lease_duration_sec))
        prev_result_status: str | None = None
        prev_attempt_status: str | None = None
        prev_attempt_reason: str | None = None

        def mutate(state: dict[str, Any]) -> None:
            nonlocal prev_result_status, prev_attempt_status, prev_attempt_reason
            prev_result = state.get("result")
            if isinstance(prev_result, dict):
                prev_result_status = cast(Optional[str], prev_result.get("status"))

            prev = (
                state.get("attempt") if isinstance(state.get("attempt"), dict) else None
            )
            if prev is not None:
                prev_attempt_status = cast(Optional[str], prev.get("status"))
                prev_attempt_reason = cast(Optional[str], prev.get("reason"))

            number = int(prev.get("number", 0) + 1) if prev else 1
            state["attempt"] = {
                "id": attempt_id,
                "number": number,
                "backend": backend,
                "status": status,
                "started_at": now.isoformat(timespec="seconds"),
                "heartbeat_at": now.isoformat(timespec="seconds"),
                "lease_duration_sec": float(lease_duration_sec),
                "lease_expires_at": expires.isoformat(timespec="seconds"),
                "owner": owner,
                "scheduler": scheduler or {},
                "error": None,
            }
            state["result"] = {"status": "incomplete"}

        state = cls.update_state(directory, mutate)
        if status == "running":
            from ..runtime.logging import get_logger

            logger = get_logger()
            if prev_result_status == "failed":
                logger.warning(
                    "state: retrying after previous failure %s",
                    directory,
                )
            elif prev_attempt_status == "crashed" and prev_attempt_reason in {
                "pid_dead",
                "lease_expired",
            }:
                logger.warning(
                    "state: restarting after stale attempt (%s) %s",
                    prev_attempt_reason,
                    directory,
                )

        cls.append_event(
            directory,
            {
                "type": "attempt_started",
                "attempt_id": attempt_id,
                "backend": backend,
                "status": status,
            },
        )
        return cast(str, state["attempt"]["id"])  # type: ignore[no-any-return]

    @classmethod
    def heartbeat(
        cls, directory: Path, *, attempt_id: str, lease_duration_sec: float
    ) -> bool:
        ok = False

        def mutate(state: dict[str, Any]) -> None:
            nonlocal ok
            attempt = state.get("attempt")
            if not isinstance(attempt, dict):
                return
            if attempt.get("id") != attempt_id:
                return
            if attempt.get("status") != "running":
                return
            now = cls._utcnow()
            expires = now + _dt.timedelta(seconds=float(lease_duration_sec))
            attempt["heartbeat_at"] = now.isoformat(timespec="seconds")
            attempt["lease_duration_sec"] = float(lease_duration_sec)
            attempt["lease_expires_at"] = expires.isoformat(timespec="seconds")
            ok = True

        cls.update_state(directory, mutate)
        return ok

    @classmethod
    def set_attempt_fields(
        cls, directory: Path, *, attempt_id: str, fields: dict[str, Any]
    ) -> bool:
        ok = False

        def mutate(state: dict[str, Any]) -> None:
            nonlocal ok
            attempt = state.get("attempt")
            if not isinstance(attempt, dict) or attempt.get("id") != attempt_id:
                return
            attempt.update(fields)
            ok = True

        cls.update_state(directory, mutate)
        return ok

    @classmethod
    def finish_attempt_success(cls, directory: Path, *, attempt_id: str) -> None:
        now = cls._iso_now()

        def mutate(state: dict[str, Any]) -> None:
            attempt = state.get("attempt")
            if isinstance(attempt, dict) and attempt.get("id") == attempt_id:
                attempt["status"] = "success"
                attempt["ended_at"] = now
            state["result"] = {"status": "success", "created_at": now}

        cls.update_state(directory, mutate)
        cls.append_event(
            directory,
            {"type": "attempt_finished", "attempt_id": attempt_id, "status": "success"},
        )

    @classmethod
    def finish_attempt_failed(
        cls,
        directory: Path,
        *,
        attempt_id: str,
        error: dict[str, Any],
        status: str = "failed",
    ) -> None:
        if status not in cls.TERMINAL_STATUSES:
            status = "failed"
        now = cls._iso_now()

        def mutate(state: dict[str, Any]) -> None:
            attempt = state.get("attempt")
            if isinstance(attempt, dict) and attempt.get("id") == attempt_id:
                attempt["status"] = status
                attempt["ended_at"] = now
                attempt["error"] = error
            # `failed` is treated as non-loadable; next run can decide whether to retry.
            state["result"] = {
                "status": "failed" if status == "failed" else "incomplete"
            }

        cls.update_state(directory, mutate)
        cls.append_event(
            directory,
            {"type": "attempt_finished", "attempt_id": attempt_id, "status": status},
        )

    @classmethod
    def _local_attempt_alive(cls, attempt: dict[str, Any]) -> Optional[bool]:
        owner = attempt.get("owner")
        if not isinstance(owner, dict):
            return None
        host = owner.get("host")
        pid = owner.get("pid")
        if host != socket.gethostname():
            return None
        if not isinstance(pid, int):
            return None
        return cls._pid_alive(pid)

    @classmethod
    def reconcile(
        cls,
        directory: Path,
        *,
        submitit_probe: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Reconcile a possibly-stale running/queued attempt.

        - If a success marker exists, promote to success.
        - For local attempts, if PID is provably dead or lease expired, mark as crashed and
          remove compute lock so waiters can proceed.
        - For submitit attempts, rely on `submitit_probe` when provided; otherwise fall back
          to lease expiry.
        """

        def mutate(state: dict[str, Any]) -> None:
            attempt = state.get("attempt")
            if not isinstance(attempt, dict):
                return
            if attempt.get("status") not in cls.RUNNING_STATUSES:
                return

            # Fast promotion if we can see a durable success marker.
            if cls.success_marker_exists(directory):
                attempt["status"] = "success"
                attempt["ended_at"] = cls._iso_now()
                state["result"] = {"status": "success", "created_at": cls._iso_now()}
                return

            backend = attempt.get("backend")
            now = cls._iso_now()

            terminal_status: str | None = None
            reason: str | None = None

            if backend == "local":
                alive = cls._local_attempt_alive(attempt)
                if alive is False:
                    terminal_status = "crashed"
                    reason = "pid_dead"
                elif cls._lease_expired(attempt):
                    terminal_status = "crashed"
                    reason = "lease_expired"
            elif backend == "submitit":
                if submitit_probe is not None:
                    verdict = submitit_probe(state)
                    if verdict.get("terminal_status") in cls.TERMINAL_STATUSES:
                        terminal_status = str(verdict["terminal_status"])
                        reason = str(verdict.get("reason") or "scheduler_terminal")
                        scheduler = attempt.get("scheduler")
                        if isinstance(scheduler, dict):
                            scheduler.update(
                                {
                                    k: v
                                    for k, v in verdict.items()
                                    if k != "terminal_status"
                                }
                            )
                if terminal_status is None and cls._lease_expired(attempt):
                    terminal_status = "crashed"
                    reason = "lease_expired"
            else:
                if cls._lease_expired(attempt):
                    terminal_status = "crashed"
                    reason = "lease_expired"

            if terminal_status is None:
                return

            attempt["status"] = terminal_status
            attempt["ended_at"] = now
            attempt["error"] = attempt.get("error")
            attempt["reason"] = reason
            state["result"] = {
                "status": "incomplete" if terminal_status != "failed" else "failed"
            }

        state = cls.update_state(directory, mutate)
        attempt = state.get("attempt")
        if isinstance(attempt, dict) and attempt.get("status") in {
            "crashed",
            "cancelled",
            "preempted",
        }:
            with contextlib.suppress(Exception):
                (directory / cls.COMPUTE_LOCK).unlink(missing_ok=True)
        return state
