import contextlib
import datetime as _dt
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _StateResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    status: str


class _StateResultAbsent(_StateResultBase):
    status: Literal["absent"]


class _StateResultIncomplete(_StateResultBase):
    status: Literal["incomplete"]


class _StateResultSuccess(_StateResultBase):
    status: Literal["success"]
    created_at: str


class _StateResultFailed(_StateResultBase):
    status: Literal["failed"]


_StateResult = Annotated[
    _StateResultAbsent
    | _StateResultIncomplete
    | _StateResultSuccess
    | _StateResultFailed,
    Field(discriminator="status"),
]


def _coerce_result(current: _StateResult, **updates: Any) -> _StateResult:
    data = current.model_dump(mode="json")
    data.update(updates)
    status = data.get("status")
    match status:
        case "absent":
            return _StateResultAbsent(status="absent")
        case "incomplete":
            return _StateResultIncomplete(status="incomplete")
        case "success":
            created_at = data.get("created_at")
            if not isinstance(created_at, str) or not created_at:
                raise ValueError("Success result requires created_at")
            return _StateResultSuccess(status="success", created_at=created_at)
        case "failed":
            return _StateResultFailed(status="failed")
        case _:
            raise ValueError(f"Invalid result status: {status!r}")


class _StateOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    pid: int | None = None
    host: str | None = None
    hostname: str | None = None
    user: str | None = None
    command: str | None = None
    timestamp: str | None = None
    python_version: str | None = None
    executable: str | None = None
    platform: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_host_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        host = data.get("host")
        hostname = data.get("hostname")
        if host is None and hostname is not None:
            data = dict(data)
            data["host"] = hostname
            return data
        if hostname is None and host is not None:
            data = dict(data)
            data["hostname"] = host
        return data


class _HuldraErrorState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    type: str = "UnknownError"
    message: str = ""
    traceback: str | None = None


class _StateAttemptBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    id: str
    number: int = 1
    backend: str
    status: str
    started_at: str
    heartbeat_at: str
    lease_duration_sec: float
    lease_expires_at: str
    owner: _StateOwner
    scheduler: dict[str, Any] = Field(default_factory=dict)


class _StateAttemptQueued(_StateAttemptBase):
    status: Literal["queued"] = "queued"


class _StateAttemptRunning(_StateAttemptBase):
    status: Literal["running"] = "running"


class _StateAttemptSuccess(_StateAttemptBase):
    status: Literal["success"] = "success"
    ended_at: str
    reason: None = None


class _StateAttemptFailed(_StateAttemptBase):
    status: Literal["failed"] = "failed"
    ended_at: str
    error: _HuldraErrorState
    reason: str | None = None


class _StateAttemptTerminal(_StateAttemptBase):
    status: Literal["cancelled", "preempted", "crashed"]
    ended_at: str
    error: _HuldraErrorState | None = None
    reason: str | None = None


_StateAttempt = Annotated[
    _StateAttemptQueued
    | _StateAttemptRunning
    | _StateAttemptSuccess
    | _StateAttemptFailed
    | _StateAttemptTerminal,
    Field(discriminator="status"),
]


class _HuldraState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    schema_version: int = 1
    result: _StateResult = Field(
        default_factory=lambda: _StateResultAbsent(status="absent")
    )
    attempt: _StateAttempt | None = None
    updated_at: str | None = None


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
    def default_state(cls) -> _HuldraState:
        return _HuldraState(schema_version=cls.SCHEMA_VERSION)

    @classmethod
    def read_state(cls, directory: Path) -> _HuldraState:
        state_path = cls.get_state_path(directory)
        try:
            text = state_path.read_text()
        except Exception:
            return cls.default_state()

        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(f"Invalid JSON in state file: {state_path}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Invalid state file (expected object): {state_path}")
        if data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported state schema_version (expected {cls.SCHEMA_VERSION}): {state_path}"
            )
        try:
            return _HuldraState.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Invalid state schema: {state_path}") from e

    @classmethod
    def _write_state_unlocked(cls, directory: Path, state: _HuldraState) -> None:
        state_path = cls.get_state_path(directory)
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2))
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
        cls, directory: Path, mutator: Callable[[_HuldraState], None]
    ) -> _HuldraState:
        lock_path = directory / cls.STATE_LOCK
        fd: Optional[int] = None
        try:
            fd = cls._acquire_lock_blocking(lock_path)
            state = cls.read_state(directory)
            mutator(state)
            state.schema_version = cls.SCHEMA_VERSION
            state.updated_at = cls._iso_now()
            validated = _HuldraState.model_validate(state)
            cls._write_state_unlocked(directory, validated)
            return validated
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
    def _lease_expired(
        cls, attempt: _StateAttemptQueued | _StateAttemptRunning
    ) -> bool:
        expires = cls._parse_time(attempt.lease_expires_at)
        if expires is None:
            return True
        return cls._utcnow() >= expires

    @classmethod
    def start_attempt_queued(
        cls,
        directory: Path,
        *,
        backend: str,
        lease_duration_sec: float,
        owner: dict[str, Any],
        scheduler: dict[str, Any] | None = None,
    ) -> str:
        return cls._start_attempt(
            directory,
            backend=backend,
            lease_duration_sec=lease_duration_sec,
            owner=owner,
            scheduler=scheduler,
            attempt_cls=_StateAttemptQueued,
        )

    @classmethod
    def start_attempt_running(
        cls,
        directory: Path,
        *,
        backend: str,
        lease_duration_sec: float,
        owner: dict[str, Any],
        scheduler: dict[str, Any] | None = None,
    ) -> str:
        return cls._start_attempt(
            directory,
            backend=backend,
            lease_duration_sec=lease_duration_sec,
            owner=owner,
            scheduler=scheduler,
            attempt_cls=_StateAttemptRunning,
        )

    @classmethod
    def _start_attempt(
        cls,
        directory: Path,
        *,
        backend: str,
        lease_duration_sec: float,
        owner: dict[str, Any],
        scheduler: dict[str, Any] | None,
        attempt_cls: type[_StateAttemptQueued] | type[_StateAttemptRunning],
    ) -> str:
        attempt_id = uuid.uuid4().hex
        now = cls._utcnow()
        expires = now + _dt.timedelta(seconds=float(lease_duration_sec))
        prev_result_failed = False
        prev_attempt_status: str | None = None
        prev_attempt_reason: str | None = None

        def mutate(state: _HuldraState) -> None:
            nonlocal prev_result_failed, prev_attempt_status, prev_attempt_reason
            prev_result_failed = isinstance(state.result, _StateResultFailed)
            prev = state.attempt
            if prev is not None:
                prev_attempt_status = prev.status
                prev_attempt_reason = getattr(prev, "reason", None)

            number = (prev.number + 1) if prev is not None else 1

            owner_state = _StateOwner.model_validate(owner)
            started_at = now.isoformat(timespec="seconds")
            heartbeat_at = started_at
            lease_duration = float(lease_duration_sec)
            lease_expires_at = expires.isoformat(timespec="seconds")
            scheduler_state: dict[str, Any] = scheduler or {}

            attempt_common = dict(
                id=attempt_id,
                number=int(number),
                backend=backend,
                started_at=started_at,
                heartbeat_at=heartbeat_at,
                lease_duration_sec=lease_duration,
                lease_expires_at=lease_expires_at,
                owner=owner_state,
                scheduler=scheduler_state,
            )
            state.attempt = attempt_cls(**attempt_common)

            state.result = _coerce_result(state.result, status="incomplete")

        state = cls.update_state(directory, mutate)
        if attempt_cls is _StateAttemptRunning:
            from ..runtime.logging import get_logger

            logger = get_logger()
            if prev_result_failed:
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
                "status": state.attempt.status
                if state.attempt is not None
                else "unknown",
            },
        )
        attempt = state.attempt
        if attempt is None:  # pragma: no cover
            raise RuntimeError("start_attempt did not create attempt")
        return attempt.id

    @classmethod
    def heartbeat(
        cls, directory: Path, *, attempt_id: str, lease_duration_sec: float
    ) -> bool:
        ok = False

        def mutate(state: _HuldraState) -> None:
            nonlocal ok
            attempt = state.attempt
            if not isinstance(attempt, _StateAttemptRunning):
                return
            if attempt.id != attempt_id:
                return
            now = cls._utcnow()
            expires = now + _dt.timedelta(seconds=float(lease_duration_sec))
            attempt.heartbeat_at = now.isoformat(timespec="seconds")
            attempt.lease_duration_sec = float(lease_duration_sec)
            attempt.lease_expires_at = expires.isoformat(timespec="seconds")
            ok = True

        cls.update_state(directory, mutate)
        return ok

    @classmethod
    def set_attempt_fields(
        cls, directory: Path, *, attempt_id: str, fields: dict[str, Any]
    ) -> bool:
        ok = False

        def mutate(state: _HuldraState) -> None:
            nonlocal ok
            attempt = state.attempt
            if attempt is None or attempt.id != attempt_id:
                return
            for key, value in fields.items():
                if key == "scheduler" and isinstance(value, dict):
                    attempt.scheduler.update(value)
                    continue
                if hasattr(attempt, key):
                    setattr(attempt, key, value)
            ok = True

        cls.update_state(directory, mutate)
        return ok

    @classmethod
    def finish_attempt_success(cls, directory: Path, *, attempt_id: str) -> None:
        now = cls._iso_now()

        def mutate(state: _HuldraState) -> None:
            attempt = state.attempt
            if attempt is not None and attempt.id == attempt_id:
                state.attempt = _StateAttemptSuccess(
                    id=attempt.id,
                    number=attempt.number,
                    backend=attempt.backend,
                    started_at=attempt.started_at,
                    heartbeat_at=attempt.heartbeat_at,
                    lease_duration_sec=attempt.lease_duration_sec,
                    lease_expires_at=attempt.lease_expires_at,
                    owner=attempt.owner,
                    scheduler=attempt.scheduler,
                    ended_at=now,
                )
            state.result = _coerce_result(
                state.result, status="success", created_at=now
            )

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
    ) -> None:
        now = cls._iso_now()

        error_state = _HuldraErrorState.model_validate(error)

        def mutate(state: _HuldraState) -> None:
            attempt = state.attempt
            if attempt is not None and attempt.id == attempt_id:
                state.attempt = _StateAttemptFailed(
                    id=attempt.id,
                    number=attempt.number,
                    backend=attempt.backend,
                    started_at=attempt.started_at,
                    heartbeat_at=attempt.heartbeat_at,
                    lease_duration_sec=attempt.lease_duration_sec,
                    lease_expires_at=attempt.lease_expires_at,
                    owner=attempt.owner,
                    scheduler=attempt.scheduler,
                    ended_at=now,
                    error=error_state,
                )

            state.result = _coerce_result(state.result, status="failed")

        cls.update_state(directory, mutate)
        cls.append_event(
            directory,
            {"type": "attempt_finished", "attempt_id": attempt_id, "status": "failed"},
        )

    @classmethod
    def finish_attempt_preempted(
        cls,
        directory: Path,
        *,
        attempt_id: str,
        error: dict[str, Any],
        reason: str | None = None,
    ) -> None:
        now = cls._iso_now()
        error_state = _HuldraErrorState.model_validate(error)

        def mutate(state: _HuldraState) -> None:
            attempt = state.attempt
            if attempt is not None and attempt.id == attempt_id:
                state.attempt = _StateAttemptTerminal(
                    status="preempted",
                    id=attempt.id,
                    number=attempt.number,
                    backend=attempt.backend,
                    started_at=attempt.started_at,
                    heartbeat_at=attempt.heartbeat_at,
                    lease_duration_sec=attempt.lease_duration_sec,
                    lease_expires_at=attempt.lease_expires_at,
                    owner=attempt.owner,
                    scheduler=attempt.scheduler,
                    ended_at=now,
                    error=error_state,
                    reason=reason,
                )
            state.result = _coerce_result(state.result, status="incomplete")

        cls.update_state(directory, mutate)
        cls.append_event(
            directory,
            {
                "type": "attempt_finished",
                "attempt_id": attempt_id,
                "status": "preempted",
            },
        )

    @classmethod
    def _local_attempt_alive(
        cls, attempt: _StateAttemptQueued | _StateAttemptRunning
    ) -> Optional[bool]:
        host = attempt.owner.host
        pid = attempt.owner.pid
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
        submitit_probe: Optional[Callable[[_HuldraState], dict[str, Any]]] = None,
    ) -> _HuldraState:
        """
        Reconcile a possibly-stale running/queued attempt.

        - If a success marker exists, promote to success.
        - For local attempts, if PID is provably dead or lease expired, mark as crashed and
          remove compute lock so waiters can proceed.
        - For submitit attempts, rely on `submitit_probe` when provided; otherwise fall back
          to lease expiry.
        """

        def mutate(state: _HuldraState) -> None:
            attempt = state.attempt
            if not isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning)):
                return

            # Fast promotion if we can see a durable success marker.
            if cls.success_marker_exists(directory):
                ended = cls._iso_now()
                state.attempt = _StateAttemptSuccess(
                    id=attempt.id,
                    number=attempt.number,
                    backend=attempt.backend,
                    started_at=attempt.started_at,
                    heartbeat_at=attempt.heartbeat_at,
                    lease_duration_sec=attempt.lease_duration_sec,
                    lease_expires_at=attempt.lease_expires_at,
                    owner=attempt.owner,
                    scheduler=attempt.scheduler,
                    ended_at=ended,
                )
                state.result = _coerce_result(
                    state.result, status="success", created_at=ended
                )
                return

            backend = attempt.backend
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
                        attempt.scheduler.update(
                            {k: v for k, v in verdict.items() if k != "terminal_status"}
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
            if terminal_status == "success":
                terminal_status = "crashed"
                reason = reason or "scheduler_success_no_success_marker"

            if terminal_status == "failed":
                state.attempt = _StateAttemptFailed(
                    id=attempt.id,
                    number=attempt.number,
                    backend=attempt.backend,
                    started_at=attempt.started_at,
                    heartbeat_at=attempt.heartbeat_at,
                    lease_duration_sec=attempt.lease_duration_sec,
                    lease_expires_at=attempt.lease_expires_at,
                    owner=attempt.owner,
                    scheduler=attempt.scheduler,
                    ended_at=now,
                    error=_HuldraErrorState(
                        type="HuldraComputeError", message=reason or ""
                    ),
                    reason=reason,
                )
            else:
                if terminal_status == "cancelled":
                    state.attempt = _StateAttemptTerminal(
                        status="cancelled",
                        id=attempt.id,
                        number=attempt.number,
                        backend=attempt.backend,
                        started_at=attempt.started_at,
                        heartbeat_at=attempt.heartbeat_at,
                        lease_duration_sec=attempt.lease_duration_sec,
                        lease_expires_at=attempt.lease_expires_at,
                        owner=attempt.owner,
                        scheduler=attempt.scheduler,
                        ended_at=now,
                        reason=reason,
                    )
                elif terminal_status == "preempted":
                    state.attempt = _StateAttemptTerminal(
                        status="preempted",
                        id=attempt.id,
                        number=attempt.number,
                        backend=attempt.backend,
                        started_at=attempt.started_at,
                        heartbeat_at=attempt.heartbeat_at,
                        lease_duration_sec=attempt.lease_duration_sec,
                        lease_expires_at=attempt.lease_expires_at,
                        owner=attempt.owner,
                        scheduler=attempt.scheduler,
                        ended_at=now,
                        reason=reason,
                    )
                else:
                    state.attempt = _StateAttemptTerminal(
                        status="crashed",
                        id=attempt.id,
                        number=attempt.number,
                        backend=attempt.backend,
                        started_at=attempt.started_at,
                        heartbeat_at=attempt.heartbeat_at,
                        lease_duration_sec=attempt.lease_duration_sec,
                        lease_expires_at=attempt.lease_expires_at,
                        owner=attempt.owner,
                        scheduler=attempt.scheduler,
                        ended_at=now,
                        reason=reason,
                    )

            state.result = _coerce_result(
                state.result,
                status="failed" if terminal_status == "failed" else "incomplete",
            )

        state = cls.update_state(directory, mutate)
        attempt = state.attempt
        if attempt is not None and attempt.status in {
            "crashed",
            "cancelled",
            "preempted",
        }:
            with contextlib.suppress(Exception):
                (directory / cls.COMPUTE_LOCK).unlink(missing_ok=True)
        return state
