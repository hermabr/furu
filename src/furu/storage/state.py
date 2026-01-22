import datetime as _dt
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, TypedDict, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..errors import FuruLockNotAcquired, FuruWaitTimeout


# Type alias for scheduler-specific metadata. Different schedulers (SLURM, LSF, PBS, local)
# return different fields, so this must remain dynamic.
SchedulerMetadata = dict[str, Any]

# Type alias for probe results from submitit adapter
ProbeResult = dict[str, Any]

EventValue: TypeAlias = str | int | float | bool
EventMapping: TypeAlias = Mapping[str, EventValue]


class _LockInfoDict(TypedDict, total=False):
    """TypedDict for lock file information."""

    pid: int
    host: str
    created_at: str
    lock_id: str


class _OwnerDict(TypedDict, total=False):
    """TypedDict for owner information passed to state manager functions."""

    pid: int | None
    host: str | None
    hostname: str | None
    user: str | None
    command: str | None
    timestamp: str | None
    python_version: str | None
    executable: str | None
    platform: str | None


class _ErrorDict(TypedDict, total=False):
    """TypedDict for error information passed to state manager functions."""

    type: str
    message: str
    traceback: str | None


class _StateResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)


class _StateResultAbsent(_StateResultBase):
    status: Literal["absent"] = "absent"


class _StateResultIncomplete(_StateResultBase):
    status: Literal["incomplete"] = "incomplete"


class _StateResultSuccess(_StateResultBase):
    status: Literal["success"] = "success"
    created_at: str


class _StateResultFailed(_StateResultBase):
    status: Literal["failed"] = "failed"


class _StateResultMigrated(_StateResultBase):
    status: Literal["migrated"] = "migrated"


_StateResult = Annotated[
    _StateResultAbsent
    | _StateResultIncomplete
    | _StateResultSuccess
    | _StateResultFailed
    | _StateResultMigrated,
    Field(discriminator="status"),
]


def _coerce_result(current: _StateResult, **updates: str) -> _StateResult:
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
        case "migrated":
            return _StateResultMigrated(status="migrated")

        case _:
            raise ValueError(f"Invalid result status: {status!r}")


class StateOwner(BaseModel):
    """Owner information for a Furu attempt."""

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
    def _normalize_host_keys(
        cls, data: dict[str, str | int | None] | Any
    ) -> dict[str, str | int | None] | Any:
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


class FuruErrorState(BaseModel):
    """Error state information for a Furu attempt."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    type: str = "UnknownError"
    message: str = ""
    traceback: str | None = None


class _StateAttemptBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    id: str
    number: int = 1
    backend: str
    started_at: str
    heartbeat_at: str
    lease_duration_sec: float
    lease_expires_at: str
    owner: StateOwner
    scheduler: SchedulerMetadata = Field(default_factory=dict)


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
    error: FuruErrorState
    reason: str | None = None


class _StateAttemptTerminal(_StateAttemptBase):
    status: Literal["cancelled", "preempted", "crashed"]
    ended_at: str
    error: FuruErrorState | None = None
    reason: str | None = None


_StateAttempt = Annotated[
    _StateAttemptQueued
    | _StateAttemptRunning
    | _StateAttemptSuccess
    | _StateAttemptFailed
    | _StateAttemptTerminal,
    Field(discriminator="status"),
]


class StateAttempt(BaseModel):
    """
    Public read-only representation of a Furu attempt.

    This model is used for external APIs (like the dashboard) to expose
    attempt information without coupling to internal state variants.
    All fields that may not be present on all attempt types are optional.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    number: int
    backend: str
    status: str
    started_at: str
    heartbeat_at: str
    lease_duration_sec: float
    lease_expires_at: str
    owner: StateOwner
    scheduler: SchedulerMetadata = Field(default_factory=dict)
    ended_at: str | None = None
    error: FuruErrorState | None = None
    reason: str | None = None

    @classmethod
    def from_internal(cls, attempt: _StateAttempt) -> "StateAttempt":
        """Create a StateAttempt from an internal attempt state."""
        return cls(
            id=attempt.id,
            number=attempt.number,
            backend=attempt.backend,
            status=attempt.status,
            started_at=attempt.started_at,
            heartbeat_at=attempt.heartbeat_at,
            lease_duration_sec=attempt.lease_duration_sec,
            lease_expires_at=attempt.lease_expires_at,
            owner=attempt.owner,
            scheduler=attempt.scheduler,
            ended_at=getattr(attempt, "ended_at", None),
            error=getattr(attempt, "error", None),
            reason=getattr(attempt, "reason", None),
        )


class _FuruState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    schema_version: int = 1
    result: _StateResult = Field(
        default_factory=lambda: _StateResultAbsent(status="absent")
    )
    attempt: _StateAttempt | None = None
    updated_at: str | None = None


class StateManager:
    """
    Crash-safe state and liveness management for a single Furu artifact directory.

    Design principles:
    - Only `result.status == "success"` is treated as loadable by default.
    - `attempt.status == "running"` is a lease-based claim that must be reconcilable.
    - Writes are atomic (`os.replace`) and serialized via a state lock.
    """

    SCHEMA_VERSION = 1

    INTERNAL_DIR = ".furu"

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
    def get_internal_dir(cls, directory: Path) -> Path:
        return directory / cls.INTERNAL_DIR

    @classmethod
    def get_state_path(cls, directory: Path) -> Path:
        return cls.get_internal_dir(directory) / cls.STATE_FILE

    @classmethod
    def get_events_path(cls, directory: Path) -> Path:
        return cls.get_internal_dir(directory) / cls.EVENTS_FILE

    @classmethod
    def get_success_marker_path(cls, directory: Path) -> Path:
        return cls.get_internal_dir(directory) / cls.SUCCESS_MARKER

    @classmethod
    def get_lock_path(cls, directory: Path, lock_name: str) -> Path:
        return cls.get_internal_dir(directory) / lock_name

    @classmethod
    def _utcnow(cls) -> _dt.datetime:
        return _dt.datetime.now(_dt.timezone.utc)

    @classmethod
    def _iso_now(cls) -> str:
        return cls._utcnow().isoformat(timespec="seconds")

    @classmethod
    def _parse_time(cls, value: str | None) -> _dt.datetime | None:
        if not isinstance(value, str) or not value:
            return None
        dt = _dt.datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc)

    @classmethod
    def default_state(cls) -> _FuruState:
        return _FuruState(schema_version=cls.SCHEMA_VERSION)

    @classmethod
    def read_state(cls, directory: Path) -> _FuruState:
        state_path = cls.get_state_path(directory)
        if not state_path.is_file():
            return cls.default_state()

        text = state_path.read_text()

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
            return _FuruState.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Invalid state schema: {state_path}") from e

    @classmethod
    def _write_state_unlocked(cls, directory: Path, state: _FuruState) -> None:
        state_path = cls.get_state_path(directory)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2))
        os.replace(tmp_path, state_path)

    @classmethod
    def _pid_alive(cls, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it - still alive
            return True

    @classmethod
    def try_lock(cls, lock_path: Path) -> int | None:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
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
    def release_lock(cls, fd: int | None, lock_path: Path) -> None:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)

    @classmethod
    def _read_lock_info(cls, lock_path: Path) -> _LockInfoDict | None:
        if not lock_path.is_file():
            return None
        text = lock_path.read_text().strip()
        if not text:
            return None
        lines = text.splitlines()
        if not lines:
            return None
        data = json.loads(lines[0])
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
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
                try:
                    stat_result = lock_path.stat()
                    age = time.time() - stat_result.st_mtime
                    if age > stale_after_sec:
                        should_break = True
                except FileNotFoundError:
                    # Lock file was deleted by another process, retry
                    pass

            if should_break:
                lock_path.unlink(missing_ok=True)
                continue

            if time.time() >= deadline:
                raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
            time.sleep(0.05)

    @classmethod
    def update_state(
        cls, directory: Path, mutator: Callable[[_FuruState], None]
    ) -> _FuruState:
        lock_path = cls.get_lock_path(directory, cls.STATE_LOCK)
        fd: int | None = None
        try:
            fd = cls._acquire_lock_blocking(lock_path)
            state = cls.read_state(directory)
            mutator(state)
            state.schema_version = cls.SCHEMA_VERSION
            state.updated_at = cls._iso_now()
            validated = _FuruState.model_validate(state)
            cls._write_state_unlocked(directory, validated)
            return validated
        finally:
            cls.release_lock(fd, lock_path)

    @classmethod
    def append_event(cls, directory: Path, event: EventMapping) -> None:
        path = cls.get_events_path(directory)
        enriched = {
            "ts": cls._iso_now(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **event,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(enriched) + "\n")

    @classmethod
    def write_success_marker(cls, directory: Path, *, attempt_id: str) -> None:
        marker = cls.get_success_marker_path(directory)
        marker.parent.mkdir(parents=True, exist_ok=True)
        payload = {"attempt_id": attempt_id, "created_at": cls._iso_now()}
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, marker)

    @classmethod
    def success_marker_exists(cls, directory: Path) -> bool:
        return cls.get_success_marker_path(directory).is_file()

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
        owner: _OwnerDict,
        scheduler: SchedulerMetadata | None = None,
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
        owner: _OwnerDict,
        scheduler: SchedulerMetadata | None = None,
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
        owner: _OwnerDict,
        scheduler: SchedulerMetadata | None,
        attempt_cls: type[_StateAttemptQueued] | type[_StateAttemptRunning],
    ) -> str:
        attempt_id = uuid.uuid4().hex
        now = cls._utcnow()
        expires = now + _dt.timedelta(seconds=float(lease_duration_sec))
        prev_result_failed = False
        prev_attempt_status: str | None = None
        prev_attempt_reason: str | None = None

        def mutate(state: _FuruState) -> None:
            nonlocal prev_result_failed, prev_attempt_status, prev_attempt_reason
            prev_result_failed = isinstance(state.result, _StateResultFailed)
            prev = state.attempt
            if prev is not None:
                prev_attempt_status = prev.status
                prev_attempt_reason = getattr(prev, "reason", None)

            number = (prev.number + 1) if prev is not None else 1

            owner_state = StateOwner.model_validate(owner)
            started_at = now.isoformat(timespec="seconds")
            heartbeat_at = started_at
            lease_duration = float(lease_duration_sec)
            lease_expires_at = expires.isoformat(timespec="seconds")
            scheduler_state: SchedulerMetadata = scheduler or {}

            attempt_kwargs = dict(
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
            state.attempt = attempt_cls(**attempt_kwargs)  # type: ignore[arg-type, misc]

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

        def mutate(state: _FuruState) -> None:
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
        cls, directory: Path, *, attempt_id: str, fields: SchedulerMetadata
    ) -> bool:
        ok = False

        def mutate(state: _FuruState) -> None:
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

        def mutate(state: _FuruState) -> None:
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
        error: _ErrorDict,
    ) -> None:
        now = cls._iso_now()

        error_state = FuruErrorState.model_validate(error)

        def mutate(state: _FuruState) -> None:
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
        error: _ErrorDict,
        reason: str | None = None,
    ) -> None:
        now = cls._iso_now()
        error_state = FuruErrorState.model_validate(error)

        def mutate(state: _FuruState) -> None:
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
    ) -> bool | None:
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
        submitit_probe: Callable[[_FuruState], ProbeResult] | None = None,
    ) -> _FuruState:
        """
        Reconcile a possibly-stale running/queued attempt.

        - If a success marker exists, promote to success.
        - For local attempts, if PID is provably dead or lease expired, mark as crashed and
          remove compute lock so waiters can proceed.
        - For submitit attempts, rely on `submitit_probe` when provided; otherwise fall back
          to lease expiry.
        """

        def mutate(state: _FuruState) -> None:
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
                    error=FuruErrorState(type="FuruComputeError", message=reason or ""),
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
            cls.get_lock_path(directory, cls.COMPUTE_LOCK).unlink(missing_ok=True)
        return state


@dataclass
class ComputeLockContext:
    """Context returned when a compute lock is successfully acquired."""

    attempt_id: str
    stop_heartbeat: Callable[[], None]


@contextmanager
def compute_lock(
    directory: Path,
    *,
    backend: str,
    lease_duration_sec: float,
    heartbeat_interval_sec: float,
    owner: _OwnerDict,
    scheduler: SchedulerMetadata | None = None,
    max_wait_time_sec: float | None = None,
    poll_interval_sec: float = 10.0,
    wait_log_every_sec: float = 10.0,
    reconcile_fn: Callable[[Path], None] | None = None,
    allow_failed: bool = False,
) -> Generator[ComputeLockContext, None, None]:
    """
    Context manager that atomically acquires lock + records attempt + starts heartbeat.

    This ensures there can never be a mismatch between the lock file and state:
    - Lock acquisition and attempt recording happen together
    - Heartbeat starts immediately after attempt is recorded
    - On exit, heartbeat is stopped and lock is released

    The context manager handles the wait loop internally, blocking until the lock
    is acquired or timeout is reached.

    Args:
        directory: The furu directory for this experiment
        backend: Backend type (e.g., "local", "submitit")
        lease_duration_sec: Duration of the lease in seconds
        heartbeat_interval_sec: Interval between heartbeats in seconds
        owner: Owner information (pid, host, user, etc.)
        scheduler: Optional scheduler metadata
        max_wait_time_sec: Maximum time to wait for lock (None = wait forever)
        poll_interval_sec: Interval between lock acquisition attempts
        wait_log_every_sec: Interval between "waiting for lock" log messages
        reconcile_fn: Optional function to call to reconcile stale attempts
        allow_failed: Allow recomputation even if state is failed

    Yields:
        ComputeLockContext with attempt_id and stop_heartbeat callable

    Raises:
        FuruLockNotAcquired: If lock cannot be acquired (after waiting)
        FuruWaitTimeout: If max_wait_time_sec is exceeded
    """

    def _format_wait_duration(seconds: float) -> str:
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes = seconds / 60.0
        if minutes < 60.0:
            return f"{minutes:.1f}m"
        hours = minutes / 60.0
        if hours < 24.0:
            return f"{hours:.1f}h"
        days = hours / 24.0
        return f"{days:.1f}d"

    def _format_owner(attempt: _StateAttempt) -> str:
        owner = attempt.owner
        parts: list[str] = []
        if attempt.id:
            parts.append(f"attempt {attempt.id}")
        if owner.host:
            parts.append(f"host {owner.host}")
        if owner.pid is not None:
            parts.append(f"pid {owner.pid}")
        if owner.user:
            parts.append(f"user {owner.user}")
        if not parts:
            return "owner unknown"
        return ", ".join(parts)

    def _describe_wait(attempt: _StateAttempt, waited_sec: float) -> str:
        label = "last heartbeat"
        timestamp = attempt.heartbeat_at
        if attempt.status == "queued":
            label = "queued at"
            timestamp = attempt.started_at
        parsed = StateManager._parse_time(timestamp)
        timestamp_info = timestamp
        if parsed is not None:
            age = (StateManager._utcnow() - parsed).total_seconds()
            timestamp_info = f"{timestamp} ({_format_wait_duration(age)} ago)"
        return (
            "waited "
            f"{_format_wait_duration(waited_sec)}, {label} {timestamp_info}, "
            f"status {attempt.status}, backend {attempt.backend}, {_format_owner(attempt)}"
        )

    lock_path = StateManager.get_lock_path(directory, StateManager.COMPUTE_LOCK)

    lock_fd: int | None = None
    start_time = time.time()
    next_wait_log_at = 0.0

    # Import here to avoid circular import
    from ..runtime import get_logger

    logger = get_logger()

    # Wait loop to acquire lock
    while lock_fd is None:
        # Check timeout
        if max_wait_time_sec is not None:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time_sec:
                state = StateManager.read_state(directory)
                attempt = state.attempt
                attempt_info = "no active attempt"
                if isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning)):
                    attempt_info = _describe_wait(attempt, elapsed)
                message = (
                    f"Timed out waiting for compute lock after {elapsed:.1f}s."
                    f"\nDirectory: {directory}"
                    f"\nLock file: {lock_path}"
                    f"\nDetails: {attempt_info}"
                )
                raise FuruWaitTimeout(
                    message,
                    hints=[
                        "Increase max wait: set FURU_MAX_WAIT_SECS (or override Furu._max_wait_time_sec).",
                        "Change poll cadence: set FURU_POLL_INTERVAL_SECS.",
                        "Change wait logging cadence: set FURU_WAIT_LOG_EVERY_SECS.",
                        "If locks look stale too quickly/slowly: tune FURU_LEASE_SECS and FURU_HEARTBEAT_SECS.",
                        "For more logs: set FURU_LOG_LEVEL=DEBUG.",
                    ],
                )

        lock_fd = StateManager.try_lock(lock_path)
        if lock_fd is not None:
            state = StateManager.read_state(directory)
            if isinstance(state.result, _StateResultSuccess):
                StateManager.release_lock(lock_fd, lock_path)
                raise FuruLockNotAcquired(
                    "Cannot acquire lock: experiment already succeeded"
                )
            if isinstance(state.result, _StateResultFailed) and not allow_failed:
                StateManager.release_lock(lock_fd, lock_path)
                raise FuruLockNotAcquired(
                    "Cannot acquire lock: experiment already failed"
                )
            attempt = state.attempt
            if (
                isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning))
                and attempt.backend != backend
            ):
                StateManager.release_lock(lock_fd, lock_path)
                lock_fd = None
                if reconcile_fn is not None:
                    reconcile_fn(directory)
                    state = StateManager.read_state(directory)
                if isinstance(state.result, _StateResultSuccess):
                    raise FuruLockNotAcquired(
                        "Cannot acquire lock: experiment already succeeded"
                    )
                if isinstance(state.result, _StateResultFailed) and not allow_failed:
                    raise FuruLockNotAcquired(
                        "Cannot acquire lock: experiment already failed"
                    )
                attempt = state.attempt
                if not isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning)):
                    continue
                if attempt.backend == backend:
                    continue
                now = time.time()
                if now >= next_wait_log_at:
                    waited_sec = now - start_time
                    logger.info(
                        "compute_lock: waiting for lock creation %s (%s)",
                        directory,
                        _describe_wait(attempt, waited_sec),
                    )
                    next_wait_log_at = now + wait_log_every_sec
                time.sleep(poll_interval_sec)
                continue
            break

        # Lock held by someone else - reconcile and check state
        if reconcile_fn is not None:
            reconcile_fn(directory)

        state = StateManager.read_state(directory)
        attempt = state.attempt

        # If result is terminal, no point waiting
        if isinstance(state.result, _StateResultSuccess):
            raise FuruLockNotAcquired(
                "Cannot acquire lock: experiment already succeeded"
            )
        if isinstance(state.result, _StateResultFailed) and not allow_failed:
            raise FuruLockNotAcquired("Cannot acquire lock: experiment already failed")

        # If no active attempt but lock exists, it's orphaned - clean it up
        if attempt is None or isinstance(
            attempt,
            (
                _StateAttemptSuccess,
                _StateAttemptFailed,
                _StateAttemptTerminal,
            ),
        ):
            # Orphaned lock file - remove it and retry immediately
            lock_path.unlink(missing_ok=True)
            continue

        # Active attempt exists - wait for it
        now = time.time()
        if now >= next_wait_log_at:
            waited_sec = now - start_time
            logger.info(
                "compute_lock: waiting for lock %s (%s)",
                directory,
                _describe_wait(attempt, waited_sec),
            )
            next_wait_log_at = now + wait_log_every_sec
        time.sleep(poll_interval_sec)

    # Lock acquired - now atomically record attempt and start heartbeat
    stop_event = threading.Event()
    attempt_id: str | None = None

    try:
        # Record attempt IMMEDIATELY to minimize orphan window
        attempt_id = StateManager.start_attempt_running(
            directory,
            backend=backend,
            lease_duration_sec=lease_duration_sec,
            owner=owner,
            scheduler=scheduler,
        )

        # Start heartbeat IMMEDIATELY
        def heartbeat() -> None:
            while not stop_event.wait(heartbeat_interval_sec):
                StateManager.heartbeat(
                    directory,
                    attempt_id=attempt_id,  # type: ignore[arg-type]
                    lease_duration_sec=lease_duration_sec,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()

        yield ComputeLockContext(
            attempt_id=attempt_id,
            stop_heartbeat=stop_event.set,
        )
    finally:
        # Always stop heartbeat and release lock
        stop_event.set()
        StateManager.release_lock(lock_fd, lock_path)
