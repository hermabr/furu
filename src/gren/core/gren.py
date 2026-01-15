import datetime
import getpass
import inspect
import os
import signal
import socket
import sys
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Self, TypedDict, TypeVar, cast, overload

import chz
import submitit
from typing_extensions import dataclass_transform

from ..adapters import SubmititAdapter
from ..adapters.submitit import SubmititJob
from ..config import GREN_CONFIG
from ..errors import (
    MISSING,
    GrenComputeError,
    GrenLockNotAcquired,
    GrenWaitTimeout,
)
from ..runtime import current_holder
from ..runtime.logging import enter_holder, get_logger, log, write_separator
from ..runtime.tracebacks import format_traceback
from ..serialization import GrenSerializer
from ..serialization.serializer import JsonValue
from ..storage import GrenMetadata, MetadataManager, StateManager
from ..storage.state import (
    _GrenState,
    _StateAttemptFailed,
    _StateAttemptQueued,
    _StateAttemptRunning,
    _StateResultAbsent,
    _StateResultFailed,
    _StateResultSuccess,
    compute_lock,
)


class _SubmititEnvInfo(TypedDict, total=False):
    """Environment info collected for submitit jobs."""

    backend: str
    slurm_job_id: str | None
    pid: int
    host: str
    user: str
    started_at: str
    command: str


class _CallerInfo(TypedDict, total=False):
    """Caller location info for logging."""

    gren_caller_file: str
    gren_caller_line: int


@dataclass_transform(
    field_specifiers=(chz.field,), kw_only_default=True, frozen_default=True
)
class Gren[T](ABC):
    """
    Base class for cached computations with provenance tracking.

    Subclasses must implement:
    - _create(self) -> T
    - _load(self) -> T
    """

    MISSING = MISSING

    # Configuration (can be overridden in subclasses)
    version_controlled: bool = False

    # Maximum time to wait for result (seconds). Default: 10 minutes.
    _max_wait_time_sec: float = 600.0

    def __init_subclass__(
        cls,
        *,
        version_controlled: bool | None = None,
        version: str | None = None,
        typecheck: bool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "Gren" and cls.__module__ == __name__:
            return

        # Python 3.14+ may not populate `__annotations__` in `cls.__dict__` (PEP 649).
        # `chz` expects annotations to exist for every `chz.field()` attribute, so we
        # materialize them and (as a last resort) fill missing ones with `Any`.
        try:
            annotations = dict(getattr(cls, "__annotations__", {}) or {})
        except Exception:
            annotations = {}

        try:
            materialized = inspect.get_annotations(cls, eval_str=False)
        except TypeError:  # pragma: no cover
            materialized = inspect.get_annotations(cls)
        except Exception:
            materialized = {}

        if materialized:
            annotations.update(materialized)

        FieldType: type[object] | None
        try:
            from chz.data_model import Field as _ChzField  # type: ignore
        except Exception:  # pragma: no cover
            FieldType = None
        else:
            FieldType = _ChzField
        if FieldType is not None:
            for field_name, value in cls.__dict__.items():
                if isinstance(value, FieldType) and field_name not in annotations:
                    annotations[field_name] = Any

        if annotations:
            type.__setattr__(cls, "__annotations__", annotations)

        chz_kwargs: dict[str, str | bool] = {}
        if version is not None:
            chz_kwargs["version"] = version
        if typecheck is not None:
            chz_kwargs["typecheck"] = typecheck
        chz.chz(cls, **chz_kwargs)

        if version_controlled is not None:
            setattr(cls, "version_controlled", version_controlled)

    @classmethod
    def _namespace(cls) -> Path:
        module = getattr(cls, "__module__", None)
        qualname = getattr(cls, "__qualname__", cls.__name__)
        if not module or module == "__main__":
            raise ValueError(
                "Cannot derive Gren namespace from __main__; define the class in an importable module."
            )
        if "<locals>" in qualname:
            raise ValueError(
                "Cannot derive Gren namespace for a local class; define it at module scope."
            )
        return Path(*module.split("."), *qualname.split("."))

    @abstractmethod
    def _create(self: Self) -> T:
        """Compute and save the result (implement in subclass)."""
        raise NotImplementedError(
            f"{self.__class__.__name__}._create() not implemented"
        )

    @abstractmethod
    def _load(self: Self) -> T:
        """Load the result from disk (implement in subclass)."""
        raise NotImplementedError(f"{self.__class__.__name__}._load() not implemented")

    def _validate(self: Self) -> bool:
        """Validate that result is complete and correct (override if needed)."""
        return True

    def _invalidate_cached_success(self: Self, directory: Path, *, reason: str) -> None:
        logger = get_logger()
        logger.warning(
            "invalidate %s %s %s (%s)",
            self.__class__.__name__,
            self._gren_hash,
            directory,
            reason,
        )

        StateManager.get_success_marker_path(directory).unlink(missing_ok=True)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

        def mutate(state: _GrenState) -> None:
            state.result = _StateResultAbsent(status="absent")

        StateManager.update_state(directory, mutate)
        StateManager.append_event(
            directory, {"type": "result_invalidated", "reason": reason, "at": now}
        )

    @property
    def _gren_hash(self: Self) -> str:
        """Compute hash of this object's content for storage identification."""
        return GrenSerializer.compute_hash(self)

    def _force_recompute(self: Self) -> bool:
        if not GREN_CONFIG.force_recompute:
            return False
        qualname = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        return qualname in GREN_CONFIG.force_recompute

    @property
    def gren_dir(self: Self) -> Path:
        """Get the directory for this Gren object."""
        root = GREN_CONFIG.get_root(self.version_controlled)
        return root / self.__class__._namespace() / self._gren_hash

    @property
    def raw_dir(self: Self) -> Path:
        """
        Get the raw directory for Gren.

        This is intended for large, non-versioned byproducts or inputs.
        """
        return GREN_CONFIG.raw_dir

    def to_dict(self: Self) -> JsonValue:
        """Convert to dictionary."""
        return GrenSerializer.to_dict(self)

    @classmethod
    def from_dict(cls, data: JsonValue) -> "Gren":
        """Reconstruct from dictionary."""
        return GrenSerializer.from_dict(data)

    def to_python(self: Self, multiline: bool = True) -> str:
        """Convert to Python code."""
        return GrenSerializer.to_python(self, multiline=multiline)

    def log(self: Self, message: str, *, level: str = "INFO") -> Path:
        """Log a message to the current holder's `gren.log`."""
        return log(message, level=level)

    def exists(self: Self) -> bool:
        """Check if result exists and is valid."""
        logger = get_logger()
        directory = self.gren_dir
        state = StateManager.read_state(directory)

        if not isinstance(state.result, _StateResultSuccess):
            logger.info("exists %s -> false", directory)
            return False

        ok = self._validate()
        logger.info("exists %s -> %s", directory, "true" if ok else "false")
        return ok

    def get_metadata(self: Self) -> "GrenMetadata":
        """Get metadata for this object."""
        return MetadataManager.read_metadata(self.gren_dir)

    @overload
    def load_or_create(self, executor: submitit.Executor) -> T | submitit.Job[T]: ...

    @overload
    def load_or_create(self, executor: None = None) -> T: ...

    def load_or_create(
        self: Self,
        executor: submitit.Executor | None = None,
    ) -> T | submitit.Job[T]:
        """
        Load result if it exists, computing if necessary.

        Args:
            executor: Optional executor for batch submission (e.g., submitit.Executor)

        Returns:
            Result if wait=True, job handle if wait=False, or None if already exists

        Raises:
            GrenComputeError: If computation fails with detailed error information
        """
        logger = get_logger()
        parent_holder = current_holder()
        has_parent = parent_holder is not None and parent_holder is not self
        if has_parent:
            logger.debug(
                "dep: begin %s %s %s",
                self.__class__.__name__,
                self._gren_hash,
                self.gren_dir,
            )

        ok = False
        try:
            with enter_holder(self):
                start_time = time.time()
                directory = self.gren_dir
                directory.mkdir(parents=True, exist_ok=True)

                # Optimistic read: if state is already good, we don't need to reconcile (write lock)
                # Optimization: Check for success marker first to avoid reading state.json
                # This is much faster for cache hits (11x speedup on check).
                success_marker = StateManager.get_success_marker_path(directory)
                if success_marker.is_file():
                    # We have a success marker. Check if we can use it.
                    if self._force_recompute():
                        self._invalidate_cached_success(
                            directory, reason="force_recompute enabled"
                        )
                        # Fall through to normal load
                    else:
                        try:
                            if not self._validate():
                                self._invalidate_cached_success(
                                    directory, reason="_validate returned false"
                                )
                                # Fall through
                            else:
                                # Valid success! Return immediately.
                                # Since we didn't read state, we skip the logging below for speed
                                # or we can log a minimal message if needed.
                                return self._load()
                        except Exception as e:
                            self._invalidate_cached_success(
                                directory,
                                reason=f"_validate raised {type(e).__name__}: {e}",
                            )
                            # Fall through

                state0 = StateManager.read_state(directory)

                needs_reconcile = True
                if isinstance(state0.result, _StateResultSuccess):
                    # Double check logic if we fell through to here (e.g. race condition or invalidation above)
                    if self._force_recompute():
                        self._invalidate_cached_success(
                            directory, reason="force_recompute enabled"
                        )
                        state0 = StateManager.read_state(directory)
                    else:
                        try:
                            if not self._validate():
                                self._invalidate_cached_success(
                                    directory, reason="_validate returned false"
                                )
                                state0 = StateManager.read_state(directory)
                            else:
                                # Valid success found, skip reconcile
                                needs_reconcile = False
                        except Exception as e:
                            self._invalidate_cached_success(
                                directory,
                                reason=f"_validate raised {type(e).__name__}: {e}",
                            )
                            state0 = StateManager.read_state(directory)

                if needs_reconcile and executor is not None:
                    adapter0 = SubmititAdapter(executor)
                    self._reconcile(directory, adapter=adapter0)
                    state0 = StateManager.read_state(directory)

                attempt0 = state0.attempt
                if isinstance(state0.result, _StateResultSuccess):
                    decision = "success->load"
                    action_color = "green"
                elif isinstance(attempt0, (_StateAttemptQueued, _StateAttemptRunning)):
                    decision = f"{attempt0.status}->wait"
                    action_color = "yellow"
                else:
                    decision = "create"
                    action_color = "blue"

                # Cache hits can be extremely noisy in pipelines; keep logs for state
                # transitions (create/wait) and error cases, but suppress repeated
                # "success->load" lines and the raw separator on successful loads.
                caller_frame = inspect.currentframe()
                caller_info: _CallerInfo = {}
                if caller_frame is not None:
                    # Walk up the stack to find the caller outside of gren package
                    gren_pkg_dir = str(Path(__file__).parent.parent)
                    frame = caller_frame.f_back
                    while frame is not None:
                        filename = frame.f_code.co_filename
                        # Skip frames from within the gren package
                        if not filename.startswith(gren_pkg_dir):
                            caller_info = {
                                "gren_caller_file": filename,
                                "gren_caller_line": frame.f_lineno,
                            }
                            break
                        frame = frame.f_back

                logger.info(
                    "load_or_create %s %s",
                    self.__class__.__name__,
                    self._gren_hash,
                    extra={
                        "gren_console_only": True,
                        "gren_action_color": action_color,
                        **caller_info,
                    },
                )
                if decision != "success->load":
                    write_separator()
                    logger.debug(
                        "load_or_create %s %s %s (%s)",
                        self.__class__.__name__,
                        self._gren_hash,
                        directory,
                        decision,
                        extra={"gren_action_color": action_color},
                    )

                # Fast path: already successful
                state_now = StateManager.read_state(directory)
                if isinstance(state_now.result, _StateResultSuccess):
                    try:
                        result = self._load()
                        ok = True
                        return result
                    except Exception as e:
                        # Ensure there is still a clear marker in logs for unexpected
                        # failures even when we suppressed the cache-hit header line.
                        write_separator()
                        logger.error(
                            "load_or_create %s %s (load failed)",
                            self.__class__.__name__,
                            self._gren_hash,
                        )
                        raise GrenComputeError(
                            f"Failed to load result from {directory}",
                            StateManager.get_state_path(directory),
                            e,
                        ) from e

                # Synchronous execution
                if executor is None:
                    status, created_here, result = self._run_locally(
                        start_time=start_time
                    )
                    if status == "success":
                        ok = True
                        if created_here:
                            logger.debug(
                                "load_or_create: %s created -> return",
                                self.__class__.__name__,
                            )
                            return cast(T, result)
                        logger.debug(
                            "load_or_create: %s success -> _load()",
                            self.__class__.__name__,
                        )
                        return self._load()

                    state = StateManager.read_state(directory)
                    attempt = state.attempt
                    message = (
                        attempt.error.message
                        if isinstance(attempt, _StateAttemptFailed)
                        else None
                    )
                    suffix = (
                        f": {message}" if isinstance(message, str) and message else ""
                    )
                    raise GrenComputeError(
                        f"Computation {status}{suffix}",
                        StateManager.get_state_path(directory),
                    )

                # Asynchronous execution with submitit
                (submitit_folder := self.gren_dir / "submitit").mkdir(
                    exist_ok=True, parents=True
                )
                executor.folder = submitit_folder
                adapter = SubmititAdapter(executor)

                logger.debug(
                    "load_or_create: %s -> submitit submit_once()",
                    self.__class__.__name__,
                )
                job = self._submit_once(adapter, directory, None)
                ok = True
                return cast(submitit.Job[T], job)
        finally:
            if has_parent:
                logger.debug(
                    "dep: end %s %s (%s)",
                    self.__class__.__name__,
                    self._gren_hash,
                    "ok" if ok else "error",
                )

    def _check_timeout(self, start_time: float) -> None:
        """Check if operation has timed out."""
        if self._max_wait_time_sec is not None:
            if time.time() - start_time > self._max_wait_time_sec:
                raise GrenWaitTimeout(
                    f"Gren operation timed out after {self._max_wait_time_sec} seconds."
                )

    def _submit_once(
        self,
        adapter: SubmititAdapter,
        directory: Path,
        on_job_id: Callable[[str], None] | None,
    ) -> SubmititJob | None:
        """Submit job once without waiting (fire-and-forget mode)."""
        logger = get_logger()
        self._reconcile(directory, adapter=adapter)
        state = StateManager.read_state(directory)
        attempt = state.attempt
        if (
            isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning))
            and attempt.backend == "submitit"
        ):
            job = adapter.load_job(directory)
            if job is not None:
                return job

        # Try to acquire submit lock
        lock_path = StateManager.get_lock_path(directory, StateManager.SUBMIT_LOCK)
        lock_fd = StateManager.try_lock(lock_path)

        if lock_fd is None:
            # Someone else is submitting, wait briefly and return their job
            logger.debug(
                "submit: waiting for submit lock %s %s %s",
                self.__class__.__name__,
                self._gren_hash,
                directory,
            )
            time.sleep(0.5)
            return adapter.load_job(directory)

        try:
            # Create metadata
            metadata = MetadataManager.create_metadata(
                self, directory, ignore_diff=GREN_CONFIG.ignore_git_diff
            )
            MetadataManager.write_metadata(metadata, directory)

            attempt_id = StateManager.start_attempt_queued(
                directory,
                backend="submitit",
                lease_duration_sec=GREN_CONFIG.lease_duration_sec,
                owner=MetadataManager.collect_environment_info().model_dump(),
                scheduler={},
            )
            job = adapter.submit(lambda: self._worker_entry())

            # Save job handle and watch for job ID
            adapter.pickle_job(job, directory)
            adapter.watch_job_id(
                job,
                directory,
                attempt_id=attempt_id,
                callback=on_job_id,
            )

            return job
        except Exception as e:
            if "attempt_id" in locals():
                StateManager.finish_attempt_failed(
                    directory,
                    attempt_id=attempt_id,
                    error={
                        "type": type(e).__name__,
                        "message": f"Failed to submit: {e}",
                    },
                )
            else:

                def mutate(state: _GrenState) -> None:
                    state.result = _StateResultFailed(status="failed")

                StateManager.update_state(directory, mutate)
            raise GrenComputeError(
                "Failed to submit job",
                StateManager.get_state_path(directory),
                e,
            ) from e
        finally:
            StateManager.release_lock(lock_fd, lock_path)

    def _worker_entry(self: Self) -> None:
        """Entry point for worker process (called by submitit or locally)."""
        with enter_holder(self):
            logger = get_logger()
            directory = self.gren_dir
            directory.mkdir(parents=True, exist_ok=True)

            env_info = self._collect_submitit_env()

            try:
                with compute_lock(
                    directory,
                    backend="submitit",
                    lease_duration_sec=GREN_CONFIG.lease_duration_sec,
                    heartbeat_interval_sec=GREN_CONFIG.heartbeat_interval_sec,
                    owner={
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "user": getpass.getuser(),
                        "command": " ".join(sys.argv) if sys.argv else "<unknown>",
                    },
                    scheduler={
                        "backend": env_info.get("backend"),
                        "job_id": env_info.get("slurm_job_id"),
                    },
                    max_wait_time_sec=None,  # Workers wait indefinitely
                    poll_interval_sec=GREN_CONFIG.poll_interval,
                    wait_log_every_sec=GREN_CONFIG.wait_log_every_sec,
                    reconcile_fn=lambda d: self._reconcile(d),
                ) as ctx:
                    # Refresh metadata (now safe - attempt is already recorded)
                    metadata = MetadataManager.create_metadata(
                        self, directory, ignore_diff=GREN_CONFIG.ignore_git_diff
                    )
                    MetadataManager.write_metadata(metadata, directory)

                    # Set up signal handlers
                    self._setup_signal_handlers(
                        directory, ctx.stop_heartbeat, attempt_id=ctx.attempt_id
                    )

                    try:
                        # Run computation
                        logger.debug(
                            "_create: begin %s %s %s",
                            self.__class__.__name__,
                            self._gren_hash,
                            directory,
                        )
                        self._create()
                        logger.debug(
                            "_create: ok %s %s %s",
                            self.__class__.__name__,
                            self._gren_hash,
                            directory,
                        )
                        StateManager.write_success_marker(
                            directory, attempt_id=ctx.attempt_id
                        )
                        StateManager.finish_attempt_success(
                            directory, attempt_id=ctx.attempt_id
                        )
                        logger.info(
                            "_create ok %s %s",
                            self.__class__.__name__,
                            self._gren_hash,
                            extra={"gren_console_only": True},
                        )
                    except Exception as e:
                        logger.error(
                            "_create failed %s %s %s",
                            self.__class__.__name__,
                            self._gren_hash,
                            directory,
                            extra={"gren_file_only": True},
                        )
                        logger.error(
                            "%s", format_traceback(e), extra={"gren_file_only": True}
                        )

                        tb = "".join(
                            traceback.format_exception(type(e), e, e.__traceback__)
                        )
                        StateManager.finish_attempt_failed(
                            directory,
                            attempt_id=ctx.attempt_id,
                            error={
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": tb,
                            },
                        )
                        raise
            except GrenLockNotAcquired:
                # Experiment already completed (success or failed), nothing to do
                return

    def _collect_submitit_env(self: Self) -> _SubmititEnvInfo:
        """Collect submitit/slurm environment information."""
        slurm_id = os.getenv("SLURM_JOB_ID")

        info: _SubmititEnvInfo = {
            "backend": "slurm" if slurm_id else "local",
            "slurm_job_id": slurm_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command": " ".join(sys.argv) if sys.argv else "<unknown>",
        }

        # Only call submitit.JobEnvironment() when actually in a submitit job
        if slurm_id:
            env = submitit.JobEnvironment()
            info["backend"] = "submitit"
            info["slurm_job_id"] = str(getattr(env, "job_id", slurm_id))

        return info

    def _run_locally(self: Self, start_time: float) -> tuple[str, bool, T | None]:
        """Run computation locally, returning (status, created_here, result)."""
        logger = get_logger()
        directory = self.gren_dir

        # Calculate remaining time for the lock wait
        max_wait: float | None = None
        if self._max_wait_time_sec is not None:
            elapsed = time.time() - start_time
            max_wait = max(0.0, self._max_wait_time_sec - elapsed)

        try:
            with compute_lock(
                directory,
                backend="local",
                lease_duration_sec=GREN_CONFIG.lease_duration_sec,
                heartbeat_interval_sec=GREN_CONFIG.heartbeat_interval_sec,
                owner={
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "user": getpass.getuser(),
                    "command": " ".join(sys.argv) if sys.argv else "<unknown>",
                },
                scheduler={},
                max_wait_time_sec=max_wait,
                poll_interval_sec=GREN_CONFIG.poll_interval,
                wait_log_every_sec=GREN_CONFIG.wait_log_every_sec,
                reconcile_fn=lambda d: self._reconcile(d),
            ) as ctx:
                # Create metadata (now safe - attempt is already recorded)
                try:
                    metadata = MetadataManager.create_metadata(
                        self, directory, ignore_diff=GREN_CONFIG.ignore_git_diff
                    )
                    MetadataManager.write_metadata(metadata, directory)
                except Exception as e:
                    raise GrenComputeError(
                        "Failed to create metadata",
                        StateManager.get_state_path(directory),
                        e,
                    ) from e

                # Set up preemption handler
                self._setup_signal_handlers(
                    directory, ctx.stop_heartbeat, attempt_id=ctx.attempt_id
                )

                try:
                    # Run the computation
                    logger.debug(
                        "_create: begin %s %s %s",
                        self.__class__.__name__,
                        self._gren_hash,
                        directory,
                    )
                    result = self._create()
                    logger.debug(
                        "_create: ok %s %s %s",
                        self.__class__.__name__,
                        self._gren_hash,
                        directory,
                    )
                    StateManager.write_success_marker(
                        directory, attempt_id=ctx.attempt_id
                    )
                    StateManager.finish_attempt_success(
                        directory, attempt_id=ctx.attempt_id
                    )
                    logger.info(
                        "_create ok %s %s",
                        self.__class__.__name__,
                        self._gren_hash,
                        extra={"gren_console_only": True},
                    )
                    return "success", True, result
                except Exception as e:
                    logger.error(
                        "_create failed %s %s %s",
                        self.__class__.__name__,
                        self._gren_hash,
                        directory,
                        extra={"gren_file_only": True},
                    )
                    logger.error(
                        "%s", format_traceback(e), extra={"gren_file_only": True}
                    )

                    # Record failure (plain text in file)
                    tb = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                    StateManager.finish_attempt_failed(
                        directory,
                        attempt_id=ctx.attempt_id,
                        error={
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": tb,
                        },
                    )
                    raise
        except GrenLockNotAcquired:
            # Lock couldn't be acquired because experiment already completed
            state = StateManager.read_state(directory)
            if isinstance(state.result, _StateResultSuccess):
                return "success", False, None
            if isinstance(state.result, _StateResultFailed):
                return "failed", False, None
            # Shouldn't happen, but re-raise if state is unexpected
            raise

    def _reconcile(
        self: Self, directory: Path, *, adapter: SubmititAdapter | None = None
    ) -> None:
        if adapter is None:
            StateManager.reconcile(directory)
            return

        StateManager.reconcile(
            directory,
            submitit_probe=lambda state: adapter.probe(directory, state),
        )

    def _setup_signal_handlers(
        self,
        directory: Path,
        stop_heartbeat: Callable[[], None],
        *,
        attempt_id: str,
    ) -> None:
        """Set up signal handlers for graceful preemption."""

        def handle_signal(signum: int, frame: FrameType | None) -> None:
            try:
                StateManager.finish_attempt_preempted(
                    directory,
                    attempt_id=attempt_id,
                    error={"type": "signal", "message": f"signal:{signum}"},
                )
            finally:
                stop_heartbeat()
                exit_code = 143 if signum == signal.SIGTERM else 130
                os._exit(exit_code)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handle_signal)


_H = TypeVar("_H", bound=Gren, covariant=True)
