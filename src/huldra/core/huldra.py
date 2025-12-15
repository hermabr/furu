import contextlib
import datetime
import getpass
import inspect
import os
import signal
import socket
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Self, TypeVar, cast, overload

import chz
import submitit
from typing_extensions import dataclass_transform

from ..adapters import SubmititAdapter
from ..config import HULDRA_CONFIG
from ..errors import HuldraComputeError, HuldraWaitTimeout, MISSING
from ..runtime import _print_colored_traceback
from ..runtime.logging import enter_holder, get_logger, log
from ..serialization import HuldraSerializer
from ..storage import MetadataManager, StateManager


@dataclass_transform(
    field_specifiers=(chz.field,), kw_only_default=True, frozen_default=True
)
class Huldra[T](ABC):
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
        version: Any | None = None,
        typecheck: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "Huldra" and cls.__module__ == __name__:
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

        FieldType: type[Any] | None
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

        chz_kwargs: dict[str, Any] = {}
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
                "Cannot derive Huldra namespace from __main__; define the class in an importable module."
            )
        if "<locals>" in qualname:
            raise ValueError(
                "Cannot derive Huldra namespace for a local class; define it at module scope."
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

    @property
    def hexdigest(self: Self) -> str:
        """Compute hash of this object."""
        return HuldraSerializer.compute_hash(self)

    @property
    def huldra_dir(self: Self) -> Path:
        """Get the directory for this Huldra object."""
        root = HULDRA_CONFIG.get_root(self.version_controlled)
        return root / self.__class__._namespace() / self.hexdigest

    @property
    def raw_dir(self: Self) -> Path:
        """
        Get the raw directory for this Huldra object.

        This is intended for large, non-versioned byproducts or inputs you want
        colocated with the object but separated from the main artifact directory.
        """
        return HULDRA_CONFIG.raw_dir / self.__class__._namespace() / self.hexdigest

    def to_dict(self: Self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return HuldraSerializer.to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Huldra":
        """Reconstruct from dictionary."""
        return HuldraSerializer.from_dict(data)

    def to_python(self: Self, multiline: bool = True) -> str:
        """Convert to Python code."""
        return HuldraSerializer.to_python(self, multiline=multiline)

    def log(self: Self, message: str, *, level: str = "INFO") -> Path:
        """Log a message to the current holder's `huldra.log`."""
        return log(message, level=level)

    def exists(self: Self) -> bool:
        """Check if result exists and is valid."""
        directory = self.huldra_dir
        state = StateManager.read_state(directory)

        if state.get("status") != "success":
            return False

        try:
            return self._validate()
        except Exception:
            return False

    def get_metadata(self: Self) -> Dict[str, Any]:
        """Get metadata for this object."""
        return MetadataManager.read_metadata(self.huldra_dir)

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
            HuldraComputeError: If computation fails with detailed error information
        """
        with enter_holder(self):
            logger = get_logger()
            start_time = time.time()
            directory = self.huldra_dir
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "load_or_create: enter %s digest=%s dir=%s",
                self.__class__.__name__,
                self.hexdigest,
                directory,
            )

            # Fast path: already successful
            if StateManager.read_state(directory).get("status") == "success":
                try:
                    logger.debug(
                        "load_or_create: %s digest=%s -> _load()",
                        self.__class__.__name__,
                        self.hexdigest,
                    )
                    return self._load()
                except Exception as e:
                    raise HuldraComputeError(
                        f"Failed to load result from {directory}",
                        StateManager.get_state_path(directory),
                        e,
                    ) from e

            # Synchronous execution
            if executor is None:
                try:
                    status, created_here, result = self._run_locally(
                        start_time=start_time
                    )
                    if status == "success":
                        if created_here:
                            logger.debug(
                                "load_or_create: %s digest=%s created -> return",
                                self.__class__.__name__,
                                self.hexdigest,
                            )
                            return cast(T, result)
                        logger.debug(
                            "load_or_create: %s digest=%s success -> _load()",
                            self.__class__.__name__,
                            self.hexdigest,
                        )
                        return self._load()

                    state = StateManager.read_state(directory)
                    raise HuldraComputeError(
                        f"Computation {status}: {state.get('reason', 'unknown error')}",
                        StateManager.get_state_path(directory),
                    )
                except HuldraComputeError:
                    raise
                except Exception as e:
                    raise HuldraComputeError(
                        "Unexpected error during computation",
                        StateManager.get_state_path(directory),
                        e,
                    ) from e

            # Asynchronous execution with submitit
            (submitit_folder := self.huldra_dir / "submitit").mkdir(
                exist_ok=True, parents=True
            )
            executor.folder = submitit_folder
            adapter = SubmititAdapter(executor)

            logger.debug(
                "load_or_create: %s digest=%s -> submitit submit_once()",
                self.__class__.__name__,
                self.hexdigest,
            )
            return self._submit_once(adapter, directory, None)  # ty: ignore[invalid-return-type] # TODO: fix typing here

    def _check_timeout(self, start_time: float) -> None:
        """Check if operation has timed out."""
        if self._max_wait_time_sec is not None:
            if time.time() - start_time > self._max_wait_time_sec:
                raise HuldraWaitTimeout(
                    f"Huldra operation timed out after {self._max_wait_time_sec} seconds."
                )

    def _submit_once(
        self,
        adapter: SubmititAdapter,
        directory: Path,
        on_job_id: Optional[Callable[[Any], None]],
    ) -> Optional[Any]:
        """Submit job once without waiting (fire-and-forget mode)."""
        state = StateManager.read_state(directory)

        # If already queued or running, return existing job
        if state.get("status") in {"queued", "running"}:
            return adapter.load_job(directory)

        # Try to acquire submit lock
        lock_path = directory / StateManager.SUBMIT_LOCK
        lock_fd = StateManager.try_lock(lock_path)

        if lock_fd is None:
            # Someone else is submitting, wait briefly and return their job
            print("waiting on lock")
            time.sleep(0.5)
            return adapter.load_job(directory)

        try:
            # Create metadata
            metadata = MetadataManager.create_metadata(
                self, directory, ignore_diff=HULDRA_CONFIG.ignore_git_diff
            )
            MetadataManager.write_metadata(metadata, directory)

            # Submit job
            StateManager.write_state(directory, status="queued")
            job = adapter.submit(lambda: self._worker_entry())

            # Save job handle and watch for job ID
            adapter.pickle_job(job, directory)
            adapter.watch_job_id(job, directory, on_job_id)

            return job
        except Exception as e:
            StateManager.write_state(
                directory, status="failed", reason=f"Failed to submit: {e}"
            )
            raise HuldraComputeError(
                "Failed to submit job",
                StateManager.get_state_path(directory),
                e,
            ) from e
        finally:
            StateManager.release_lock(lock_fd, lock_path)

    def _submit_and_wait_with_retries(
        self,
        adapter: SubmititAdapter,
        directory: Path,
        on_job_id: Optional[Callable[[Any], None]],
        max_requeues: int,
        start_time: float,
    ) -> T:
        """Submit job and wait, resubmitting on preemption up to max_requeues times."""
        logger = get_logger()
        attempts_left = max_requeues

        while True:
            self._check_timeout(start_time)
            state = StateManager.read_state(directory)
            status = state.get("status")

            # Success - load and return result
            if status == "success":
                try:
                    logger.debug(
                        "load_or_create: %s digest=%s submitit success -> _load()",
                        self.__class__.__name__,
                        self.hexdigest,
                    )
                    return self._load()
                except Exception as e:
                    raise HuldraComputeError(
                        f"Failed to load result from {directory}",
                        StateManager.get_state_path(directory),
                        e,
                    ) from e

            # Already queued/running - wait for it
            if status in {"queued", "running"} and not StateManager.is_stale(
                directory, HULDRA_CONFIG.stale_timeout
            ):
                job = adapter.load_job(directory)
                if job:
                    # Calculate remaining time for wait
                    remaining = None
                    if self._max_wait_time_sec is not None:
                        remaining = max(
                            0.1,
                            self._max_wait_time_sec - (time.time() - start_time),
                        )

                    # We have the job handle, wait for it
                    adapter.wait(job, timeout=remaining)

                    # Check scheduler state
                    scheduler_state = adapter.get_state(job)
                    final_status = adapter.classify_scheduler_state(scheduler_state)

                    if final_status:
                        StateManager.write_state(directory, status=final_status)
                else:
                    # No job handle, just poll
                    print("no job handle, just poll")
                    time.sleep(HULDRA_CONFIG.poll_interval)
                continue

            # Stale running state - break the lock
            if status == "running" and StateManager.is_stale(
                directory, HULDRA_CONFIG.stale_timeout
            ):
                with contextlib.suppress(Exception):
                    (directory / StateManager.COMPUTE_LOCK).unlink()

            # Need to submit (or resubmit)
            try:
                status = self._submit_and_wait_once(
                    adapter, directory, on_job_id, start_time
                )
            except Exception as e:
                raise HuldraComputeError(
                    "Failed during submit and wait",
                    StateManager.get_state_path(directory),
                    e,
                ) from e

            if status == "success":
                try:
                    return self._load()
                except Exception as e:
                    raise HuldraComputeError(
                        f"Failed to load result from {directory}",
                        StateManager.get_state_path(directory),
                        e,
                    ) from e

            # Preempted - retry if we have attempts left
            if status == "preempted" and attempts_left > 0:
                attempts_left -= 1
                print(f"Job preempted, resubmitting ({attempts_left} attempts left)...")
                continue

            # Failed or out of retries
            state = StateManager.read_state(directory)
            raise HuldraComputeError(
                f"Computation {status}: {state.get('reason', 'unknown error')}",
                StateManager.get_state_path(directory),
            )

    def _submit_and_wait_once(
        self,
        adapter: SubmititAdapter,
        directory: Path,
        on_job_id: Optional[Callable[[Any], None]],
        start_time: float,
    ) -> str:
        """Submit job and wait for completion once (no retries)."""
        state = StateManager.read_state(directory)

        # Only submit if not already queued/running
        if state.get("status") not in {"queued", "running"}:
            lock_path = directory / StateManager.SUBMIT_LOCK
            lock_fd = StateManager.try_lock(lock_path)

            if lock_fd is not None:
                try:
                    # Create metadata
                    metadata = MetadataManager.create_metadata(
                        self, directory, ignore_diff=HULDRA_CONFIG.ignore_git_diff
                    )
                    MetadataManager.write_metadata(metadata, directory)

                    # Submit job
                    StateManager.write_state(directory, status="queued")
                    job = adapter.submit(lambda: self._worker_entry())

                    # Save job handle and watch for job ID
                    adapter.pickle_job(job, directory)
                    adapter.watch_job_id(job, directory, on_job_id)
                finally:
                    StateManager.release_lock(lock_fd, lock_path)
            else:
                # Someone else submitted, wait briefly
                print("someone else submitted. waiting")
                time.sleep(0.5)

        # Wait for completion
        job = adapter.load_job(directory)
        if job:
            # Calculate remaining time for wait
            remaining = None
            if self._max_wait_time_sec is not None:
                remaining = max(
                    0.1, self._max_wait_time_sec - (time.time() - start_time)
                )

            adapter.wait(job, timeout=remaining)

            # Get final state from scheduler
            scheduler_state = adapter.get_state(job)
            final_status = adapter.classify_scheduler_state(scheduler_state)

            if final_status:
                StateManager.write_state(directory, status=final_status)
                return final_status

        # Poll until done
        while True:
            self._check_timeout(start_time)
            state = StateManager.read_state(directory)
            status = state.get("status")

            if status in {"success", "failed", "preempted"}:
                return cast(str, status)

            print("poll until done")
            time.sleep(HULDRA_CONFIG.poll_interval)

    def _worker_entry(self: Self) -> None:
        """Entry point for worker process (called by submitit or locally)."""
        with enter_holder(self):
            logger = get_logger()
            directory = self.huldra_dir
            directory.mkdir(parents=True, exist_ok=True)

            # Try to acquire compute lock
            lock_path = directory / StateManager.COMPUTE_LOCK
            lock_fd = StateManager.try_lock(lock_path)

            if lock_fd is None:
                # Someone else is computing, wait for them
                while True:
                    state = StateManager.read_state(directory)
                    status = state.get("status")

                    if status in {"success", "failed", "preempted"}:
                        return

                    print("someone else is computing. waiting for them")
                    time.sleep(HULDRA_CONFIG.poll_interval)

            try:
                # Collect submitit environment info
                env_info = self._collect_submitit_env()

                # Refresh metadata
                metadata = MetadataManager.create_metadata(
                    self, directory, ignore_diff=HULDRA_CONFIG.ignore_git_diff
                )
                MetadataManager.write_metadata(metadata, directory)

                # Update state to running
                StateManager.write_state(directory, status="running", **env_info)

                # Start heartbeat
                stop_heartbeat = self._start_heartbeat(directory)

                # Set up signal handlers
                self._setup_signal_handlers(directory, stop_heartbeat)

                try:
                    # Run computation
                    logger.debug(
                        "_create: begin %s digest=%s dir=%s",
                        self.__class__.__name__,
                        self.hexdigest,
                        directory,
                    )
                    self._create()
                    logger.debug(
                        "_create: ok %s digest=%s dir=%s",
                        self.__class__.__name__,
                        self.hexdigest,
                        directory,
                    )
                    StateManager.write_state(directory, status="success")
                except Exception as e:
                    # Always show a full, colored traceback on stderr
                    _print_colored_traceback(e)

                    # Check if we were preempted
                    current_state = StateManager.read_state(directory)
                    if current_state.get("status") != "preempted":
                        tb = "".join(
                            traceback.format_exception(type(e), e, e.__traceback__)
                        )
                        StateManager.write_state(
                            directory, status="failed", reason=str(e), traceback=tb
                        )
                    raise
                finally:
                    stop_heartbeat()
            finally:
                StateManager.release_lock(lock_fd, lock_path)

    def _collect_submitit_env(self: Self) -> Dict[str, Any]:
        """Collect submitit/slurm environment information."""
        info = {
            "backend": "local",
            "slurm_job_id": None,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command": " ".join(sys.argv) if sys.argv else "<unknown>",
        }

        # Try to get SLURM job ID from environment
        slurm_id = os.getenv("SLURM_JOB_ID")
        if slurm_id:
            info["backend"] = "slurm"
            info["slurm_job_id"] = slurm_id

        # Try to use submitit if available
        try:
            import submitit

            try:
                env = submitit.JobEnvironment()
                info["backend"] = "submitit"
                info["slurm_job_id"] = str(getattr(env, "job_id", slurm_id))
            except Exception:
                pass
        except ImportError:
            pass

        return info

    def _run_locally(self: Self, start_time: float) -> tuple[str, bool, T | None]:
        """Run computation locally, returning (status, created_here, result)."""
        logger = get_logger()
        directory = self.huldra_dir
        lock_path = directory / StateManager.COMPUTE_LOCK

        # Try to acquire compute lock
        lock_fd = StateManager.try_lock(lock_path)
        if lock_fd is None:
            # Someone else is computing, wait for them
            while True:
                self._check_timeout(start_time)
                state = StateManager.read_state(directory)
                status = state.get("status")

                if status in {"success", "failed", "preempted"}:
                    return cast(str, status), False, None

                if status == "running" and StateManager.is_stale(
                    directory, HULDRA_CONFIG.stale_timeout
                ):
                    # Stale lock, break it
                    with contextlib.suppress(Exception):
                        lock_path.unlink()
                    break

                print("waiting for", directory)
                time.sleep(HULDRA_CONFIG.poll_interval)

        try:
            # Create metadata
            try:
                metadata = MetadataManager.create_metadata(
                    self, directory, ignore_diff=HULDRA_CONFIG.ignore_git_diff
                )
                MetadataManager.write_metadata(metadata, directory)
            except Exception as e:
                raise HuldraComputeError(
                    "Failed to create metadata",
                    StateManager.get_state_path(directory),
                    e,
                ) from e

            # Update state to running
            StateManager.write_state(
                directory,
                status="running",
                backend="local",
                **MetadataManager.collect_environment_info(),
            )

            # Start heartbeat
            stop_heartbeat = self._start_heartbeat(directory)

            # Set up preemption handler
            self._setup_signal_handlers(directory, stop_heartbeat)

            try:
                # Run the computation
                logger.debug(
                    "_create: begin %s digest=%s dir=%s",
                    self.__class__.__name__,
                    self.hexdigest,
                    directory,
                )
                result = self._create()
                logger.debug(
                    "_create: ok %s digest=%s dir=%s",
                    self.__class__.__name__,
                    self.hexdigest,
                    directory,
                )
                StateManager.write_state(directory, status="success")
                return "success", True, result
            except Exception as e:
                # If it failed, always print a colored traceback
                _print_colored_traceback(e)

                # Check if we were preempted
                current_state = StateManager.read_state(directory)
                if current_state.get("status") == "preempted":
                    return "preempted", False, None

                # Record failure (plain text in file)
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                StateManager.write_state(
                    directory, status="failed", reason=str(e), traceback=tb
                )
                return "failed", False, None
            finally:
                stop_heartbeat()
        finally:
            StateManager.release_lock(lock_fd, lock_path)

    def _start_heartbeat(self: Self, directory: Path) -> Callable[[], None]:
        """Start heartbeat thread, return stop function."""
        stop_event = threading.Event()

        def heartbeat():
            while not stop_event.wait(HULDRA_CONFIG.poll_interval / 2):
                with contextlib.suppress(Exception):
                    StateManager.write_state(directory)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        return stop_event.set

    def _setup_signal_handlers(
        self, directory: Path, stop_heartbeat: Callable[[], None]
    ) -> None:
        """Set up signal handlers for graceful preemption."""

        def handle_signal(signum: int, frame: Any) -> None:
            try:
                StateManager.write_state(
                    directory, status="preempted", reason=f"signal:{signum}"
                )
            finally:
                stop_heartbeat()
                exit_code = 143 if signum == signal.SIGTERM else 130
                os._exit(exit_code)

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(Exception):
                signal.signal(sig, handle_signal)


_H = TypeVar("_H", bound=Huldra, covariant=True)
