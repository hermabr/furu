import contextlib
import datetime
import enum
import getpass
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import platform
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    TypeVar,
    cast,
    overload,
)

import chz
import submitit
from typing_extensions import dataclass_transform

def _get_pydantic_base_model() -> type[Any] | None:
    try:
        module = importlib.import_module("pydantic")
    except Exception:
        return None
    base_model = getattr(module, "BaseModel", None)
    if isinstance(base_model, type):
        return cast(type[Any], base_model)
    return None


BaseModel = _get_pydantic_base_model()

# =============================================================================
# Load .env file
# =============================================================================

try:
    from dotenv import load_dotenv

    # Load .env file from current directory or parent directories
    load_dotenv()
except ImportError:
    # python-dotenv not available, skip .env loading
    pass

# =============================================================================
# Pretty Tracebacks
# =============================================================================


def _print_colored_traceback(exc: BaseException) -> None:
    """
    Print a full, colored traceback to stderr.
    Uses 'rich' if available; otherwise ANSI-colored fallback.
    """
    try:
        from rich.console import Console
        from rich.traceback import Traceback

        console = Console(stderr=True)
        tb = Traceback.from_exception(
            type(exc),
            exc,
            exc.__traceback__,
            show_locals=False,  # flip True if you want locals
            width=None,  # auto width
            extra_lines=3,  # a bit more context
            theme="monokai",  # pick your fave; 'ansi_dark' is nice too
            word_wrap=False,
        )
        console.print(tb)
        return
    except Exception:
        # Fallback: ANSI colorize the standard traceback
        import sys
        import traceback as _tb

        RED = "\033[31m"
        CYAN = "\033[36m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        lines = _tb.format_exception(type(exc), exc, exc.__traceback__)
        # Light-touch color: header in red bold; 'File ...' lines in cyan
        out = []
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("Traceback"):
                out.append(f"{BOLD}{RED}{line.rstrip()}{RESET}\n")
            elif line.lstrip().startswith('File "'):
                out.append(f"{CYAN}{line.rstrip()}{RESET}\n")
            else:
                out.append(line)
        sys.stderr.write("".join(out))


# Optionally, auto-enable Rich for any *uncaught* exceptions too.
# (Caught exceptions in your code paths will still use _print_colored_traceback.)
try:
    from rich.traceback import install as _rich_install  # type: ignore

    _rich_install(show_locals=False)
except Exception:
    pass

# =============================================================================
# Configuration & Constants
# =============================================================================


class HuldraConfig:
    """Central configuration for Huldra behavior."""

    def __init__(self: Self):
        def _get_base_root() -> Path:
            env = os.getenv("HULDRA_PATH")
            if env:
                return Path(env).expanduser().resolve()
            return Path("data-huldra").resolve()

        self.base_root = _get_base_root()
        self.poll_interval = float(os.getenv("HULDRA_POLL_INTERVAL_SECS", "10"))
        self.stale_timeout = float(os.getenv("HULDRA_STALE_AFTER_SECS", str(30 * 60)))
        self.max_requeues = int(os.getenv("HULDRA_PREEMPT_MAX", "5"))
        self.ignore_git_diff = os.getenv("HULDRA_IGNORE_DIFF", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.cancelled_is_preempted = os.getenv(
            "HULDRA_CANCELLED_IS_PREEMPTED", "false"
        ).lower() in {"1", "true", "yes"}

    def get_root(self: Self, version_controlled: bool = False) -> Path:
        """Get root directory for storage (version_controlled determines subdirectory)."""
        if version_controlled:
            return self.base_root / "git"
        return self.base_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.base_root / "raw"


HULDRA_CONFIG = HuldraConfig()


# =============================================================================
# Special Sentinels
# =============================================================================


class _HuldraMissing:
    """Sentinel value for missing fields."""

    __slots__ = ()

    def __repr__(self: Self) -> str:
        return "Huldra.MISSING"


MISSING = _HuldraMissing()


# =============================================================================
# Exceptions
# =============================================================================


class HuldraError(Exception):
    """Base exception for Huldra errors."""

    pass


class HuldraWaitTimeout(HuldraError):
    """Raised when waiting for a result exceeds _max_wait_time_sec."""

    pass


class HuldraComputeError(HuldraError):
    """Raised when computation fails."""

    def __init__(
        self,
        message: str,
        state_path: Path,
        original_error: Optional[Exception] = None,
    ):
        self.state_path = state_path
        self.original_error = original_error
        super().__init__(message)

    def __str__(self: Self) -> str:
        msg = super().__str__()  # ty: ignore[invalid-super-argument]
        if self.original_error:
            msg += f"\n\nOriginal error: {self.original_error}"
            if hasattr(self.original_error, "__traceback__"):
                tb = "".join(
                    traceback.format_exception(
                        type(self.original_error),
                        self.original_error,
                        self.original_error.__traceback__,
                    )
                )
                msg += f"\n\nTraceback:\n{tb}"
        msg += f"\n\nState file: {self.state_path}"
        return msg


# =============================================================================
# Serialization & Hashing
# =============================================================================


class HuldraSerializer:
    """Handles serialization, deserialization, and hashing of Huldra objects."""

    CLASS_MARKER = "__class__"

    @staticmethod
    def get_classname(obj: Any) -> str:
        """Get fully qualified class name."""
        classname = obj.__class__.__module__
        if classname == "__main__":
            raise ValueError("Cannot serialize objects from __main__ module")

        if isinstance(obj, enum.Enum):
            return f"{classname}.{obj.__class__.__qualname__}:{obj.name}"
        return f"{classname}.{obj.__class__.__qualname__}"

    @classmethod
    def to_dict(cls, obj: Any) -> Any:
        """Convert object to JSON-serializable dictionary."""
        if isinstance(obj, _HuldraMissing):
            raise ValueError("Cannot serialize Huldra.MISSING")

        if chz.is_chz(obj):
            result = {cls.CLASS_MARKER: cls.get_classname(obj)}
            for field_name in chz.chz_fields(obj):
                result[field_name] = cls.to_dict(getattr(obj, field_name))
            return result

        if isinstance(obj, pathlib.Path):
            return str(obj)

        if isinstance(obj, (list, tuple)):
            return [cls.to_dict(v) for v in obj]

        if isinstance(obj, dict):
            return {k: cls.to_dict(v) for k, v in obj.items()}

        return obj

    @classmethod
    def from_dict(cls, data: Any) -> Any:
        """Reconstruct object from dictionary."""
        if isinstance(data, dict) and cls.CLASS_MARKER in data:
            module_path, _, class_name = data[cls.CLASS_MARKER].rpartition(".")
            data_class = getattr(importlib.import_module(module_path), class_name)

            kwargs = {
                k: cls.from_dict(v) for k, v in data.items() if k != cls.CLASS_MARKER
            }

            path_types = (Path, pathlib.Path)

            if chz.is_chz(data_class):
                for name, field in chz.chz_fields(data_class).items():
                    if field.final_type in path_types and isinstance(
                        kwargs.get(name), str
                    ):
                        kwargs[name] = pathlib.Path(kwargs[name])
            return data_class(**kwargs)

        if isinstance(data, list):
            return [cls.from_dict(v) for v in data]

        if isinstance(data, dict):
            return {k: cls.from_dict(v) for k, v in data.items()}

        return data

    @classmethod
    def compute_hash(cls, obj: Any, verbose: bool = False) -> str:
        """Compute deterministic hash of object."""

        def canonicalize(item: Any) -> Any:
            if isinstance(item, _HuldraMissing):
                raise ValueError("Cannot hash Huldra.MISSING")

            if chz.is_chz(item):
                fields = chz.chz_fields(item)
                return {
                    "__class__": cls.get_classname(item),
                    **{
                        name: canonicalize(getattr(item, name))
                        for name in fields
                        if not name.startswith("_")  # <--- Added filter
                    },
                }

            if isinstance(item, dict):
                return {k: canonicalize(v) for k, v in sorted(item.items())}

            if isinstance(item, (list, tuple)):
                return [canonicalize(v) for v in item]

            if isinstance(item, Path):
                return str(item)

            if isinstance(item, enum.Enum):
                return {"__enum__": cls.get_classname(item)}

            if isinstance(item, (set, frozenset)):
                return sorted(canonicalize(v) for v in item)

            if isinstance(item, (bytes, bytearray, memoryview)):
                return {"__bytes__": hashlib.sha256(item).hexdigest()}

            if isinstance(item, datetime.datetime):
                return item.astimezone(datetime.timezone.utc).isoformat(
                    timespec="microseconds"
                )

            if isinstance(item, (str, int, float, bool)) or item is None:
                return item

            if BaseModel is not None and isinstance(item, BaseModel):
                return {
                    "__class__": cls.get_classname(item),
                    **{k: canonicalize(v) for k, v in item.model_dump().items()},
                }

            raise TypeError(f"Cannot hash type: {type(item)}")

        canonical = canonicalize(obj)
        json_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))

        if verbose:
            print(json_str)

        return hashlib.blake2s(json_str.encode(), digest_size=10).hexdigest()

    @classmethod
    def to_python(cls, obj: Any, multiline: bool = True) -> str:
        """Convert object to Python code representation."""

        def to_py_recursive(item: Any, indent: int = 0) -> str:
            if isinstance(item, _HuldraMissing):
                raise ValueError("Cannot convert Huldra.MISSING to Python")

            pad = "" if not multiline else " " * indent
            next_indent = indent + (4 if multiline else 0)

            if chz.is_chz(item):
                cls_path = cls.get_classname(item)
                fields = [
                    f"{name}={to_py_recursive(getattr(item, name), next_indent)}"
                    for name in chz.chz_fields(item)
                ]

                if multiline:
                    inner = (",\n" + " " * next_indent).join(fields)
                    return f"{cls_path}(\n{pad}    {inner}\n{pad})"
                return f"{cls_path}({', '.join(fields)})"

            if isinstance(item, enum.Enum):
                return cls.get_classname(item)

            if isinstance(item, pathlib.Path):
                return f"pathlib.Path({str(item)!r})"

            if isinstance(item, datetime.datetime):
                iso = item.astimezone(datetime.timezone.utc).isoformat(
                    timespec="microseconds"
                )
                return f"datetime.datetime.fromisoformat({iso!r})"

            if isinstance(item, (bytes, bytearray, memoryview)):
                hex_str = hashlib.sha256(item).hexdigest()
                return f"bytes.fromhex({hex_str!r})"

            if isinstance(item, list):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                return f"[{items}]"

            if isinstance(item, tuple):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                comma = "," if len(item) == 1 else ""
                return f"({items}{comma})"

            if isinstance(item, set):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                return f"{{{items}}}"

            if isinstance(item, frozenset):
                items = ", ".join(to_py_recursive(v, next_indent) for v in item)
                return f"frozenset({{{items}}})"

            if isinstance(item, dict):
                kv_pairs = [
                    f"{to_py_recursive(k, next_indent)}: {to_py_recursive(v, next_indent)}"
                    for k, v in item.items()
                ]

                if multiline:
                    joined = (",\n" + " " * (indent + 4)).join(kv_pairs)
                    return f"{{\n{pad}    {joined}\n{pad}}}"
                else:
                    return "{" + ", ".join(kv_pairs) + "}"

            return repr(item)

        result = to_py_recursive(obj, indent=0)
        if multiline:
            result = textwrap.dedent(result).strip()
        return result


# =============================================================================
# Submitit Adapter
# =============================================================================


class SubmititAdapter:
    """Adapter for working with submitit executors."""

    JOB_PICKLE = "job.pkl"

    def __init__(self: Self, executor: Any):
        self.executor = executor

    def submit(self: Self, fn: Callable[[], None]) -> Any:
        """Submit a job to the executor."""
        return self.executor.submit(fn)

    def wait(self: Self, job: Any, timeout: Optional[float] = None) -> None:
        """Wait for job completion."""
        with contextlib.suppress(Exception):
            if timeout:
                job.result(timeout=timeout)
            else:
                job.wait()

    def get_job_id(self: Self, job: Any) -> Optional[str]:
        """Get job ID if available."""
        job_id = getattr(job, "job_id", None)
        if job_id:
            return str(job_id)
        return None

    def is_done(self: Self, job: Any) -> bool:
        """Check if job is done."""
        done_fn = getattr(job, "done", None)
        if done_fn and callable(done_fn):
            return done_fn()
        return False

    def get_state(self: Self, job: Any) -> Optional[str]:
        """Get job state from scheduler."""
        try:
            state_fn = getattr(job, "state", None)
            if state_fn and callable(state_fn):
                return state_fn()
        except Exception:
            pass
        return None

    def pickle_job(self: Self, job: Any, directory: Path) -> None:
        """Save job handle to disk."""
        pkl_path = directory / self.JOB_PICKLE
        try:
            import cloudpickle as pkl
        except ImportError:
            import pickle as pkl

        with open(pkl_path, "wb") as f:
            pkl.dump(job, f)

    def load_job(self: Self, directory: Path) -> Optional[Any]:
        """Load job handle from disk."""
        pkl_path = directory / self.JOB_PICKLE
        if not pkl_path.exists():
            return None

        try:
            import cloudpickle as pkl
        except ImportError:
            import pickle as pkl

        try:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)
        except Exception:
            return None

    def watch_job_id(
        self,
        job: Any,
        directory: Path,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Watch for job ID in background thread and update state."""

        def watcher():
            while True:
                job_id = self.get_job_id(job)
                if job_id:
                    StateManager.write_state(directory, job_id=job_id)
                    if callback:
                        with contextlib.suppress(Exception):
                            callback(job_id)
                    break

                if self.is_done(job):
                    break

                print("watching job id", job)
                time.sleep(0.5)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()

    def classify_scheduler_state(self: Self, state: Optional[str]) -> Optional[str]:
        """Map scheduler state to Huldra status."""
        if not state:
            return None

        s = state.upper()

        if "COMPLETE" in s or "COMPLETED" in s:
            return "success"

        if s in {
            "PREEMPTED",
            "TIMEOUT",
            "NODE_FAIL",
            "REQUEUED",
            "REQUEUE_HOLD",
        }:
            return "preempted"

        if s == "CANCELLED":
            return "preempted" if HULDRA_CONFIG.cancelled_is_preempted else "failed"

        if "FAIL" in s or "ERROR" in s:
            return "failed"

        return None


# =============================================================================
# State Management
# =============================================================================


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


# =============================================================================
# Metadata Management
# =============================================================================


class MetadataManager:
    """Handles metadata collection and storage."""

    @staticmethod
    def safe_git_command(args: List[str]) -> str:
        """Run git command safely, return output or error message."""
        try:
            proc = subprocess.run(
                ["git", *args], text=True, capture_output=True, timeout=10
            )
            if proc.returncode not in (0, 1):
                proc.check_returncode()
            return proc.stdout.strip()
        except Exception as exc:
            print(f"WARNING: git {' '.join(args)} failed: {exc}", file=sys.stderr)
            return "<unavailable>"

    @classmethod
    def collect_git_info(cls, ignore_diff: bool = False) -> Dict[str, Any]:
        """Collect git repository information."""
        head = cls.safe_git_command(["rev-parse", "HEAD"])
        branch = cls.safe_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        remote = cls.safe_git_command(["remote", "get-url", "origin"])

        if ignore_diff:
            patch = "<ignored-diff>"
        else:
            unstaged = cls.safe_git_command(["diff"])
            staged = cls.safe_git_command(["diff", "--cached"])
            untracked = cls.safe_git_command(
                ["ls-files", "--others", "--exclude-standard"]
            ).splitlines()

            untracked_patches = "\n".join(
                cls.safe_git_command(["diff", "--no-index", "/dev/null", f])
                for f in untracked
            )

            patch = "\n".join(
                filter(
                    None,
                    [
                        "# === unstaged ==================================================",
                        unstaged,
                        "# === staged ====================================================",
                        staged,
                        "# === untracked ================================================",
                        untracked_patches,
                    ],
                )
            )

            if len(patch) > 50_000:
                raise ValueError(
                    f"Git diff too large ({len(patch):,} bytes). "
                    "Use ignore_diff=True or HULDRA_IGNORE_DIFF=1"
                )

        submodules = {}
        for line in cls.safe_git_command(["submodule", "status"]).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                submodules[parts[1]] = parts[0]

        return {
            "git_commit": head,
            "git_branch": branch,
            "git_remote": remote,
            "git_patch": patch,
            "git_submodules": submodules,
        }

    @staticmethod
    def collect_environment_info() -> Dict[str, Any]:
        """Collect environment information."""
        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "command": " ".join(sys.argv) if sys.argv else "<unknown>",
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
        }

    @classmethod
    def create_metadata(
        cls, huldra_obj: Any, directory: Path, ignore_diff: bool = False
    ) -> Dict[str, Any]:
        """Create complete metadata dictionary."""
        metadata = {
            "huldra_python_def": HuldraSerializer.to_python(
                huldra_obj, multiline=False
            ),
            "huldra_obj": HuldraSerializer.to_dict(huldra_obj),
            "huldra_hash": HuldraSerializer.compute_hash(huldra_obj),
            "huldra_path": str(directory.resolve()),
            **cls.collect_git_info(ignore_diff),
            **cls.collect_environment_info(),
        }
        return metadata

    @classmethod
    def write_metadata(cls, metadata: Dict[str, Any], directory: Path) -> None:
        """Write metadata to file."""
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                default=lambda o: o.model_dump()
                if BaseModel is not None and isinstance(o, BaseModel)
                else str(o),
            )
        )

    @classmethod
    def read_metadata(cls, directory: Path) -> Dict[str, Any]:
        """Read metadata from file."""
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        return json.loads(metadata_path.read_text())


# =============================================================================
# Core Huldra Class
# =============================================================================


class Huldra[T](ABC):
    """
    Base class for cached computations with provenance tracking.

    Subclasses must implement:
    - _create(self) -> T
    - _load(self) -> T
    - _slug(self) -> str
    """

    MISSING = MISSING

    # Configuration (can be overridden in subclasses)
    version_controlled: bool = False

    # Maximum time to wait for result (seconds). Default: 10 minutes.
    _max_wait_time_sec: float = 600.0

    def _slug(self: Self) -> str:
        """Return the slug for this Huldra object (must be implemented by decorator)."""
        raise NotImplementedError("Slug not set - use @huldra decorator")

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
        return root / self._slug() / self.hexdigest

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
    def exists_or_create(self, executor: submitit.Executor) -> T | submitit.Job[T]: ...

    @overload
    def exists_or_create(self, executor: None = None) -> T: ...

    def exists_or_create(
        self: Self,
        executor: submitit.Executor | None = None,
    ) -> T | submitit.Job[T]:
        """
        Ensure result exists, computing if necessary.

        Args:
            executor: Optional executor for batch submission (e.g., submitit.Executor)
            wait: Whether to wait for completion (default: True if no executor)
            on_job_id: Callback when job ID is available
            max_requeues: Maximum number of preemption requeues

        Returns:
            Result if wait=True, job handle if wait=False, or None if already exists

        Raises:
            HuldraComputeError: If computation fails with detailed error information
        """
        start_time = time.time()
        directory = self.huldra_dir
        directory.mkdir(parents=True, exist_ok=True)

        # if wait is None:
        #     wait = executor is None

        # if max_requeues is None:
        #     max_requeues = CONFIG.max_requeues

        # Fast path: already successful
        if StateManager.read_state(directory).get("status") == "success":
            try:
                # return self._load() if (executor is None or wait) else None
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
                status, created_here, result = self._run_locally(start_time=start_time)
                if status == "success":
                    if created_here:
                        return cast(T, result)
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
        attempts_left = max_requeues

        while True:
            self._check_timeout(start_time)
            state = StateManager.read_state(directory)
            status = state.get("status")

            # Success - load and return result
            if status == "success":
                try:
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
                self._create()
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
                result = self._create()
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


# =============================================================================
# Decorator
# =============================================================================

T = TypeVar("T", bound=type)


@dataclass_transform(
    field_specifiers=(chz.field,), kw_only_default=True, frozen_default=True
)
def huldra(
    _cls: T | None = None,
    *,
    slug: str | Callable[[Any], str],
    version_controlled: bool = False,
    **huldra_kwargs: Any,
) -> Callable[[T], T] | T:
    """
    Decorator to create a Huldra chz class.

    Args:
        slug: Unique identifier for this Huldra type (string or callable)
        version_controlled: If True, store in data/huldra/git, else data/huldra/data
        **huldra_kwargs: Additional arguments passed to @chz.chz

    Example:
        @huldra(slug="my-computation", version_controlled=True)
        class MyComputation(Huldra[pd.DataFrame]):
            param1: int
            param2: str

            def _create(self) -> pd.DataFrame:
                # Compute result
                df = ...
                df.to_parquet(self.huldra_dir / "result.parquet")
                return df

            def _load(self, self.huldra_dir: Path) -> pd.DataFrame:
                return pd.read_parquet(self.huldra_dir / "result.parquet")
    """

    def decorate(cls: T) -> T:
        # Verify inheritance
        if not issubclass(cls, Huldra):
            raise TypeError(f"{cls.__name__} must inherit from Huldra")

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
            for name, value in cls.__dict__.items():
                if isinstance(value, FieldType) and name not in annotations:
                    annotations[name] = Any

        if annotations:
            type.__setattr__(cls, "__annotations__", annotations)

        chz_kwargs: dict[str, Any] = {}
        for key in ("version", "typecheck"):
            if key in huldra_kwargs:
                chz_kwargs[key] = huldra_kwargs.pop(key)

        if huldra_kwargs:
            extra = ", ".join(sorted(huldra_kwargs))
            raise TypeError(f"Unsupported huldra options: {extra}")

        # Apply chz decorator
        cls = chz.chz(cls, **chz_kwargs)  # ty: ignore[invalid-assignment]

        # Set slug method
        if isinstance(slug, str):
            setattr(cls, "_slug", lambda self, _s=slug: _s)
        else:
            setattr(cls, "_slug", slug)

        # Set version_controlled attribute
        setattr(cls, "version_controlled", version_controlled)

        return cls

    if _cls is None:
        return decorate
    else:
        raise TypeError(
            "@huldra requires explicit slug parameter: "
            "use @huldra(slug='my-slug') or @huldra(slug='my-slug', version_controlled=True)"
        )


# =============================================================================
# Typed Collections (HuldraList)
# =============================================================================

_H = TypeVar("_H", bound=Huldra, covariant=True)


class _HuldraListMeta(type):
    """Metaclass that provides collection methods for HuldraList subclasses."""

    def _entries(cls: "type[HuldraList[_H]]") -> List[_H]:
        """Collect all Huldra instances from class attributes."""
        items: List[_H] = []
        seen: set[str] = set()

        def maybe_add(obj: Any) -> None:
            if not isinstance(obj, Huldra):
                raise TypeError(f"{obj!r} is not a Huldra instance")

            digest = obj.hexdigest
            if digest not in seen:
                seen.add(digest)
                items.append(cast(_H, obj))

        for name, value in cls.__dict__.items():
            if name.startswith("_") or callable(value):
                continue

            if isinstance(value, dict):
                for v in value.values():
                    maybe_add(v)
            elif isinstance(value, list):
                for v in value:
                    maybe_add(v)
            else:
                maybe_add(value)

        return items

    def __iter__(cls: "type[HuldraList[_H]]") -> Iterator[_H]:
        """Iterate over all Huldra instances."""
        return iter(cls._entries())

    def all(cls: "type[HuldraList[_H]]") -> List[_H]:
        """Get all Huldra instances as a list."""
        return cls._entries()

    def items_iter(
        cls: "type[HuldraList[_H]]",
    ) -> Generator[tuple[str, _H], None, None]:
        """Iterate over (name, instance) pairs."""
        for name, value in cls.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            if not isinstance(value, dict):
                yield name, cast(_H, value)

    def items(cls: "type[HuldraList[_H]]") -> List[tuple[str, _H]]:
        """Get all (name, instance) pairs as a list."""
        return list(cls.items_iter())

    @overload
    def by_name(
        cls: "type[HuldraList[_H]]", name: str, *, strict: Literal[True] = True
    ) -> _H: ...

    @overload
    def by_name(
        cls: "type[HuldraList[_H]]", name: str, *, strict: Literal[False]
    ) -> Optional[_H]: ...

    def by_name(cls: "type[HuldraList[_H]]", name: str, *, strict: bool = True):
        """Get Huldra instance by name."""
        attr = cls.__dict__.get(name)
        if attr and not callable(attr) and not name.startswith("_"):
            return cast(_H, attr)

        # Check nested dicts
        for value in cls.__dict__.values():
            if isinstance(value, dict) and name in value:
                return cast(_H, value[name])

        if strict:
            raise KeyError(f"{cls.__name__} has no entry named '{name}'")
        return None


class HuldraList(Generic[_H], metaclass=_HuldraListMeta):
    """
    Base class for typed Huldra collections.

    Example:
        @huldra(slug="my-computation")
        class MyComputation(Huldra[str]):
            value: int

            def _create(self) -> str:
                result = f"Result: {self.value}"
                (self.huldra_dir / "result.txt").write_text(result)
                return result

            def _load(self) -> str:
                return (self.huldra_dir / "result.txt").read_text()

        class MyExperiments(HuldraList[MyComputation]):
            exp1 = MyComputation(value=1)
            exp2 = MyComputation(value=2)
            exp3 = MyComputation(value=3)

        # Use the collection
        for exp in MyExperiments:
            result = exp.exists_or_create()
            print(result)
    """

    pass


# =============================================================================
# Utility Functions
# =============================================================================


def get_huldra_root(version_controlled: bool = False) -> Path:
    """
    Get the root directory for Huldra storage.

    Args:
        version_controlled: If True, return git-tracked directory

    Returns:
        Path to storage root
    """
    return HULDRA_CONFIG.get_root(version_controlled)


def set_huldra_root(path: Path) -> None:
    """
    Override the base Huldra root directory.

    Args:
        path: New base root directory
    """
    HULDRA_CONFIG.base_root = path.expanduser().resolve()
