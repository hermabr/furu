import contextlib
import contextvars
import datetime
import logging
import threading
from pathlib import Path
from typing import Any, Generator

from ..config import HULDRA_CONFIG

_HULDRA_HOLDER_STACK: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    "huldra_holder_stack", default=()
)
_HULDRA_LOG_LOCK = threading.Lock()


def _holder_to_log_dir(holder: Any) -> Path:
    if isinstance(holder, Path):
        return holder
    directory = getattr(holder, "huldra_dir", None)
    if isinstance(directory, Path):
        return directory
    raise TypeError(
        "holder must be a pathlib.Path or have a .huldra_dir: pathlib.Path attribute"
    )


@contextlib.contextmanager
def enter_holder(holder: Any) -> Generator[None, None, None]:
    """
    Push a holder object onto the logging stack for this context.

    Huldra calls this automatically during `load_or_create()`, so nested
    dependencies will log to the active dependency's folder and then revert.
    """
    configure_logging()
    stack = _HULDRA_HOLDER_STACK.get()
    token = _HULDRA_HOLDER_STACK.set((*stack, holder))
    try:
        yield
    finally:
        _HULDRA_HOLDER_STACK.reset(token)


def current_holder() -> Any | None:
    """Return the current holder object for logging, if any."""
    stack = _HULDRA_HOLDER_STACK.get()
    return stack[-1] if stack else None


def current_log_dir() -> Path:
    """Return the directory logs should be written to for this context."""
    holder = current_holder()
    if holder is None:
        return HULDRA_CONFIG.base_root
    return _holder_to_log_dir(holder)


class _HuldraLogFormatter(logging.Formatter):
    def formatTime(  # noqa: N802 - keep logging.Formatter API
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        return dt.isoformat(timespec="seconds")


class _HuldraContextFileHandler(logging.Handler):
    """
    A logging handler that writes to `current_log_dir() / "huldra.log"` at emit-time.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        directory = current_log_dir()
        with contextlib.suppress(Exception):
            directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / "huldra.log"
        with _HULDRA_LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"{message}\n")


class _HuldraScopeFilter(logging.Filter):
    """
    Capture all logs while inside a holder context.

    Outside a holder context, only capture logs from the `huldra` logger namespace.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if current_holder() is not None:
            return True
        return record.name == "huldra" or record.name.startswith("huldra.")


def configure_logging() -> None:
    """
    Install a context-aware file handler on the root logger (idempotent).

    With this installed, any stdlib logger (e.g. `logging.getLogger(__name__)`)
    that propagates to the root logger will be written to the current holder's
    `huldra.log` while a holder is active.
    """
    root = logging.getLogger()
    if any(isinstance(h, _HuldraContextFileHandler) for h in root.handlers):
        return

    handler = _HuldraContextFileHandler(level=logging.DEBUG)
    handler.addFilter(_HuldraScopeFilter())
    handler.setFormatter(_HuldraLogFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(handler)


def get_logger() -> logging.Logger:
    """
    Return the default huldra logger.

    It is configured with a context-aware file handler that routes log records to
    the current holder's directory (see `enter_holder()`).
    """
    configure_logging()
    logger = logging.getLogger("huldra")
    logger.setLevel(logging.DEBUG)
    return logger


def log(message: str, *, level: str = "INFO") -> Path:
    """
    Log a message to the current holder's `huldra.log` via stdlib `logging`.

    If no holder is active, logs to `HULDRA_CONFIG.base_root / "huldra.log"`.
    Returns the path written to.
    """
    directory = current_log_dir()
    log_path = directory / "huldra.log"

    level_no = logging.getLevelNamesMapping().get(level.upper())
    if level_no is None:
        raise ValueError(f"Unknown log level: {level!r}")

    configure_logging()
    get_logger().log(level_no, message)
    return log_path

