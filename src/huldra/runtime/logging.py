import contextlib
import contextvars
import datetime
import logging
import os
import threading
from pathlib import Path
from typing import Any, Generator

from ..config import HULDRA_CONFIG

_HULDRA_HOLDER_STACK: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    "huldra_holder_stack", default=()
)
_HULDRA_LOG_LOCK = threading.Lock()
_HULDRA_CONSOLE_LOCK = threading.Lock()

_LOAD_OR_CREATE_PREFIX = "load_or_create"


def _strip_load_or_create_decision_suffix(message: str) -> str:
    """
    Strip a trailing `(<decision>)` suffix from `load_or_create ...` console lines.

    This keeps detailed decision info in file logs, but makes console output cleaner.
    """
    if not message.startswith(_LOAD_OR_CREATE_PREFIX):
        return message
    if not message.endswith(")"):
        return message
    idx = message.rfind(" (")
    if idx == -1:
        return message

    decision = message[idx + 2 : -1]
    if decision == "create" or "->" in decision:
        return message[:idx]
    return message


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


class _HuldraFileFilter(logging.Filter):
    """Filter out records intended for console only."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(record, "huldra_console_only", False))


class _HuldraConsoleFilter(logging.Filter):
    """Only show huldra namespace logs on console."""

    def filter(self, record: logging.LogRecord) -> bool:
        if bool(getattr(record, "huldra_file_only", False)):
            return False
        return record.name == "huldra" or record.name.startswith("huldra.")


def _console_level() -> int:
    level = os.getenv("HULDRA_LOG_LEVEL", "INFO").upper()
    return logging.getLevelNamesMapping().get(level, logging.INFO)


class _HuldraRichConsoleHandler(logging.Handler):
    def __init__(self, *, level: int) -> None:
        super().__init__(level=level)
        from rich.console import Console  # type: ignore

        self._console = Console(stderr=True)

    @staticmethod
    def _format_location(record: logging.LogRecord) -> str:
        filename = Path(record.pathname).name if record.pathname else "<unknown>"
        return f"[{filename}:{record.lineno}]"

    @staticmethod
    def _format_message_text(record: logging.LogRecord) -> Any:
        from rich.text import Text  # type: ignore

        message = _strip_load_or_create_decision_suffix(record.getMessage())
        action_color = getattr(record, "huldra_action_color", None)
        if isinstance(action_color, str) and message.startswith(_LOAD_OR_CREATE_PREFIX):
            prefix = _LOAD_OR_CREATE_PREFIX
            rest = message[len(prefix) :]
            text = Text()
            text.append(prefix, style=action_color)
            text.append(rest)
            return text
        return Text(message)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from rich.text import Text  # type: ignore
        except Exception:  # pragma: no cover
            return

        level_style = self._level_style(record.levelno)
        timestamp = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).strftime("%H:%M:%S")

        location = self._format_location(record)

        line = Text()
        line.append(timestamp, style="dim")
        line.append(" ")
        line.append(location, style=level_style)
        line.append(" ")
        line.append_text(self._format_message_text(record))

        with _HULDRA_CONSOLE_LOCK:
            self._console.print(line)

        if record.exc_info:
            try:
                from rich.traceback import Traceback  # type: ignore

                exc_type, exc_value, tb = record.exc_info
                if exc_type is not None and exc_value is not None and tb is not None:
                    with _HULDRA_CONSOLE_LOCK:
                        self._console.print(
                            Traceback.from_exception(
                                exc_type, exc_value, tb, show_locals=False
                            )
                        )
            except Exception:
                pass

    @staticmethod
    def _level_style(levelno: int) -> str:
        if levelno >= logging.ERROR:
            return "red"
        if levelno >= logging.WARNING:
            return "yellow"
        if levelno >= logging.INFO:
            return "blue"
        return "magenta"


def configure_logging() -> None:
    """
    Install context-aware file logging + rich console logging (idempotent).

    With this installed, any stdlib logger (e.g. `logging.getLogger(__name__)`)
    that propagates to the root logger will be written to the current holder's
    `huldra.log` while a holder is active.
    """
    root = logging.getLogger()
    if not any(isinstance(h, _HuldraContextFileHandler) for h in root.handlers):
        handler = _HuldraContextFileHandler(level=logging.DEBUG)
        handler.addFilter(_HuldraScopeFilter())
        handler.addFilter(_HuldraFileFilter())
        handler.setFormatter(
            _HuldraLogFormatter(
                "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d %(message)s"
            )
        )
        root.addHandler(handler)

    try:
        import rich  # type: ignore
    except Exception:  # pragma: no cover
        rich = None  # type: ignore

    if rich is not None and not any(
        isinstance(h, _HuldraRichConsoleHandler) for h in root.handlers
    ):
        console = _HuldraRichConsoleHandler(level=_console_level())
        console.addFilter(_HuldraConsoleFilter())
        root.addHandler(console)


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


def write_separator(line: str = "------------------") -> Path:
    """
    Write a raw separator line to the current holder's `huldra.log`.

    This bypasses standard formatting so repeated `load_or_create()` calls are easy to spot.
    """
    directory = current_log_dir()
    log_path = directory / "huldra.log"

    with contextlib.suppress(Exception):
        directory.mkdir(parents=True, exist_ok=True)

    with _HULDRA_LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"{line}\n")
    return log_path
