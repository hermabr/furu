from __future__ import annotations

import logging
import os
import shutil
import time as _time
import traceback as _traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, Iterator, Mapping
import sys

from furu.config import get_config

_BASE_LOGGER_NAME = "furu"
_FURU_DIR = Path(__file__).resolve().parent

# TODO: ContextVar state does not propagate to new threads. Logs emitted from
# worker threads inside create() or create_batched() will use fallback.log
# unless the current log path context is propagated explicitly.
_CURRENT_LOG_PATHS: ContextVar[tuple[Path, ...]] = ContextVar(
    "furu_current_log_paths", default=()
)
# The component column (e.g. "coord", "wkr·1", "slurm"). None means a local run
# with no executor, in which case the column is omitted entirely.
_LOG_COMPONENT: ContextVar[str | None] = ContextVar("furu_log_component", default=None)


# ---------------------------------------------------------------------------
# Structured event metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FuruMeta:
    """Structured fields attached to a record via ``log_event``.

    The colored console renderer lays these out as columns; the logfmt renderer
    emits them as ``key=value`` pairs. ``fields`` are file-only correlation data
    (lease ids, counts) that are dropped from the console to cut noise.
    """

    event: str | None = None
    label: str | None = None
    detail: str | None = None
    status: str | None = None  # "ok" | "error"
    duration: float | None = None
    fields: Mapping[str, Any] | None = None


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
    event: str | None = None,
    label: str | None = None,
    detail: str | None = None,
    status: str | None = None,
    duration: float | None = None,
    exc_info: Any = None,
    **fields: Any,
) -> None:
    """Emit a structured log record.

    ``message`` is the human-readable fallback (what ``caplog`` and any
    non-furu handler observe). ``event``/``label``/``detail``/``status``/
    ``duration`` drive the redesigned console layout, and ``fields`` are extra
    correlation values kept only in the file/logfmt output.
    """

    meta = _FuruMeta(
        event=event,
        label=label,
        detail=detail,
        status=status,
        duration=duration,
        fields=fields or None,
    )
    logger.log(
        level,
        message,
        *args,
        exc_info=exc_info,
        extra={"_furu_meta": meta},
        stacklevel=2,
    )


@contextmanager
def log_component(component: str | None) -> Iterator[None]:
    """Set the component column for logs emitted in this context."""

    token = _LOG_COMPONENT.set(component)
    try:
        yield
    finally:
        _LOG_COMPONENT.reset(token)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_RESET = "\x1b[0m"


def _fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


# Gruvbox dark (medium) palette — mirrors the design preview.
_C_TIME = _fg(0xA8, 0x99, 0x84)
_C_INFO = _fg(0x83, 0xA5, 0x98)
_C_DEBUG = _fg(0x92, 0x83, 0x74)
_C_WARN = _fg(0xFA, 0xBD, 0x2F)
_C_ERROR = _fg(0xFB, 0x49, 0x34)
_C_OK = _fg(0xB8, 0xBB, 0x26)
_C_COMP = _fg(0xD3, 0x86, 0x9B)
_C_ID = _fg(0x8E, 0xC0, 0x7C)
_C_MSG = _fg(0xFB, 0xF1, 0xC7)
_C_USER = _fg(0xFE, 0x80, 0x19)
_C_DIM = _fg(0x92, 0x83, 0x74)


def _level_letter_color(levelno: int) -> tuple[str, str]:
    if levelno >= logging.ERROR:
        return "E", _C_ERROR
    if levelno >= logging.WARNING:
        return "W", _C_WARN
    if levelno >= logging.INFO:
        return "I", _C_INFO
    return "D", _C_DEBUG


def _hms(created: float) -> str:
    return _time.strftime("%H:%M:%S", _time.localtime(created))


def _iso_ms(created: float) -> str:
    dt = datetime.fromtimestamp(created, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _elide_filename(name: str, max_len: int = 18) -> str:
    if len(name) <= max_len:
        return name
    keep = max_len - 1  # room for the ellipsis
    head = keep // 2
    tail = keep - head
    return name[:head] + "…" + name[-tail:]


def _user_caller(record: logging.LogRecord) -> str | None:
    """Return ``file:line`` for user code, or None for furu-internal call sites."""

    pathname = getattr(record, "pathname", None)
    if not pathname:
        return None
    try:
        Path(pathname).resolve().relative_to(_FURU_DIR)
        return None  # inside the furu package → suppressed
    except (ValueError, OSError):
        pass
    return f"{_elide_filename(Path(pathname).name)}:{record.lineno}"


def _term_width() -> int:
    try:
        return shutil.get_terminal_size((100, 24)).columns
    except Exception:
        return 100


def _wrap(text: str, first_width: int, rest_width: int) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    width = first_width
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
            width = rest_width
    lines.append(current)
    return lines or [text]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _ConsoleFormatter(logging.Formatter):
    """Colored single-screen layout: ``HH:MM:SS L [comp] body  caller``."""

    def format(self, record: logging.LogRecord) -> str:
        meta = getattr(record, "_furu_meta", None)
        component = _LOG_COMPONENT.get()
        caller = _user_caller(record)
        letter, level_color = _level_letter_color(record.levelno)
        timestamp = _hms(record.created)

        prefix_plain = f"{timestamp} {letter}"
        prefix = f"{_C_TIME}{timestamp}{_RESET} {level_color}{letter}{_RESET}"
        if component:
            prefix_plain += f" {component}"
            prefix += f" {_C_COMP}{component}{_RESET}"

        if isinstance(meta, _FuruMeta) and meta.event is not None:
            body_plain, body = _render_event_body(meta)
            wrappable = False
        else:
            text = record.getMessage()
            body_plain = text
            if caller:
                body = f"{_C_USER}{text}{_RESET}"
            else:
                body = text
            wrappable = True

        line = _compose_line(
            prefix=prefix,
            prefix_plain=prefix_plain,
            body=body,
            body_plain=body_plain,
            caller=caller,
            wrappable=wrappable,
        )

        if record.exc_info:
            line += "\n" + _format_traceback_console(record.exc_info)
        return line


def _render_event_body(meta: _FuruMeta) -> tuple[str, str]:
    plain = meta.event or ""
    colored = meta.event or ""
    if meta.label:
        plain += f" {meta.label}"
        colored += f" {_C_ID}{meta.label}{_RESET}"
    if meta.detail:
        plain += f" {meta.detail}"
        detail_color = _C_ERROR if meta.status == "error" else _C_MSG
        colored += f" {detail_color}{meta.detail}{_RESET}"
    if meta.status == "ok":
        plain += "  ok"
        colored += f"  {_C_OK}ok{_RESET}"
        if meta.duration is not None:
            suffix = f" · {_fmt_duration(meta.duration)}"
            plain += suffix
            colored += suffix
    return plain, colored


def _compose_line(
    *,
    prefix: str,
    prefix_plain: str,
    body: str,
    body_plain: str,
    caller: str | None,
    wrappable: bool,
) -> str:
    width = _term_width()
    indent = len(prefix_plain) + 1
    first_plain_len = indent + len(body_plain)
    caller_cost = (len(caller) + 1) if caller else 0

    # Short enough to fit on one row (with room for a right-aligned caller).
    if not wrappable or first_plain_len + caller_cost <= width:
        line = f"{prefix} {body}"
        if caller:
            pad = max(1, width - first_plain_len - len(caller))
            line += " " * pad + f"{_C_DIM}{caller}{_RESET}"
        return line

    # Long free-form message: wrap with a hanging indent under the message
    # column. Time + caller stay on the first row; the body overflows below.
    first_width = max(8, width - indent - caller_cost)
    rest_width = max(8, width - indent)
    chunks = _wrap(body_plain, first_width, rest_width)
    rendered: list[str] = []
    for i, chunk in enumerate(chunks):
        colored_chunk = f"{_C_USER}{chunk}{_RESET}"
        if i == 0:
            row = f"{prefix} {colored_chunk}"
            if caller:
                pad = max(1, width - (indent + len(chunk)) - len(caller))
                row += " " * pad + f"{_C_DIM}{caller}{_RESET}"
            rendered.append(row)
        else:
            rendered.append(" " * indent + colored_chunk)
    return "\n".join(rendered)


def _format_traceback_console(exc_info: Any) -> str:
    text = "".join(_traceback.format_exception(*exc_info)).rstrip("\n")
    lines = text.split("\n")
    rendered = []
    for i, line in enumerate(lines):
        color = _C_ERROR if i == len(lines) - 1 else _C_DIM
        rendered.append(f"  {color}{line}{_RESET}")
    return "\n".join(rendered)


def _logfmt_value(value: str) -> str:
    if value == "" or any(ch in value for ch in ' "='):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


class _LogfmtFormatter(logging.Formatter):
    """Machine-readable ``key=value`` layout for files and piped output."""

    def format(self, record: logging.LogRecord) -> str:
        meta = getattr(record, "_furu_meta", None)
        component = _LOG_COMPONENT.get()
        caller = _user_caller(record)

        parts = [_iso_ms(record.created), f"level={record.levelname.lower()}"]
        if component:
            parts.append(f"comp={_logfmt_value(component)}")

        if isinstance(meta, _FuruMeta) and meta.event is not None:
            parts.append(f"msg={_logfmt_value(meta.event)}")
            if meta.label:
                parts.append(f"id={_logfmt_value(meta.label)}")
            if meta.detail:
                parts.append(f"detail={_logfmt_value(meta.detail)}")
            if meta.status:
                parts.append(f"status={meta.status}")
            if meta.duration is not None:
                parts.append(f"dur={meta.duration:.3f}")
            if meta.fields:
                for key, value in meta.fields.items():
                    parts.append(f"{key}={_logfmt_value(str(value))}")
        else:
            parts.append(f"msg={_logfmt_value(record.getMessage())}")

        if caller:
            parts.append(f"caller={_logfmt_value(caller)}")

        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _console_color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Handlers and logger setup
# ---------------------------------------------------------------------------


class _ScopedFileHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        log_paths = _CURRENT_LOG_PATHS.get()
        if not log_paths:
            log_paths = (get_config().run_directories.objects / "fallback.log",)

        try:
            rendered = self.format(record)
            for log_path in log_paths:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(rendered)
                    f.write("\n")
        except Exception:
            self.handleError(record)


@cache
def _base_logger() -> logging.Logger:
    logger = logging.getLogger(_BASE_LOGGER_NAME)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if get_config().debug_mode else logging.INFO)
    stdout_handler.setFormatter(
        _ConsoleFormatter() if _console_color_enabled() else _LogfmtFormatter()
    )
    logger.addHandler(stdout_handler)

    file_handler = _ScopedFileHandler()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_LogfmtFormatter())
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    _base_logger()
    if name is None or name == _BASE_LOGGER_NAME:
        return logging.getLogger(_BASE_LOGGER_NAME)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


@contextmanager
def _scoped_log_files(log_paths: tuple[Path, ...]) -> Iterator[None]:
    token = _CURRENT_LOG_PATHS.set(tuple(dict.fromkeys(log_paths)))
    try:
        yield
    finally:
        _CURRENT_LOG_PATHS.reset(token)
