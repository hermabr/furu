from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import sys
import textwrap
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from pathlib import Path
from typing import Any, Iterator

from furu.config import get_config

_BASE_LOGGER_NAME = "furu"

# Absolute path to the furu package. A log record whose call site lives inside
# this directory is furu-internal (no file:line shown); anything else is the
# user's own code (file:line shown).
_FURU_ROOT = str(Path(__file__).resolve().parent)

# TODO: ContextVar state does not propagate to new threads. Logs emitted from
# worker threads inside create() or create_batched() will use fallback.log
# unless the current log path context is propagated explicitly.
_CURRENT_LOG_PATHS: ContextVar[tuple[Path, ...]] = ContextVar(
    "furu_current_log_paths", default=()
)
# The component label shown on the console (coord / wkr·1 / slurm). None means a
# local run with a single place things happen, so no component column is drawn.
_COMPONENT: ContextVar[str | None] = ContextVar("furu_component", default=None)


# --- console layout ---------------------------------------------------------

_COMPONENT_WIDTH = 5
_MIN_MSG_WIDTH = 24
_CALLER_GAP = 2
_FALLBACK_COLUMNS = 100

# Filenames longer than this are middle-elided to head + … + tail so the
# file:line tag stays narrow while both ends remain recognizable.
_MAX_FILENAME = 20
_FILENAME_HEAD = 9
_FILENAME_TAIL = 9

_RESET = "\x1b[0m"
_ANSI = {
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "gray": "\x1b[90m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "orange": "\x1b[38;5;208m",
}

_LEVEL_LETTER = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}
_LEVEL_COLOR = {
    logging.DEBUG: "gray",
    logging.INFO: "blue",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}

# Short artifact id, e.g. RawData:9f2a1:8c3d2 (also matches the full module-
# qualified, full-hash form), highlighted with its own color on the console.
_ARTIFACT_ID_RE = re.compile(r"[A-Za-z_][\w.]*:[0-9a-f]{5,}:[0-9a-f]{5,}")
# Standalone "ok" status word, coloured green on a finish line.
_OK_RE = re.compile(r"\bok\b")


def _elide_filename(name: str) -> str:
    if len(name) <= _MAX_FILENAME:
        return name
    return f"{name[:_FILENAME_HEAD]}…{name[-_FILENAME_TAIL:]}"


def _user_caller(pathname: str, lineno: int) -> str | None:
    """``filename:line`` for user code, or ``None`` for furu-internal code."""
    try:
        resolved = str(Path(pathname).resolve())
    except (OSError, ValueError, RuntimeError):
        resolved = pathname
    if resolved == _FURU_ROOT or resolved.startswith(_FURU_ROOT + os.sep):
        return None
    return f"{_elide_filename(Path(pathname).name)}:{lineno}"


def _use_color(stream: object) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _console_mode(stream: object) -> bool:
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


class _ContextFilter(logging.Filter):
    """Stamp every record with the current component and its user call site."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.furu_component = _COMPONENT.get()
        record.furu_user_caller = _user_caller(record.pathname, record.lineno)
        return True


class _ConsoleFormatter(logging.Formatter):
    """Minimal coloured console format. Color carries information, so text can
    shrink: HH:MM:SS, a one-letter level, an optional component, a free-form
    message, and a right-aligned file:line for the user's own code."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def _c(self, text: str, color: str | None, *, bold: bool = False) -> str:
        if not self._color or not color or not text:
            return text
        code = _ANSI.get(color, "")
        if bold:
            code = _ANSI["bold"] + code
        return f"{code}{text}{_RESET}"

    def _style_message(self, text: str, levelno: int, *, is_user: bool) -> str:
        if not self._color or not text:
            return text
        if is_user:
            return self._c(text, "orange")
        if levelno >= logging.ERROR:
            return self._c(text, "red")
        text = _ARTIFACT_ID_RE.sub(lambda m: self._c(m.group(0), "cyan"), text)
        text = _OK_RE.sub(lambda m: self._c(m.group(0), "green"), text)
        return text

    def _format_traceback(self, exc_info: Any, indent: str) -> str:
        rendered = "".join(traceback.format_exception(*exc_info))
        rule = self._c("│", "red")
        out: list[str] = []
        for line in rendered.rstrip("\n").split("\n"):
            is_exception = (
                bool(line)
                and not line[0].isspace()
                and not (line.startswith("Traceback"))
            )
            content = self._c(line, "red" if is_exception else "dim")
            out.append(f"{indent}{rule} {content}")
        return "\n".join(out)

    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%H:%M:%S")
        letter = _LEVEL_LETTER.get(record.levelno, record.levelname[:1])
        component = getattr(record, "furu_component", None)
        caller = getattr(record, "furu_user_caller", None)
        message = record.getMessage()

        plain_prefix = f"{time_str} {letter} "
        if component:
            plain_prefix += f"{component.ljust(_COMPONENT_WIDTH)} "

        columns = shutil.get_terminal_size((_FALLBACK_COLUMNS, 24)).columns
        avail = max(_MIN_MSG_WIDTH, columns - len(plain_prefix))

        msg_lines: list[str] = []
        for segment in message.split("\n"):
            msg_lines.extend(
                textwrap.wrap(
                    segment,
                    width=avail,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [""]
            )

        is_user = caller is not None
        styled_prefix = self._c(time_str, "dim") + " "
        styled_prefix += self._c(letter, _LEVEL_COLOR.get(record.levelno), bold=True)
        styled_prefix += " "
        if component:
            styled_prefix += self._c(component.ljust(_COMPONENT_WIDTH), "magenta") + " "

        out_lines = [
            styled_prefix
            + self._style_message(msg_lines[0], record.levelno, is_user=is_user)
        ]
        if caller:
            pad = columns - len(plain_prefix) - len(msg_lines[0]) - len(caller)
            out_lines[0] += " " * max(_CALLER_GAP, pad) + self._c(caller, "dim")

        indent = " " * len(plain_prefix)
        for line in msg_lines[1:]:
            out_lines.append(
                indent + self._style_message(line, record.levelno, is_user=is_user)
            )

        result = "\n".join(out_lines)
        if record.exc_info:
            result += "\n" + self._format_traceback(record.exc_info, indent)
        return result


def _iso_timestamp(record: logging.LogRecord) -> str:
    moment = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z"


def _logfmt_value(value: str) -> str:
    if value == "":
        return '""'
    if any(char in value for char in ' ="\n\t'):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    return value


class _LogfmtFormatter(logging.Formatter):
    """logfmt for files and non-TTY streams: full UTC date + ms, the level, the
    component, the message, the user call site, and any structured fields the
    call site attached via ``extra={"furu_fields": {...}}`` (correlation ids,
    full hashes, raw counts) that the console intentionally drops."""

    def format(self, record: logging.LogRecord) -> str:
        parts = [_iso_timestamp(record), f"level={record.levelname.lower()}"]
        component = getattr(record, "furu_component", None)
        if component:
            parts.append(f"comp={_logfmt_value(component.replace('·', '.'))}")
        parts.append(f"msg={_logfmt_value(record.getMessage())}")
        caller = getattr(record, "furu_user_caller", None)
        if caller:
            parts.append(f"caller={_logfmt_value(caller)}")
        fields = getattr(record, "furu_fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                parts.append(f"{key}={_logfmt_value(str(value))}")
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip(
                "\n"
            )
        return line


def _stdout_formatter() -> logging.Formatter:
    if _console_mode(sys.stdout):
        return _ConsoleFormatter(color=_use_color(sys.stdout))
    return _LogfmtFormatter()


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
    context_filter = _ContextFilter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if get_config().debug_mode else logging.INFO)
    stdout_handler.setFormatter(_stdout_formatter())
    stdout_handler.addFilter(context_filter)
    logger.addHandler(stdout_handler)

    file_handler = _ScopedFileHandler()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_LogfmtFormatter())
    file_handler.addFilter(context_filter)
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


@contextmanager
def _log_component(component: str | None) -> Iterator[None]:
    token = _COMPONENT.set(component)
    try:
        yield
    finally:
        _COMPONENT.reset(token)
