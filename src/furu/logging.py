from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import textwrap
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Iterator, Mapping

from furu.config import get_config

_BASE_LOGGER_NAME = "furu"

# The directory that holds furu's own source. Log records whose call site lives
# here are framework-internal and render without a file:line tag; records from
# anywhere else are "your code" and get one.
_FURU_PACKAGE_DIR = Path(__file__).resolve().parent

# Records may carry an extra `_furu_detail` mapping (see `log_detail`). The
# console renderer drops it (correlation ids are noise on screen); the logfmt
# renderer appends each entry as a `key=value` field so the file keeps the full,
# machine-readable record.
_DETAIL_ATTR = "_furu_detail"

# TODO: ContextVar state does not propagate to new threads. Logs emitted from
# worker threads inside create() or create_batched() will use fallback.log
# unless the current log path context is propagated explicitly.
_CURRENT_LOG_PATHS: ContextVar[tuple[Path, ...]] = ContextVar(
    "furu_current_log_paths", default=()
)
# The component label for the current execution context, or None for a plain
# local run: "coord" for the coordinator, "l<n>" for a local worker, "s<n>" for
# a slurm worker, and "slurm" for the slurm pool thread. Set near each
# process/thread entry point via `_scoped_component`; read by the renderers.
# Like the log-path context above, it does not cross into threads spawned later,
# so each pool / worker thread sets its own.
_CURRENT_COMPONENT: ContextVar[str | None] = ContextVar(
    "furu_current_component", default=None
)


# --- ANSI styling --------------------------------------------------------------

_RESET = "\x1b[0m"
_DIM = "2"
_RED = "31"
_GREEN = "32"
_YELLOW = "33"
_CYAN = "36"
_MAGENTA = "35"
# 256-colour orange, reserved for your own (non-furu) log message bodies so they
# read as "your code" at a glance; the level letter still carries severity.
_ORANGE = "38;5;208"

_LEVEL_LETTER = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}

_LEVEL_LETTER_STYLE = {
    logging.DEBUG: _DIM,
    logging.INFO: f"{_CYAN};1",
    logging.WARNING: f"{_YELLOW};1",
    logging.ERROR: f"{_RED};1",
    logging.CRITICAL: f"{_RED};1",
}

# An artifact id rendered by Furu._log_label: "<ClassName>:<5 chars>:<5 chars>",
# where each segment is the first 5 chars of a hash. Matching the exact widths
# keeps unrelated "name:host:port"-style tokens from being highlighted.
_ARTIFACT_ID_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*:[0-9A-Za-z]{5}:[0-9A-Za-z]{5}\b"
)

# Filename length past which the middle is elided, keeping both ends recognizable.
_MAX_CALLER_NAME = 21


class _Palette:
    """Wraps text in ANSI codes when colour is enabled, otherwise passes through."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, text: str, style: str) -> str:
        if not self.enabled or not text:
            return text
        return f"\x1b[{style}m{text}{_RESET}"


def _console_mode(stream: object) -> tuple[bool, bool]:
    """Return (use_console_layout, use_colour) for the given output stream.

    Honours the de-facto NO_COLOR / FORCE_COLOR conventions: a TTY (or
    FORCE_COLOR) gets the compact console layout; everything else gets logfmt.
    NO_COLOR strips ANSI but keeps the console layout when attached to a TTY.
    """
    force = bool(os.environ.get("FORCE_COLOR"))
    no_color = os.environ.get("NO_COLOR") is not None
    isatty = getattr(stream, "isatty", None)
    is_tty = callable(isatty) and bool(isatty())
    layout = force or is_tty
    return layout, layout and not no_color


# --- shared record helpers -----------------------------------------------------


def _caller_tag(record: logging.LogRecord) -> str | None:
    """`file:line` for user code, or None for furu-internal call sites."""
    pathname = record.pathname
    # `<stdin>`, `<string>`, `<frozen ...>` and similar synthetic paths are not
    # real files; tagging them as "your code" would print a misleading file:line.
    if pathname.startswith("<"):
        return None
    try:
        resolved = str(Path(pathname).resolve())
    except OSError:
        resolved = os.path.abspath(pathname)
    if resolved == str(_FURU_PACKAGE_DIR) or resolved.startswith(
        str(_FURU_PACKAGE_DIR) + os.sep
    ):
        return None
    return f"{_elide(Path(pathname).name, _MAX_CALLER_NAME)}:{record.lineno}"


def _elide(name: str, max_len: int) -> str:
    if len(name) <= max_len:
        return name
    if max_len <= 1:
        return name[:max_len]
    keep = max_len - 1  # room for the ellipsis
    head = (keep + 1) // 2
    tail = keep - head
    return f"{name[:head]}…{name[-tail:]}" if tail else f"{name[:head]}…"


def _record_detail(record: logging.LogRecord) -> Mapping[str, object] | None:
    detail = getattr(record, _DETAIL_ATTR, None)
    if isinstance(detail, Mapping) and detail:
        return detail
    return None


def _exception_text(record: logging.LogRecord) -> str:
    if record.exc_info:
        return "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
    if record.exc_text:
        return record.exc_text.rstrip("\n")
    return ""


# --- console renderer ----------------------------------------------------------


def _decorate_message(
    message: str, levelno: int, palette: _Palette, *, is_user: bool
) -> str:
    if not palette.enabled:
        return message
    if is_user and levelno < logging.ERROR:
        # Your own log message: render the body in the user colour and let the
        # level letter carry severity. Errors still go red below so a failure
        # never hides in orange.
        return palette.paint(message, _ORANGE)
    if levelno >= logging.ERROR:
        return palette.paint(message, _RED)
    if levelno >= logging.WARNING:
        return palette.paint(message, _YELLOW)
    if levelno <= logging.DEBUG:
        return palette.paint(message, _DIM)
    # INFO: keep default colour but make artifact ids and the "ok" status pop.
    message = _ARTIFACT_ID_RE.sub(lambda m: palette.paint(m.group(0), _GREEN), message)
    message = re.sub(r"\bok\b", palette.paint("ok", f"{_GREEN};1"), message)
    return message


def _render_console(record: logging.LogRecord, *, color: bool) -> str:
    palette = _Palette(color)
    timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
    letter = _LEVEL_LETTER.get(record.levelno, "?")
    component = _CURRENT_COMPONENT.get()
    caller = _caller_tag(record)

    plain_prefix = f"{timestamp} {letter} " + (f"{component} " if component else "")
    colored_prefix = (
        palette.paint(timestamp, _DIM)
        + " "
        + palette.paint(letter, _LEVEL_LETTER_STYLE.get(record.levelno, ""))
        + " "
        + (palette.paint(component, _MAGENTA) + " " if component else "")
    )

    width = shutil.get_terminal_size((100, 24)).columns
    body_width = max(20, width - len(plain_prefix))
    indent = " " * len(plain_prefix)

    # Reserve room on the first visual row for the right-aligned caller tag so it
    # never spills past the edge — but only when that still leaves a usable body.
    caller_reserve = len(caller) + 2 if caller else 0
    if caller_reserve and body_width - caller_reserve < 10:
        caller_reserve = 0

    raw_message = record.getMessage()
    visual_lines: list[str] = []
    for paragraph_index, paragraph in enumerate(raw_message.split("\n")):
        if paragraph_index == 0 and caller_reserve:
            wrapped = textwrap.wrap(
                paragraph,
                width=body_width,
                break_long_words=False,
                initial_indent=" " * caller_reserve,
            ) or [" " * caller_reserve]
            wrapped[0] = wrapped[0][caller_reserve:]
        else:
            wrapped = textwrap.wrap(
                paragraph, width=body_width, break_long_words=False
            ) or [""]
        visual_lines.extend(wrapped)

    is_user = caller is not None
    out_lines: list[str] = []
    for index, line in enumerate(visual_lines):
        decorated = _decorate_message(line, record.levelno, palette, is_user=is_user)
        if index == 0:
            first = colored_prefix + decorated
            if caller:
                gap = width - len(plain_prefix) - len(line) - len(caller)
                if gap >= 2:
                    first += " " * gap + palette.paint(caller, _DIM)
            out_lines.append(first)
        else:
            out_lines.append(indent + decorated)

    exception = _render_exception_console(record, palette, indent)
    if exception:
        out_lines.append(exception)
    return "\n".join(out_lines)


def _render_exception_console(
    record: logging.LogRecord, palette: _Palette, indent: str
) -> str:
    text = _exception_text(record)
    if not text:
        return ""
    rendered: list[str] = []
    for line in text.split("\n"):
        is_frame = line.startswith((" ", "\t")) or line.startswith(
            ("Traceback", "During handling", "The above")
        )
        style = _DIM if is_frame else f"{_RED};1"
        rendered.append(indent + palette.paint(line, style))
    return "\n".join(rendered)


# --- logfmt renderer -----------------------------------------------------------


def _logfmt_value(value: str) -> str:
    if value == "":
        return '""'
    chars: list[str] = []
    needs_quote = False
    for ch in value:
        if ch == "\\":
            chars.append("\\\\")
            needs_quote = True
        elif ch == '"':
            chars.append('\\"')
            needs_quote = True
        elif ch == "\n":
            chars.append("\\n")
            needs_quote = True
        elif ch == "\r":
            chars.append("\\r")
            needs_quote = True
        elif ch == "\t":
            chars.append("\\t")
            needs_quote = True
        elif ch < " " or "\x7f" <= ch <= "\x9f":
            # C0 (< 0x20), DEL (0x7f) and C1 (0x80-0x9f) controls. C1 matters
            # because some tools treat 0x85 (NEL) as a line terminator, which
            # would split a logfmt record across physical lines.
            chars.append(f"\\x{ord(ch):02x}")
            needs_quote = True
        else:
            chars.append(ch)
            if ch in " =":
                needs_quote = True
    rendered = "".join(chars)
    return f'"{rendered}"' if needs_quote else rendered


def _render_logfmt(record: logging.LogRecord) -> str:
    created = datetime.fromtimestamp(record.created, tz=timezone.utc)
    timestamp = created.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z"

    parts = [timestamp, f"level={record.levelname.lower()}"]
    component = _CURRENT_COMPONENT.get()
    if component:
        parts.append(f"comp={_logfmt_value(component)}")
    parts.append(f"msg={_logfmt_value(record.getMessage())}")

    detail = _record_detail(record)
    if detail:
        for key, value in detail.items():
            parts.append(f"{key}={_logfmt_value(str(value))}")

    caller = _caller_tag(record)
    if caller:
        parts.append(f"caller={caller}")

    line = " ".join(parts)
    exception = _exception_text(record)
    if exception:
        line += "\n" + exception
    return line


class _FuruFormatter(logging.Formatter):
    """Renders the compact coloured console layout on a TTY, logfmt otherwise.

    The file sink always uses logfmt (files are never terminals); the stdout
    sink decides per-record from the live stream so piped output and captured
    test output fall back to logfmt automatically.
    """

    def __init__(self, *, console: bool) -> None:
        super().__init__()
        self._console = console

    def format(self, record: logging.LogRecord) -> str:
        if self._console:
            layout, color = _console_mode(sys.stdout)
            if layout:
                return _render_console(record, color=color)
        return _render_logfmt(record)


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
    stdout_handler.setFormatter(_FuruFormatter(console=True))
    logger.addHandler(stdout_handler)

    file_handler = _ScopedFileHandler()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FuruFormatter(console=False))
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    _base_logger()
    if name is None or name == _BASE_LOGGER_NAME:
        return logging.getLogger(_BASE_LOGGER_NAME)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


def log_detail(**fields: object) -> dict[str, dict[str, object]]:
    """Build an ``extra=`` payload of fields kept out of the console line.

    The fields are dropped from the coloured console output (correlation ids and
    raw counts are screen noise) but appended as ``key=value`` pairs to the
    logfmt file record, which keeps full ids for later inspection.
    """
    return {_DETAIL_ATTR: fields}


@contextmanager
def _scoped_log_files(log_paths: tuple[Path, ...]) -> Iterator[None]:
    token = _CURRENT_LOG_PATHS.set(tuple(dict.fromkeys(log_paths)))
    try:
        yield
    finally:
        _CURRENT_LOG_PATHS.reset(token)


@contextmanager
def _scoped_component(component: str) -> Iterator[None]:
    token = _CURRENT_COMPONENT.set(component)
    try:
        yield
    finally:
        _CURRENT_COMPONENT.reset(token)
