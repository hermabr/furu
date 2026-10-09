"""Where Furu's output goes: one home per fact, pointers everywhere else.

Three kinds of file exist:

- ``<spec dir>/run.log``: everything one ``create()`` produced, one section per
  attempt, each opened by a ``── <label> · …`` header and closed by a
  ``── ok|failed · <duration>`` footer. In-process, ``_run_log_scope`` holds it
  open and ``_RunLogHandler`` (on the ``furu`` and root loggers) writes the log
  records there; ``print`` stays on your terminal. In a worker child
  (``worker/_child.py``) fds 1 and 2 point at it for the job, so prints,
  logging and C output all land there.
- ``executions/<id>/execution.log``: the coordinator's story, written by a
  ``FileHandler`` on the ``furu`` logger for the life of
  ``ExecutionCoordinator.run``.
- one file per Slurm worker (sbatch ``--output``): its boot output, what it ran
  where, and why it exited.

The terminal is a view, not a home: Furu logs to stdout at INFO. An in-process
run shows your ``furu.get_logger()`` lines; a worker run shows the coordinator's
story and worker warnings. The component column is the logger name minus
``furu.``. Records may carry ``extra={"path": p}``, a pointer to where the
details live.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import sys
import textwrap
import threading
import time
import traceback
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TextIO

from furu.config import get_config

_BASE_LOGGER_NAME = "furu"

# The directory that holds furu's own source. Log records whose call site lives
# here are framework-internal and render without a file:line tag; records from
# anywhere else are "your code" and get one.
_FURU_PACKAGE_DIR = Path(__file__).resolve().parent

# The run.log of the innermost in-process create() in this context. Like any
# ContextVar it does not cross into threads started inside create().
_RUN_LOG: ContextVar[TextIO | None] = ContextVar("furu_run_log", default=None)


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

# An artifact id rendered by Spec._log_label: "<ClassName>:<5 chars>:<5 chars>",
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
    FORCE_COLOR) gets the compact console layout; everything else gets plain
    lines. NO_COLOR strips ANSI but keeps the console layout on a TTY.
    """
    force = bool(os.environ.get("FORCE_COLOR"))
    no_color = os.environ.get("NO_COLOR") is not None
    isatty = getattr(stream, "isatty", None)
    is_tty = callable(isatty) and bool(isatty())
    layout = force or is_tty
    return layout, layout and not no_color


# --- shared record helpers -----------------------------------------------------


def _user_pathname(record: logging.LogRecord) -> str | None:
    """The call site's file for user code, or None for furu-internal call sites."""
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
    return pathname


def _caller_tag(record: logging.LogRecord) -> str | None:
    """`file:line` for user code, or None for furu-internal call sites."""
    if (pathname := _user_pathname(record)) is None:
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


def _display_path(path: str | os.PathLike[str]) -> str:
    """`path` relative to the working directory when inside it, else as given."""
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _component(record: logging.LogRecord) -> str | None:
    if record.name in (_BASE_LOGGER_NAME, "root"):
        return None
    return record.name.removeprefix(f"{_BASE_LOGGER_NAME}.")


def _record_path(record: logging.LogRecord) -> str | None:
    path = getattr(record, "path", None)
    return None if path is None else _display_path(path)


def _exception_text(record: logging.LogRecord) -> str:
    if record.exc_info:
        return "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
    if record.exc_text:
        return record.exc_text.rstrip("\n")
    return ""


def _utc_timestamp(created: float) -> str:
    moment = datetime.fromtimestamp(created, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S") + f".{moment.microsecond // 1000:03d}Z"


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
    timestamp = (
        datetime.fromtimestamp(record.created, tz=UTC).astimezone().strftime("%H:%M:%S")
    )
    letter = _LEVEL_LETTER.get(record.levelno, "?")
    component = _component(record)
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

    # Problems point at their details on screen too; other pointers stay in files.
    if record.levelno >= logging.WARNING and (path := _record_path(record)):
        out_lines.append(indent + palette.paint(f"→ {path}", _DIM))

    return "\n".join(out_lines)


# --- plain renderer ------------------------------------------------------------


def _render_plain(record: logging.LogRecord) -> str:
    """One grep-friendly line per record; continuation lines indented 4 spaces.

    `grep -v '^ '` drops tracebacks and `grep -E '^\\S+ [WE] '` finds problems.
    """
    head = f"{_utc_timestamp(record.created)} {_LEVEL_LETTER.get(record.levelno, '?')} "
    if component := _component(record):
        head += f"{component:<5} "
    first, *rest = record.getMessage().split("\n")
    line = head + first
    if path := _record_path(record):
        line += f" → {path}"
    if (pathname := _user_pathname(record)) is not None:
        line += f"  [{_display_path(pathname)}:{record.lineno}]"
    rest += _exception_text(record).splitlines()
    rest += (record.stack_info or "").splitlines()
    return "\n".join([line, *(f"    {extra}" for extra in rest)])


class _Formatter(logging.Formatter):
    """Renders the coloured console layout on a TTY, plain lines otherwise.

    File sinks always render plain lines; the stdout sink decides per record
    from the live stream, so piped and captured output fall back to plain.
    """

    def __init__(self, *, console: bool) -> None:
        super().__init__()
        self._console = console

    def format(self, record: logging.LogRecord) -> str:
        if self._console:
            layout, color = _console_mode(sys.stdout)
            if layout:
                return _render_console(record, color=color)
        return _render_plain(record)


def _plain_handler[H: logging.Handler](handler: H) -> H:
    handler.setFormatter(_Formatter(console=False))
    return handler


# --- run.log -------------------------------------------------------------------


class _RunLogHandler(logging.Handler):
    """Writes records to the run.log of the enclosing `_run_log_scope`, if any."""

    def emit(self, record: logging.LogRecord) -> None:
        if (run_log := _RUN_LOG.get()) is None:
            return
        try:
            run_log.write(self.format(record) + "\n")
            run_log.flush()
        except Exception:  # noqa: BLE001 -- logging.Handler.emit contract: never raise
            self.handleError(record)


_run_log_handler = _plain_handler(_RunLogHandler())
# The root logger belongs to your script, so the handler sits there only while a
# scope is open: outside create() logging.basicConfig and lastResort behave as
# if Furu were not there.
_open_scopes = 0
_open_scopes_lock = threading.Lock()


@contextmanager
def _run_log_scope(path: Path) -> Generator[TextIO]:
    """Send this context's log records, Furu's and stdlib's, to ``path``."""
    global _open_scopes
    _base_logger()
    root = logging.getLogger()
    with path.open("a", encoding="utf-8") as run_log:
        token = _RUN_LOG.set(run_log)
        with _open_scopes_lock:
            _open_scopes += 1
            root.addHandler(_run_log_handler)  # a no-op when already there
        try:
            yield run_log
        finally:
            with _open_scopes_lock:
                _open_scopes -= 1
                if not _open_scopes:
                    root.removeHandler(_run_log_handler)
            _RUN_LOG.reset(token)


def _section_header(label: str, *details: str) -> str:
    """The first line of one attempt's section in a run.log."""
    return " · ".join(
        [
            f"── {label}",
            *details,
            f"host {socket.gethostname()}",
            f"pid {os.getpid()}",
            _utc_timestamp(time.time()),
        ]
    )


def _open_sections(
    run_logs: Sequence[tuple[str, Path]],
    lead: TextIO,
    *details: str,
    notes: Sequence[str] = (),
) -> None:
    """Start an attempt's section in each (label, run.log) of a group.

    The lead's section goes to ``lead`` and receives the output; every other
    member's run.log points at it.
    """
    (lead_label, lead_path), *others = run_logs
    header = _section_header(lead_label, *details)
    lead.write(header + "\n" + "".join(f"   {note}\n" for note in notes))
    lead.flush()
    for label, path in others:
        _append(
            path,
            f"{_section_header(label, *details)}\n"
            f"   batched with {lead_label} → {_display_path(lead_path)}\n",
        )


def _close_sections(
    run_logs: Sequence[tuple[str, Path]], lead: TextIO, outcome: str
) -> None:
    """End the sections `_open_sections` started, e.g. with ``ok · 3ms``."""
    footer = f"── {outcome}\n\n"
    lead.write(footer)
    lead.flush()
    for _, path in run_logs[1:]:
        _append(path, footer)


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(text)


# --- loggers -------------------------------------------------------------------


@cache
def _base_logger() -> logging.Logger:
    logger = logging.getLogger(_BASE_LOGGER_NAME)
    level = logging.DEBUG if get_config().debug_mode else logging.INFO

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(_Formatter(console=True))
    logger.addHandler(stdout_handler)

    _run_log_handler.setLevel(level)
    logger.addHandler(_run_log_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    _base_logger()
    if name is None or name == _BASE_LOGGER_NAME:
        return logging.getLogger(_BASE_LOGGER_NAME)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


def _configure_child_logging() -> None:
    """Send every record in a worker child, once, to stderr: the job's run.log.

    Furu owns this process, so the root logger is set to INFO and stdlib and
    library logging is kept alongside Furu's.
    """
    furu_logger = _base_logger()
    for handler in list(furu_logger.handlers):
        furu_logger.removeHandler(handler)
    furu_logger.setLevel(logging.NOTSET)
    furu_logger.propagate = True
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_plain_handler(logging.StreamHandler(sys.stderr)))
    root.setLevel(logging.INFO)


@contextmanager
def _execution_log(path: Path) -> Generator[None]:
    """Write every Furu record in this process to ``path`` until exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = _plain_handler(logging.FileHandler(path, encoding="utf-8"))
    logger = _base_logger()
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
