"""Where Furu's output goes: one home per fact, pointers everywhere else.

Three kinds of file exist:

- ``<spec dir>/run.log``: everything one ``create()`` produced, one section per
  attempt, each opened by a ``── <label> · …`` header and closed by a
  ``── ok|failed · <duration>`` footer. In-process, ``_run_log_scope`` holds it
  open and ``_RunLogHandler`` (on the ``furu`` and root loggers) writes the log
  records there; ``print`` stays on your terminal. In a worker child
  (``worker/_child.py``) fds 1 and 2 point at it for the job, so prints,
  logging and C output all land there, including those of processes the job
  starts.
- ``executions/<id>/execution.log``: a worker run's story, written by a
  ``FileHandler`` on the ``furu`` logger for the life of ``run_on_workers``.
  The furu manager keeps its own story the same way in ``manager.log``.
- one file per Slurm worker (sbatch ``--output``): its boot output, what it ran
  where, and why it exited.

The terminal is a view, not a home: Furu logs to stdout at INFO. An in-process
run shows your ``furu.get_logger()`` lines; a worker run shows the run's story
and worker warnings. Every line reads ``<time> <level> [rank<n>] [component]
<message> [→ path] [file:line]``: the component is the logger name minus
``furu.``, the path a pointer to where the details live (from
``extra={"path": p}``), and file:line the call site in your code.

Processes torchrun starts (``WORLD_SIZE`` > 1) tag their lines with their
``RANK``, so ranks sharing a run.log or terminal stay apart: ``grep ' rank3 '``.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import traceback
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TextIO

_BASE_LOGGER_NAME = "furu"
_FURU_PACKAGE_DIR = Path(__file__).resolve().parent

# The run.log of the innermost in-process create() in this context. Like any
# ContextVar it does not cross into threads started inside create().
_RUN_LOG: ContextVar[TextIO | None] = ContextVar("furu_run_log", default=None)

_LEVEL_LETTER = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}
_DIM, _RED, _YELLOW, _MAGENTA, _ORANGE = "2", "31", "33", "35", "38;5;208"
_LEVEL_STYLE = {
    logging.DEBUG: _DIM,
    logging.INFO: "36;1",
    logging.WARNING: f"{_YELLOW};1",
    logging.ERROR: f"{_RED};1",
    logging.CRITICAL: f"{_RED};1",
}


# --- rendering -----------------------------------------------------------------


def _render(record: logging.LogRecord, *, console: bool, color: bool) -> str:
    """One line per record; continuation lines indented 4 spaces.

    Files get a UTC timestamp and the traceback, so ``grep -v '^ '`` drops
    tracebacks and ``grep -E '^\\S+ [WE] '`` finds problems. The console gets
    local time, colour, and pointers only for problems.
    """

    def paint(text: str, style: str) -> str:
        return f"\x1b[{style}m{text}\x1b[0m" if color and style else text

    moment = datetime.fromtimestamp(record.created, tz=UTC)
    level = record.levelno
    caller = _user_caller(record)
    tags = [
        paint(moment.astimezone().strftime("%H:%M:%S"), _DIM)
        if console
        else _utc_timestamp(moment),
        paint(_LEVEL_LETTER.get(level, "?"), _LEVEL_STYLE.get(level, "")),
    ]
    if (rank := _rank()) is not None:
        tags.append(paint(f"rank{rank}", _MAGENTA))
    if (component := _component(record)) is not None:
        tags.append(paint(f"{component:<5}", _MAGENTA))

    # Errors stay red even from your code; otherwise your lines are orange.
    if level >= logging.ERROR:
        style = _RED
    elif caller is not None:
        style = _ORANGE
    else:
        style = {logging.DEBUG: _DIM, logging.WARNING: _YELLOW}.get(level, "")
    first, *rest = record.getMessage().split("\n")
    line = " ".join([*tags, paint(first, style)])
    path = getattr(record, "path", None)
    if path is not None and (not console or level >= logging.WARNING):
        line += paint(f" → {_display_path(path)}", _DIM)
    if caller is not None:
        line += paint(f"  [{caller}]", _DIM)

    rest = [paint(extra, style) for extra in rest]
    if not console:
        if record.exc_info:
            rest += "".join(traceback.format_exception(*record.exc_info)).splitlines()
        elif record.exc_text:
            rest += record.exc_text.splitlines()
        rest += (record.stack_info or "").splitlines()
    return "\n".join([line, *(f"    {extra}" for extra in rest)])


def _rank() -> str | None:
    """This process's torchrun rank, when it is one of several."""
    if os.environ.get("WORLD_SIZE", "1") == "1":
        return None
    return os.environ.get("RANK")


def _component(record: logging.LogRecord) -> str | None:
    if record.name in (_BASE_LOGGER_NAME, "root"):
        return None
    return record.name.removeprefix(f"{_BASE_LOGGER_NAME}.")


def _user_caller(record: logging.LogRecord) -> str | None:
    """``path:line`` of the call site in your code, or None inside Furu."""
    if not _is_user_file(record.pathname):
        return None
    return f"{_display_path(record.pathname)}:{record.lineno}"


@cache
def _is_user_file(pathname: str) -> bool:
    # `<stdin>`, `<string>`, `<frozen ...>` are not files a file:line could name.
    if pathname.startswith("<"):
        return False
    return not Path(pathname).resolve().is_relative_to(_FURU_PACKAGE_DIR)


def _display_path(path: str | os.PathLike[str]) -> str:
    """`path` relative to the working directory when inside it, else as given."""
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _utc_timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _group_label(label: str, count: int) -> str:
    """``label`` for a batch of ``count`` specs led by it."""
    return label + (f" ×{count}" if count > 1 else "")


class _Formatter(logging.Formatter):
    """Renders plain lines for files, or the console view of stdout.

    The stdout sink decides per record from the live stream: a TTY (or
    ``FORCE_COLOR``) gets the console view, coloured unless ``NO_COLOR`` is set;
    piped and captured output gets plain lines.
    """

    def __init__(self, *, console: bool = False) -> None:
        super().__init__()
        self._console = console

    def format(self, record: logging.LogRecord) -> str:
        console = self._console and bool(
            os.environ.get("FORCE_COLOR") or sys.stdout.isatty()
        )
        color = console and os.environ.get("NO_COLOR") is None
        return _render(record, console=console, color=color)


def _plain_handler[H: logging.Handler](handler: H) -> H:
    handler.setFormatter(_Formatter())
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


_run_log_handler = _plain_handler(_RunLogHandler(logging.INFO))
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
            _utc_timestamp(datetime.now(UTC)),
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
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(_Formatter(console=True))
    logger.addHandler(stdout_handler)
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
