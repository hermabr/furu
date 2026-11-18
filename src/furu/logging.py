"""Where Furu's output goes: one home per fact, pointers everywhere else.

Three kinds of file exist:

- ``<run folder>/log``: what one run's ``create()`` wrote, every attempt after
  the other, and nothing of Furu's own but the traceback of a ``create()``
  that failed: its bookkeeping is in the ``events.jsonl`` beside it, and its
  log lines go where they do outside a ``create()``. ``_log_scope`` sends
  output there while the working folder is held. In-process,
  ``_RunLogHandler`` (on the ``furu`` and root loggers) writes the log records
  there; ``print`` stays on your terminal. In a worker child
  (``execution/_child.py``) fds 1 and 2 point at it, so prints, logging and C
  output all land there, including those of processes the job starts.
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
``RANK``, so ranks sharing a log or terminal stay apart: ``grep ' rank3 '``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import traceback
from collections.abc import Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TextIO

_BASE_LOGGER_NAME = "furu"
_FURU_PACKAGE_DIR = Path(__file__).resolve().parent

# The log of the in-process create() in this context. Like any ContextVar it
# does not cross into threads started inside create().
_RUN_LOG: ContextVar[TextIO | None] = ContextVar("furu_run_log", default=None)
# In a worker child: the fd its output returns to between jobs.
_child_stderr: int | None = None

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
        paint(record.levelname[:1], _LEVEL_STYLE.get(level, "")),
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
    # `<stdin>`, `<string>`, `<frozen ...>` are not files a file:line could name.
    if record.pathname.startswith("<") or _in_furu(record.pathname):
        return None
    return f"{_display_path(record.pathname)}:{record.lineno}"


@cache
def _in_furu(pathname: str) -> bool:
    return Path(pathname).resolve().is_relative_to(_FURU_PACKAGE_DIR)


def _furus_own(record: logging.LogRecord) -> bool:
    """Whether Furu wrote the record, other than the traceback of a failed create().

    A run's log holds what your code wrote and that traceback; the rest of
    what Furu has to say about a run is in its events.jsonl.
    """
    return record.exc_info is None and _in_furu(record.pathname)


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


# --- a run's log ---------------------------------------------------------------


class _RunLogHandler(logging.Handler):
    """Writes records to the log of the enclosing `_log_scope`, if any."""

    def emit(self, record: logging.LogRecord) -> None:
        if (run_log := _RUN_LOG.get()) is None:
            return
        try:
            run_log.write(self.format(record) + "\n")
            run_log.flush()
        except Exception:  # noqa: BLE001 -- logging.Handler.emit contract: never raise
            self.handleError(record)


_run_log_handler = _plain_handler(_RunLogHandler(logging.INFO))
_run_log_handler.addFilter(lambda record: not _furus_own(record))
# The root logger belongs to your script, so the handler sits there only while a
# scope is open: outside create() logging.basicConfig and lastResort behave as
# if Furu were not there.
_open_scopes = 0
_open_scopes_lock = threading.Lock()


@contextmanager
def _log_scope(path: Path) -> Generator[None]:
    """Send what a create() writes to its run's log at ``path``.

    In a worker child that is everything on fds 1 and 2. Elsewhere it is this
    context's log records, Furu's and stdlib's.
    """
    global _open_scopes
    if _child_stderr is not None:
        with _output_to(path, restore_fd=_child_stderr):
            yield
        return
    _base_logger()
    root = logging.getLogger()
    with path.open("a", encoding="utf-8") as run_log:
        token = _RUN_LOG.set(run_log)
        with _open_scopes_lock:
            _open_scopes += 1
            root.addHandler(_run_log_handler)  # a no-op when already there
        try:
            yield
        finally:
            with _open_scopes_lock:
                _open_scopes -= 1
                if not _open_scopes:
                    root.removeHandler(_run_log_handler)
            _RUN_LOG.reset(token)


def _flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    with suppress(OSError, AttributeError):  # C stdio buffers, where reachable
        ctypes.CDLL(None).fflush(None)


@contextmanager
def _output_to(path: Path, *, restore_fd: int) -> Generator[None]:
    """Point fds 1 and 2 at ``path`` so everything the job writes lands there."""
    _flush()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        _flush()
        os.dup2(restore_fd, 1)
        os.dup2(restore_fd, 2)


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


def _configure_child_logging(worker_stderr: int) -> None:
    """Send every record in a worker child, once, to stderr.

    During a create() that is the run's log, and ``worker_stderr`` (the
    worker's own output) otherwise; Furu's own records always go to the
    latter. Furu owns this process, so the root logger is set to INFO and
    stdlib and library logging is kept alongside Furu's.
    """
    global _child_stderr
    _child_stderr = worker_stderr
    furu_logger = _base_logger()
    for handler in list(furu_logger.handlers):
        furu_logger.removeHandler(handler)
    furu_logger.setLevel(logging.NOTSET)
    furu_logger.propagate = True
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    yours = _plain_handler(logging.StreamHandler(sys.stderr))
    yours.addFilter(lambda record: not _furus_own(record))
    root.addHandler(yours)
    worker_output = os.fdopen(
        worker_stderr, "w", errors="backslashreplace", closefd=False
    )
    furus = _plain_handler(logging.StreamHandler(worker_output))
    furus.addFilter(_furus_own)
    root.addHandler(furus)
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
