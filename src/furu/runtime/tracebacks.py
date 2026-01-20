import io
import os

from rich.console import Console
from rich.traceback import Traceback


def format_traceback(exc: BaseException) -> str:
    """
    Format an exception traceback for writing to logs.

    Uses Rich traceback (box-drawn, readable).
    """
    buffer = io.StringIO()
    console = Console(file=buffer, record=True, width=120)
    tb = Traceback.from_exception(
        type(exc),
        exc,
        exc.__traceback__,
        show_locals=False,
        width=120,
        extra_lines=3,
        theme="monokai",
        word_wrap=False,
    )
    console.print(tb)
    return console.export_text(styles=False).rstrip()


def _print_colored_traceback(exc: BaseException) -> None:
    """
    Print a full, colored traceback to stderr.
    Uses rich for pretty formatting.
    """
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


def _install_rich_uncaught_exceptions() -> None:
    from rich.traceback import install as _rich_install  # type: ignore

    _rich_install(show_locals=False)


_RICH_UNCAUGHT_ENABLED = os.getenv("FURU_RICH_UNCAUGHT_TRACEBACKS", "").lower() in {
    "",
    "1",
    "true",
    "yes",
}

# Enable rich tracebacks for uncaught exceptions by default (opt-out via env var).
if _RICH_UNCAUGHT_ENABLED:
    _install_rich_uncaught_exceptions()
