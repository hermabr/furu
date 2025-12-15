import io
import os
import sys
import traceback as _tb


def format_traceback(exc: BaseException) -> str:
    """
    Format an exception traceback for writing to logs.

    Prefers a Rich traceback (box-drawn, readable); falls back to stdlib formatting.
    """
    try:
        from rich.console import Console
        from rich.traceback import Traceback

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
    except Exception:
        return "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)).rstrip()


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
        RED = "\033[31m"
        CYAN = "\033[36m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        lines = _tb.format_exception(type(exc), exc, exc.__traceback__)
        out: list[str] = []
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("Traceback"):
                out.append(f"{BOLD}{RED}{line.rstrip()}{RESET}\n")
            elif line.lstrip().startswith('File "'):
                out.append(f"{CYAN}{line.rstrip()}{RESET}\n")
            else:
                out.append(line)
        sys.stderr.write("".join(out))


def _install_rich_uncaught_exceptions() -> None:
    try:
        from rich.traceback import install as _rich_install  # type: ignore

        _rich_install(show_locals=False)
    except Exception:
        return


_RICH_UNCAUGHT_ENABLED = os.getenv("HULDRA_RICH_UNCAUGHT_TRACEBACKS", "").lower() in {
    "",
    "1",
    "true",
    "yes",
}

# Enable rich tracebacks for uncaught exceptions by default (opt-out via env var).
if _RICH_UNCAUGHT_ENABLED:
    _install_rich_uncaught_exceptions()
