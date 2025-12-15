import sys
import traceback as _tb


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


# Preserve previous behavior: enable rich tracebacks for uncaught exceptions if available.
_install_rich_uncaught_exceptions()
