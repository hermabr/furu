import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
import sys
from pathlib import Path
from typing import Iterator

from furu.config import get_config

_BASE_LOGGER_NAME = "furu"
# TODO: ContextVar state does not propagate to new threads. Logs emitted from
# worker threads inside create() or create_batched() will use fallback.log
# unless the current log path context is propagated explicitly.
_CURRENT_LOG_PATHS: ContextVar[tuple[Path, ...]] = ContextVar(
    "furu_current_log_paths", default=()
)


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
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if get_config().debug_mode else logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = _ScopedFileHandler()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
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
