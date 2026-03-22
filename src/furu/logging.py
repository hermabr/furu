import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

_BASE_LOGGER_NAME = "furu"
_CURRENT_LOG_PATH: ContextVar[Path | None] = ContextVar(
    "furu_current_log_path", default=None
)


class _ScopedFileHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        log_path = _CURRENT_LOG_PATH.get()
        if log_path is None:
            return

        try:
            rendered = self.format(record)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(rendered)
                f.write("\n")
        except Exception:
            self.handleError(record)


def _base_logger() -> logging.Logger:
    logger = logging.getLogger(_BASE_LOGGER_NAME)
    if not any(isinstance(handler, _ScopedFileHandler) for handler in logger.handlers):
        handler = _ScopedFileHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    _base_logger()
    if name is None or name == _BASE_LOGGER_NAME:
        return logging.getLogger(_BASE_LOGGER_NAME)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


class _ParentScopeLogger:
    def __init__(self, log_path: Path | None) -> None:
        self._log_path = log_path

    def info(self, msg: str, *args: object) -> None:
        if self._log_path is None:
            return
        token = _CURRENT_LOG_PATH.set(self._log_path)
        try:
            get_logger().info(msg, *args)
        finally:
            _CURRENT_LOG_PATH.reset(token)


class ScopedLoggers:
    def __init__(self, *, current: logging.Logger, parent_log_path: Path | None) -> None:
        self.current = current
        self.parent = _ParentScopeLogger(parent_log_path)


@contextmanager
def scoped_loggers(log_path: Path) -> Iterator[ScopedLoggers]:
    _base_logger()
    parent_log_path = _CURRENT_LOG_PATH.get()
    token = _CURRENT_LOG_PATH.set(log_path)
    try:
        yield ScopedLoggers(current=get_logger(), parent_log_path=parent_log_path)
    finally:
        _CURRENT_LOG_PATH.reset(token)
