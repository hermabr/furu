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


@contextmanager
def _scoped_log_file(log_path: Path) -> Iterator[None]:
    _base_logger()
    token = _CURRENT_LOG_PATH.set(log_path)
    try:
        yield
    finally:
        _CURRENT_LOG_PATH.reset(token)


def _log_to_parent(msg: str, *args: object) -> None:
    parent_log_path = _CURRENT_LOG_PATH.get()
    if parent_log_path is None:
        return

    token = _CURRENT_LOG_PATH.set(parent_log_path)
    try:
        get_logger().info(msg, *args)
    finally:
        _CURRENT_LOG_PATH.reset(token)
