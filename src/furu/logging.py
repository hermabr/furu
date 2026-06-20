from __future__ import annotations

import logging
import re
import shutil
import sys
import textwrap
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from furu.config import get_config

_BASE_LOGGER_NAME = "furu"
_FURU_PACKAGE_DIR = Path(__file__).resolve().parent
_ARTIFACT_ID_RE = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*:"
    r"[0-9a-f]{5,40}:[0-9a-f]{5,40}\b"
)

# TODO: ContextVar state does not propagate to new threads. Logs emitted from
# worker threads inside create() or create_batched() will use fallback.log
# unless the current log path context is propagated explicitly.
_CURRENT_LOG_PATHS: ContextVar[tuple[Path, ...]] = ContextVar(
    "furu_current_log_paths", default=()
)
_CURRENT_LOG_COMPONENT: ContextVar[str | None] = ContextVar(
    "furu_current_log_component", default=None
)

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_COLORS = {
    "debug": "\033[90m",
    "info": "\033[36m",
    "warning": "\033[33m",
    "error": "\033[31m",
    "critical": "\033[31;1m",
    "time": "\033[90m",
    "component": "\033[35m",
    "task": "\033[32m",
    "ok": "\033[32m",
    "user": "\033[38;5;208m",
    "caller": "\033[90m",
}
_LEVEL_LETTERS = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}


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


def _log_extra(
    *,
    event: str | None = None,
    component: str | None = None,
    task: str | None = None,
    status: str | None = None,
    duration_s: float | None = None,
    fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    log_fields = dict(fields or {})
    if task is not None:
        log_fields.setdefault("task", task)
    if status is not None:
        log_fields.setdefault("status", status)
    if duration_s is not None:
        log_fields.setdefault("duration_s", round(duration_s, 3))

    extra: dict[str, object] = {"furu_log_fields": log_fields}
    if event is not None:
        extra["furu_event"] = event
    if component is not None:
        extra["furu_component"] = component
    if task is not None:
        extra["furu_task"] = task
    if status is not None:
        extra["furu_status"] = status
    if duration_s is not None:
        extra["furu_duration_s"] = duration_s
    return extra


def _record_event(record: logging.LogRecord) -> str | None:
    event = getattr(record, "furu_event", None)
    return event if isinstance(event, str) else None


def _record_task(record: logging.LogRecord) -> str | None:
    task = getattr(record, "furu_task", None)
    return task if isinstance(task, str) else None


def _record_status(record: logging.LogRecord) -> str | None:
    status = getattr(record, "furu_status", None)
    return status if isinstance(status, str) else None


def _record_duration_s(record: logging.LogRecord) -> float | None:
    duration_s = getattr(record, "furu_duration_s", None)
    if isinstance(duration_s, int | float):
        return float(duration_s)
    return None


def _record_fields(record: logging.LogRecord) -> dict[str, object]:
    fields = getattr(record, "furu_log_fields", None)
    if not isinstance(fields, Mapping):
        return {}
    return dict(fields)


def _record_component(record: logging.LogRecord) -> str | None:
    component = getattr(record, "furu_component", None)
    if isinstance(component, str) and component:
        return component

    component = _CURRENT_LOG_COMPONENT.get()
    if component:
        return component

    if record.name.startswith(f"{_BASE_LOGGER_NAME}.worker.loop"):
        return "wkr"
    if ".worker.backends.slurm" in record.name:
        return "slurm"
    return None


def _is_furu_internal_path(pathname: str) -> bool:
    try:
        path = Path(pathname).resolve()
    except OSError:
        return False
    return path == _FURU_PACKAGE_DIR or _FURU_PACKAGE_DIR in path.parents


def _middle_elide(value: str, max_len: int = 22) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 5:
        return value[:max_len]
    head = (max_len - 3) // 2
    tail = max_len - 3 - head
    return f"{value[:head]}...{value[-tail:]}"


def _caller(record: logging.LogRecord) -> str | None:
    if _is_furu_internal_path(record.pathname):
        return None
    filename = Path(record.pathname).name or record.filename
    return f"{_middle_elide(filename)}:{record.lineno}"


def _logfmt_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if text and not any(ch.isspace() or ch in '="' for ch in text):
        return text
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _timestamp_logfmt(record: logging.LogRecord) -> str:
    return (
        datetime.fromtimestamp(record.created, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _timestamp_console(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created).strftime("%H:%M:%S")


def _format_duration(duration_s: float) -> str:
    if duration_s < 10:
        return f"{duration_s:.1f}s"
    if duration_s < 60:
        return f"{duration_s:.0f}s"
    minutes, seconds = divmod(duration_s, 60)
    if minutes < 60:
        return f"{minutes:.0f}m{seconds:02.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f}h{minutes:02.0f}m"


def _ansi(text: str, color: str | None) -> str:
    if not color:
        return text
    return f"{color}{text}{_RESET}"


def _console_field_parts(fields: Mapping[str, object]) -> list[str]:
    remaining = dict(fields)
    parts: list[str] = []

    completed = remaining.pop("completed", None)
    total = remaining.pop("total", None)
    if completed is not None and total is not None:
        parts.append(f"{completed}/{total}")

    ready = remaining.pop("ready", None)
    running = remaining.pop("running", None)
    blocked = remaining.pop("blocked", None)
    if ready is not None:
        parts.append(f"{ready} ready")
    if running is not None:
        parts.append(f"{running} running")
    if blocked is not None:
        parts.append(f"{blocked} blocked")

    failed_retry = remaining.pop("failed_retry", None)
    failed = remaining.pop("failed", None)
    if failed_retry:
        parts.append(f"{failed_retry} retry")
    if failed:
        parts.append(f"{failed} failed")

    cached = remaining.pop("cached", None)
    to_build = remaining.pop("to_build", None)
    if cached is not None:
        parts.append(f"{cached} cached")
    if to_build is not None:
        parts.append(f"{to_build} to build")

    # Keep full IDs, paths, and correlation fields in logfmt only.
    for noisy in (
        "duration_s",
        "object_id",
        "object_ids",
        "lease",
        "lease_id",
        "executor_dir",
        "server_url",
        "status",
        "task",
    ):
        remaining.pop(noisy, None)

    for key, value in remaining.items():
        parts.append(f"{key}={value}")
    return parts


def _console_message(record: logging.LogRecord) -> str:
    event = _record_event(record)
    if event is None:
        return record.getMessage()

    fields = _record_fields(record)
    task = _record_task(record)
    status = _record_status(record)
    duration_s = _record_duration_s(record)

    parts = [event]
    if task:
        parts.append(task)
    if status:
        parts.append(status)
    if duration_s is not None:
        parts.append(_format_duration(duration_s))

    data_parts = _console_field_parts(fields)
    if data_parts:
        parts.append("-")
        parts.extend(data_parts)
    return " ".join(parts)


def _colorize_console_message(
    message: str,
    *,
    record: logging.LogRecord,
    has_user_caller: bool,
) -> str:
    if has_user_caller and _record_event(record) is None:
        message = _ansi(message, _COLORS["user"])

    message = _ARTIFACT_ID_RE.sub(
        lambda match: _ansi(match.group(0), _COLORS["task"]),
        message,
    )

    status = _record_status(record)
    if status == "ok":
        message = re.sub(r"\bok\b", _ansi("ok", _COLORS["ok"]), message, count=1)
    elif status:
        message = re.sub(
            rf"\b{re.escape(status)}\b",
            _ansi(status, _COLORS["error"]),
            message,
            count=1,
        )
    return message


class _FuruLogfmtFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = _record_event(record)
        fields: dict[str, object] = {}
        fields["level"] = record.levelname.lower()

        component = _record_component(record)
        if component:
            fields["comp"] = component

        fields["msg"] = event or record.getMessage()
        fields.update(_record_fields(record))

        if caller := _caller(record):
            fields["caller"] = caller

        if record.exc_info:
            fields["exc"] = self.formatException(record.exc_info)

        rendered_fields = " ".join(
            f"{key}={_logfmt_value(value)}"
            for key, value in fields.items()
            if value is not None
        )
        return f"{_timestamp_logfmt(record)} {rendered_fields}"


class _FuruConsoleFormatter(logging.Formatter):
    def __init__(self, stream: Any) -> None:
        super().__init__()
        self._stream = stream
        self._logfmt = _FuruLogfmtFormatter()

    def format(self, record: logging.LogRecord) -> str:
        is_tty = bool(getattr(self._stream, "isatty", lambda: False)())
        if not is_tty:
            return self._logfmt.format(record)

        timestamp = _timestamp_console(record)
        level_letter = _LEVEL_LETTERS.get(record.levelno, record.levelname[:1])
        level_color = _COLORS.get(record.levelname.lower(), _COLORS["info"])
        component = _record_component(record)
        caller = _caller(record)

        prefix_parts = [
            _ansi(timestamp, _COLORS["time"]),
            _ansi(level_letter, f"{level_color}{_BOLD}"),
        ]
        prefix_plain = f"{timestamp} {level_letter}"
        if component:
            prefix_parts.append(_ansi(component, _COLORS["component"]))
            prefix_plain += f" {component}"
        prefix = " ".join(prefix_parts) + " "
        prefix_plain += " "

        message = _console_message(record)
        rendered = self._wrap_console_message(
            prefix=prefix,
            prefix_plain=prefix_plain,
            message=message,
            record=record,
            caller=caller,
        )
        if record.exc_info:
            rendered += "\n" + self._format_console_exception(record.exc_info)
        return rendered

    def _wrap_console_message(
        self,
        *,
        prefix: str,
        prefix_plain: str,
        message: str,
        record: logging.LogRecord,
        caller: str | None,
    ) -> str:
        width = max(60, shutil.get_terminal_size(fallback=(120, 24)).columns)
        has_user_caller = caller is not None
        rows: list[str] = []

        logical_lines = message.splitlines() or [""]
        first_visual_row = True
        for logical_line in logical_lines:
            caller_width = len(caller) + 2 if first_visual_row and caller else 0
            first_width = max(20, width - len(prefix_plain) - caller_width)
            rest_width = max(20, width - len(prefix_plain))
            wrapped = textwrap.wrap(
                logical_line,
                width=first_width,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            extra_wrapped: list[str] = []
            for part in wrapped[1:]:
                extra_wrapped.extend(
                    textwrap.wrap(
                        part,
                        width=rest_width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    or [""]
                )
            wrapped = [wrapped[0], *extra_wrapped]

            for part in wrapped:
                colored = _colorize_console_message(
                    part,
                    record=record,
                    has_user_caller=has_user_caller,
                )
                if first_visual_row:
                    if caller:
                        visible_len = len(prefix_plain) + len(part)
                        pad = max(2, width - visible_len - len(caller))
                        rows.append(
                            prefix
                            + colored
                            + (" " * pad)
                            + _ansi(caller, _COLORS["caller"])
                        )
                    else:
                        rows.append(prefix + colored)
                    first_visual_row = False
                else:
                    rows.append((" " * len(prefix_plain)) + colored)
        return "\n".join(rows)

    def _format_console_exception(self, exc_info: Any) -> str:
        lines = self.formatException(exc_info).splitlines()
        if not lines:
            return ""
        formatted: list[str] = []
        for line in lines[:-1]:
            formatted.append(_ansi(f"  | {line}", _DIM))
        formatted.append(_ansi(f"  | {lines[-1]}", _COLORS["error"]))
        return "\n".join(formatted)


@cache
def _base_logger() -> logging.Logger:
    logger = logging.getLogger(_BASE_LOGGER_NAME)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if get_config().debug_mode else logging.INFO)
    stdout_handler.setFormatter(_FuruConsoleFormatter(sys.stdout))
    logger.addHandler(stdout_handler)

    file_handler = _ScopedFileHandler()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FuruLogfmtFormatter())
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
def _scoped_log_component(component: str | None) -> Iterator[None]:
    token = _CURRENT_LOG_COMPONENT.set(component)
    try:
        yield
    finally:
        _CURRENT_LOG_COMPONENT.reset(token)


@contextmanager
def _scoped_log_files(log_paths: tuple[Path, ...]) -> Iterator[None]:
    token = _CURRENT_LOG_PATHS.set(tuple(dict.fromkeys(log_paths)))
    try:
        yield
    finally:
        _CURRENT_LOG_PATHS.reset(token)
