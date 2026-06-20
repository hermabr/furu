import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import furu.logging as furu_logging
from furu.config import _set_config, get_config

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _reset_furu_logger() -> None:
    logger = logging.getLogger("furu")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    furu_logging._base_logger.cache_clear()


@pytest.fixture
def isolated_furu_logger() -> Iterator[None]:
    original_config = get_config()
    _reset_furu_logger()
    try:
        yield
    finally:
        _set_config(original_config)
        _reset_furu_logger()
        furu_logging.get_logger()


@pytest.mark.parametrize(
    ("debug_mode", "expected_level"),
    [
        (False, logging.INFO),
        (True, logging.DEBUG),
    ],
)
def test_stdout_handler_level_tracks_debug_mode(
    isolated_furu_logger: None,
    debug_mode: bool,
    expected_level: int,
) -> None:
    _set_config(get_config().model_copy(update={"debug_mode": debug_mode}))

    logger = furu_logging.get_logger()

    stdout_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    assert len(stdout_handlers) == 1
    assert stdout_handlers[0].level == expected_level


def test_scoped_file_handler_writes_logfmt_with_user_caller(
    isolated_furu_logger: None,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.log"
    logger = furu_logging.get_logger()

    with furu_logging._scoped_log_files((log_path,)):
        logger.info("hello %s", "world")

    (line,) = log_path.read_text(encoding="utf-8").splitlines()
    assert line.startswith("20")
    assert " level=info " in line
    assert 'msg="hello world"' in line
    assert "caller=test_logging.py:" in line


def test_structured_log_extra_writes_component_and_fields(
    isolated_furu_logger: None,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.log"
    logger = furu_logging.get_logger()

    with (
        furu_logging._scoped_log_files((log_path,)),
        furu_logging.log_component("coord"),
    ):
        logger.info(
            "leased job: lease_id=lease-1 object_id=demo.Task:abcde:12345",
            extra=furu_logging.log_extra(
                event="leased",
                task="Task:abcde:12345",
                fields={
                    "lease": "lease-1",
                    "object_id": "demo.Task:abcde:12345",
                    "ready": 2,
                    "running": 1,
                },
            ),
        )

    (line,) = log_path.read_text(encoding="utf-8").splitlines()
    assert "comp=coord" in line
    assert "msg=leased" in line
    assert "task=Task:abcde:12345" in line
    assert "lease=lease-1" in line
    assert "object_id=demo.Task:abcde:12345" in line
    assert "ready=2 running=1" in line


def test_tty_formatter_uses_compact_colored_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    formatter = furu_logging._FuruConsoleFormatter(_FakeStream(tty=True))
    record = logging.LogRecord(
        "furu",
        logging.INFO,
        __file__,
        123,
        "creating %s",
        ("Task:abcde:12345",),
        None,
    )
    for key, value in furu_logging.log_extra(
        event="creating",
        task="Task:abcde:12345",
    ).items():
        setattr(record, key, value)

    rendered = formatter.format(record)

    assert "\033[" in rendered
    assert "I" in rendered
    assert "creating" in rendered
    assert "Task:abcde:12345" in rendered
    assert "test_logging.py:123" in rendered


def test_color_respects_no_color_and_force_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert furu_logging._console_mode(_FakeStream(tty=True)) == (True, False)

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert furu_logging._console_mode(_FakeStream(tty=False)) == (True, True)


def test_non_tty_formatter_uses_logfmt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    formatter = furu_logging._FuruConsoleFormatter(_FakeStream(tty=False))
    record = logging.LogRecord(
        "furu",
        logging.INFO,
        __file__,
        123,
        "loaded %s",
        ("rows",),
        None,
    )

    rendered = formatter.format(record)

    assert " level=info " in rendered
    assert 'msg="loaded rows"' in rendered
    assert "\033[" not in rendered


def test_no_color_keeps_console_layout_without_ansi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    formatter = furu_logging._FuruConsoleFormatter(_FakeStream(tty=True))
    record = logging.LogRecord(
        "furu",
        logging.INFO,
        __file__,
        123,
        "creating %s",
        ("Task:abcde:12345",),
        None,
    )
    for key, value in furu_logging.log_extra(
        event="creating",
        task="Task:abcde:12345",
    ).items():
        setattr(record, key, value)

    rendered = formatter.format(record)

    assert "\033[" not in rendered
    assert " level=info " not in rendered
    assert _ANSI_RE.sub("", rendered) == rendered
    assert "creating Task:abcde:12345" in rendered


def test_logfmt_quotes_and_escapes_control_characters() -> None:
    assert (
        furu_logging._logfmt_value('path with\t"quotes"\n')
        == '"path with\\t\\"quotes\\"\\n"'
    )
