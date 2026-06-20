import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import furu.logging as furu_logging
from furu.config import _set_config, get_config


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
        furu_logging._scoped_log_component("coord"),
    ):
        logger.info(
            "leased job: lease_id=lease-1 object_id=demo.Task:abcde:12345",
            extra=furu_logging._log_extra(
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


def test_tty_formatter_uses_compact_colored_layout() -> None:
    class TtyStream:
        def isatty(self) -> bool:
            return True

    formatter = furu_logging._FuruConsoleFormatter(TtyStream())
    record = logging.LogRecord(
        "furu",
        logging.INFO,
        __file__,
        123,
        "creating %s",
        ("Task:abcde:12345",),
        None,
    )
    for key, value in furu_logging._log_extra(
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
