import io
import logging
import re
from collections.abc import Iterator

import pytest

import furu.logging as furu_logging
from furu.config import _set_config, get_config
from furu.logging import (
    _ConsoleFormatter,
    _LogfmtFormatter,
    log_component,
    log_event,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _capture(formatter: logging.Formatter) -> tuple[logging.Logger, io.StringIO]:
    logger = logging.getLogger("furu.test.formatter")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, stream


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


def test_console_layout_time_first_then_level_and_component() -> None:
    logger, stream = _capture(_ConsoleFormatter())
    with log_component("coord"):
        log_event(
            logger,
            logging.INFO,
            "leased",
            event="leased",
            label="RawData:9f2a1:8c3d2",
            lease="x",
        )
    out = _ANSI.sub("", stream.getvalue())
    assert re.match(r"^\d\d:\d\d:\d\d I coord leased RawData:9f2a1:8c3d2", out)


def test_console_omits_component_for_local_runs() -> None:
    logger, stream = _capture(_ConsoleFormatter())
    log_event(logger, logging.INFO, "creating", event="creating", label="X:a:b")
    out = _ANSI.sub("", stream.getvalue())
    assert re.match(r"^\d\d:\d\d:\d\d I creating X:a:b", out)


@pytest.mark.parametrize(
    ("level", "letter"),
    [
        (logging.DEBUG, "D"),
        (logging.INFO, "I"),
        (logging.WARNING, "W"),
        (logging.ERROR, "E"),
    ],
)
def test_console_uses_one_letter_levels(level: int, letter: str) -> None:
    logger, stream = _capture(_ConsoleFormatter())
    logger.log(level, "hello")
    out = _ANSI.sub("", stream.getvalue())
    assert re.search(rf"^\d\d:\d\d:\d\d {letter} ", out)


def test_console_finish_shows_green_ok_and_duration() -> None:
    logger, stream = _capture(_ConsoleFormatter())
    log_event(
        logger,
        logging.INFO,
        "finished",
        event="finished",
        label="X:a:b",
        status="ok",
        duration=3.2,
    )
    out = _ANSI.sub("", stream.getvalue())
    assert "finished X:a:b  ok · 3.2s" in out


def test_console_colors_artifact_id_aqua() -> None:
    logger, stream = _capture(_ConsoleFormatter())
    log_event(logger, logging.INFO, "leased", event="leased", label="X:a:b")
    # Gruvbox aqua truecolor escape around the label.
    assert "\x1b[38;2;142;192;124mX:a:b\x1b[0m" in stream.getvalue()


def test_console_shows_caller_for_user_code_only() -> None:
    logger, stream = _capture(_ConsoleFormatter())
    logger.info("a user-code message")
    out = _ANSI.sub("", stream.getvalue()).rstrip("\n")
    assert re.search(r"test_logging\.py:\d+$", out)


def test_console_styles_traceback_below_error() -> None:
    logger, stream = _capture(_ConsoleFormatter())
    try:
        raise RuntimeError("boom")
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "create failed",
            event="failed",
            label="X:a:b",
            detail=f"{type(exc).__name__}: {exc}",
            status="error",
            exc_info=True,
        )
    out = _ANSI.sub("", stream.getvalue())
    assert "failed X:a:b RuntimeError: boom" in out
    assert "Traceback (most recent call last):" in out
    assert out.rstrip().endswith("RuntimeError: boom")


def test_logfmt_renders_structured_event() -> None:
    logger, stream = _capture(_LogfmtFormatter())
    with log_component("coord"):
        log_event(
            logger,
            logging.INFO,
            "leased",
            event="leased",
            label="RawData:9f2a1:8c3d2",
            lease="550e8400",
            ready=2,
        )
    out = stream.getvalue().strip()
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z level=info ", out)
    assert "comp=coord" in out
    assert "msg=leased" in out
    assert "id=RawData:9f2a1:8c3d2" in out
    assert "lease=550e8400" in out
    assert "ready=2" in out


def test_logfmt_quotes_freeform_message() -> None:
    logger, stream = _capture(_LogfmtFormatter())
    logger.info("loaded 1,234 rows from s3")
    assert 'msg="loaded 1,234 rows from s3"' in stream.getvalue()
