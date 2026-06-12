import logging
from collections.abc import Iterator

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
