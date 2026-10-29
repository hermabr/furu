import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import furu.logging as furu_logging
from furu.config import _set_config, get_config


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="furu",
        level=logging.INFO,
        pathname=furu_logging.__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


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


def test_run_log_holds_one_line_per_record_with_indented_continuations(
    tmp_path: Path,
) -> None:
    run_log = tmp_path / "run.log"
    logger = furu_logging.get_logger("coord")

    with furu_logging._run_log_scope(run_log):
        logger.info("leased it", extra={"path": tmp_path / "objects" / "run.log"})
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("create failed", stack_info=True)

    first, second, *trailer = run_log.read_text(encoding="utf-8").splitlines()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z I ", first)
    assert "leased it" in first
    assert str(tmp_path / "objects" / "run.log") in first
    assert "test_logging.py:" in first  # user call sites are tagged
    assert re.match(r"^\S+ E ", second)
    # Exception and stack follow the record line, indented so grep can drop
    # them, exception first.
    assert all(line.startswith("    ") for line in trailer)
    trailer_text = "\n".join(trailer)
    assert "Traceback (most recent call last):" in trailer_text
    assert trailer_text.index("ValueError: boom") < trailer_text.index(
        "Stack (most recent call last):"
    )


def test_run_log_scope_opens_the_file_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_log = tmp_path / "run.log"
    opened: list[Path] = []
    original_open = Path.open

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    logger = furu_logging.get_logger()

    with furu_logging._run_log_scope(run_log):
        for index in range(5):
            logger.info("record %d", index)
    logger.info("outside the scope")

    assert opened == [run_log]
    text = run_log.read_text()
    assert all(f"record {index}" in text for index in range(5))
    assert "outside the scope" not in text


def test_run_log_scope_captures_stdlib_logging_without_touching_root(
    tmp_path: Path,
) -> None:
    run_log = tmp_path / "run.log"
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    library = logging.getLogger("some_library")

    with furu_logging._run_log_scope(run_log):
        library.warning("weights not used")
        library.info("below the root level")
        assert root.level == level

    assert root.level == level
    assert root.handlers == handlers
    (line,) = run_log.read_text().splitlines()
    assert "weights not used" in line


def test_nested_run_log_scope_restores_the_outer_file(tmp_path: Path) -> None:
    logger = furu_logging.get_logger()

    with furu_logging._run_log_scope(tmp_path / "outer.log"):
        logger.info("before")
        with furu_logging._run_log_scope(tmp_path / "inner.log"):
            logger.info("inside")
        logger.info("after")

    assert "inside" not in (tmp_path / "outer.log").read_text()
    assert "after" in (tmp_path / "outer.log").read_text()
    assert "inside" in (tmp_path / "inner.log").read_text()


@pytest.mark.parametrize(
    ("tty", "no_color", "force_color", "expected"),
    [
        (True, False, False, (True, True)),  # interactive terminal
        (True, True, False, (True, False)),  # NO_COLOR keeps layout, drops colour
        (False, False, False, (False, False)),  # piped / captured → plain
        (False, False, True, (True, True)),  # FORCE_COLOR forces layout + colour
    ],
)
def test_console_mode_respects_tty_and_env(
    monkeypatch: pytest.MonkeyPatch,
    tty: bool,
    no_color: bool,
    force_color: bool,
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    if force_color:
        monkeypatch.setenv("FORCE_COLOR", "1")

    assert furu_logging._console_mode(_FakeStream(tty=tty)) == expected


def test_tty_gets_console_layout_while_files_stay_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))

    console = furu_logging._Formatter(console=True).format(_record("hi"))
    file = furu_logging._Formatter(console=False).format(_record("hi"))

    assert "hi" in console
    assert not re.match(r"^\d{4}-", console)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\S+ I hi$", file)
