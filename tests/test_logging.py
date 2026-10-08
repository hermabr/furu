import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import furu.logging as furu_logging
from furu.config import _Config, _FuruDirectories, _set_config, get_config


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


def test_log_file_holds_one_logfmt_line_per_record(
    isolated_furu_logger: None, tmp_path: Path
) -> None:
    log_file = tmp_path / "run.log"
    logger = furu_logging.get_logger()

    with (
        furu_logging._scoped_log_files((log_file,)),
        furu_logging._scoped_component("slurm-worker-1234567a10"),
    ):
        logger.info(
            "leased it",
            extra=furu_logging.log_detail(lease="L1", error="Trace\n  x\nboom"),
        )
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("create failed", stack_info=True)

    first, second, *trailer = log_file.read_text(encoding="utf-8").splitlines()
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z level=info "
        r'comp=slurm-worker-1234567a10 msg="leased it" lease=L1 '
        r'error="Trace\\n  x\\nboom" caller=test_logging\.py:\d+$',
        first,
    )
    assert re.match(r'^\S+ level=error .*msg="create failed"', second)
    # Exception and stack follow the record line, exception first.
    trailer_text = "\n".join(trailer)
    assert trailer[0] == "Traceback (most recent call last):"
    assert trailer_text.index("ValueError: boom") < trailer_text.index(
        "Stack (most recent call last):"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc", "abc"),
        ("", '""'),
        ("a b", '"a b"'),
        ("a=b", '"a=b"'),
        ('a"b', '"a\\"b"'),
        ("a\\b", '"a\\\\b"'),
        ("a\nb", '"a\\nb"'),
        ("a\tb", '"a\\tb"'),
        ("a\x01b", '"a\\x01b"'),
        ("a\x7fb", '"a\\x7fb"'),  # DEL
        ("a\x85b", '"a\\x85b"'),  # C1 NEL — splits a record if left raw
        ("a\x9bb", '"a\\x9bb"'),  # C1 CSI
        ("a\xa0b", "a\xa0b"),  # NBSP (>= 0xa0) is printable, left untouched
    ],
)
def test_logfmt_value_escaping(value: str, expected: str) -> None:
    assert furu_logging._logfmt_value(value) == expected


def test_unscoped_log_rotates_with_timestamped_name_when_it_reaches_limit(
    isolated_furu_logger: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(furu_logging, "_UNSCOPED_LOG_MAX_BYTES", 4)
    _set_config(
        _Config(
            directories=_FuruDirectories(
                objects=tmp_path / "objects",
                executions=tmp_path / "executions",
                debug=tmp_path / "debug",
            )
        )
    )
    unscoped_log = tmp_path / "objects" / "unscoped.log"
    unscoped_log.parent.mkdir(parents=True)
    unscoped_log.write_text("old\n", encoding="utf-8")

    furu_logging._ScopedFileHandler().emit(_record("new"))

    archived_logs = list(unscoped_log.parent.glob("unscoped-*.log"))
    assert unscoped_log.read_text(encoding="utf-8") == "new\n"
    assert len(archived_logs) == 1
    assert archived_logs[0].read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize(
    ("tty", "no_color", "force_color", "expected"),
    [
        (True, False, False, (True, True)),  # interactive terminal
        (True, True, False, (True, False)),  # NO_COLOR keeps layout, drops colour
        (False, False, False, (False, False)),  # piped / captured → logfmt
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


def test_tty_gets_console_layout_while_files_stay_logfmt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))

    console = furu_logging._FuruFormatter(console=True).format(_record("hi"))
    file = furu_logging._FuruFormatter(console=False).format(_record("hi"))

    assert "hi" in console
    assert "level=" not in console
    assert re.match(r"^\S+ level=info msg=hi$", file)
