import logging
import os
import re
import sys
from collections.abc import Iterator
from typing import Any

import pytest

import furu.logging as furu_logging
from furu.config import _set_config, get_config

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _make_record(
    msg: str,
    *,
    level: int = logging.INFO,
    pathname: str = "/home/user/datasets.py",
    lineno: int = 64,
    component: str | None = None,
    fields: dict[str, object] | None = None,
    exc_info: Any = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="furu",
        level=level,
        pathname=pathname,
        lineno=lineno,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    record.furu_component = component
    record.furu_user_caller = furu_logging._user_caller(pathname, lineno)
    if fields is not None:
        record.furu_fields = fields
    return record


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


# --- console formatter ------------------------------------------------------


def test_console_layout_time_level_message_and_caller() -> None:
    formatter = furu_logging._ConsoleFormatter(color=True)
    record = _make_record("loaded rows", pathname="/home/user/datasets.py", lineno=64)

    out = formatter.format(record)
    plain = _strip_ansi(out)

    assert re.match(r"^\d{2}:\d{2}:\d{2} I ", plain)
    assert "loaded rows" in plain
    assert plain.rstrip().endswith("datasets.py:64")
    assert "\x1b[" in out  # color present


@pytest.mark.parametrize(
    ("level", "letter"),
    [
        (logging.DEBUG, "D"),
        (logging.INFO, "I"),
        (logging.WARNING, "W"),
        (logging.ERROR, "E"),
    ],
)
def test_console_one_letter_levels(level: int, letter: str) -> None:
    formatter = furu_logging._ConsoleFormatter(color=False)
    record = _make_record("msg", level=level, pathname=furu_logging.__file__)

    plain = formatter.format(record)

    assert plain.split()[1] == letter


def test_console_component_column_only_when_set() -> None:
    formatter = furu_logging._ConsoleFormatter(color=False)
    internal = furu_logging.__file__

    with_component = formatter.format(
        _make_record("leased Foo:abcde:12345", component="coord", pathname=internal)
    )
    without_component = formatter.format(
        _make_record("creating Foo:abcde:12345", pathname=internal)
    )

    assert re.match(r"^\d{2}:\d{2}:\d{2} I coord ", with_component)
    assert re.match(r"^\d{2}:\d{2}:\d{2} I creating", without_component)


def test_console_furu_internal_lines_have_no_caller() -> None:
    formatter = furu_logging._ConsoleFormatter(color=False)
    record = _make_record("leased Foo:abcde:12345", pathname=furu_logging.__file__)

    plain = formatter.format(record)

    assert "datasets.py" not in plain
    assert ".py:" not in plain


def test_console_colors_artifact_id_and_ok_status() -> None:
    formatter = furu_logging._ConsoleFormatter(color=True)
    record = _make_record(
        "finished RawData:9f2a1:8c3d2 ok · 3.2s", pathname=furu_logging.__file__
    )

    out = formatter.format(record)

    assert furu_logging._ANSI["cyan"] in out  # artifact id
    assert furu_logging._ANSI["green"] in out  # ok
    assert "RawData:9f2a1:8c3d2" in _strip_ansi(out)


def test_console_user_message_is_distinctly_colored() -> None:
    formatter = furu_logging._ConsoleFormatter(color=True)
    record = _make_record(
        "downloaded 1,243,910 rows", pathname="/home/user/datasets.py"
    )

    out = formatter.format(record)

    assert furu_logging._ANSI["orange"] in out


def test_console_error_renders_traceback_with_red_rule() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    formatter = furu_logging._ConsoleFormatter(color=False)
    record = _make_record(
        "failed Foo:abcde:12345",
        level=logging.ERROR,
        pathname=furu_logging.__file__,
        exc_info=exc_info,
    )

    plain = formatter.format(record)

    assert "│" in plain  # red rule tying the traceback to the entry
    assert "Traceback (most recent call last):" in plain
    assert "ValueError: boom" in plain


def test_console_wraps_long_message_with_hanging_indent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        furu_logging.shutil,
        "get_terminal_size",
        lambda fallback=(100, 24): os.terminal_size((40, 24)),
    )
    formatter = furu_logging._ConsoleFormatter(color=False)
    record = _make_record(
        " ".join(f"word{i}" for i in range(30)),
        pathname="/home/user/train.py",
        lineno=9,
    )

    lines = formatter.format(record).split("\n")

    assert len(lines) > 1
    prefix_width = len("00:00:00 I ")
    assert lines[1].startswith(" " * prefix_width)
    assert lines[1].strip()  # message content continues, not blank
    # the file:line tag stays on the first visual row
    assert lines[0].rstrip().endswith("train.py:9")


# --- logfmt formatter -------------------------------------------------------


def test_logfmt_full_timestamp_level_component_and_fields() -> None:
    formatter = furu_logging._LogfmtFormatter()
    record = _make_record(
        "leased Foo:abcde:12345",
        component="wkr·1",
        pathname=furu_logging.__file__,
        fields={"lease": "L1", "ready": 2},
    )

    out = formatter.format(record)

    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z level=info ", out)
    assert "comp=wkr.1" in out  # middot normalised to a dot for logfmt
    assert 'msg="leased Foo:abcde:12345"' in out
    assert "lease=L1" in out
    assert "ready=2" in out
    assert "caller=" not in out  # furu-internal call site
    assert "\x1b[" not in out  # never coloured


def test_logfmt_keeps_caller_for_user_code() -> None:
    formatter = furu_logging._LogfmtFormatter()
    record = _make_record("loaded rows", pathname="/home/user/datasets.py", lineno=64)

    assert "caller=datasets.py:64" in formatter.format(record)


def test_logfmt_carries_exception_on_same_record() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    formatter = furu_logging._LogfmtFormatter()
    record = _make_record(
        "pool stop failed",
        level=logging.ERROR,
        pathname=furu_logging.__file__,
        exc_info=exc_info,
    )

    out = formatter.format(record)

    assert out.splitlines()[0].endswith('msg="pool stop failed"')
    assert "Traceback (most recent call last):" in out
    assert "ValueError: boom" in out


# --- helpers ----------------------------------------------------------------


def test_elide_filename_keeps_both_ends() -> None:
    assert furu_logging._elide_filename("datasets.py") == "datasets.py"
    assert (
        furu_logging._elide_filename("gradient_boosting_trainer.py")
        == "gradient_…rainer.py"
    )
    assert (
        furu_logging._elide_filename("transformer_pretraining_loop.py")
        == "transform…g_loop.py"
    )


def test_user_caller_distinguishes_furu_from_user_code() -> None:
    assert furu_logging._user_caller(furu_logging.__file__, 10) is None
    assert furu_logging._user_caller("/home/me/datasets.py", 64) == "datasets.py:64"


def test_color_respects_no_color_and_force_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert furu_logging._use_color(_FakeStream(tty=True)) is False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert furu_logging._use_color(_FakeStream(tty=False)) is True
    assert furu_logging._console_mode(_FakeStream(tty=False)) is True


def test_non_tty_stream_uses_logfmt_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=False))
    assert isinstance(furu_logging._stdout_formatter(), furu_logging._LogfmtFormatter)

    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))
    assert isinstance(furu_logging._stdout_formatter(), furu_logging._ConsoleFormatter)
