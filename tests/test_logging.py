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


def _record(
    msg: str,
    *,
    level: int = logging.INFO,
    pathname: str = "/home/user/datasets.py",
    lineno: int = 64,
    detail: dict[str, object] | None = None,
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
    if detail is not None:
        setattr(record, furu_logging._DETAIL_ATTR, detail)
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


@pytest.fixture(autouse=True)
def _stable_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the terminal width so console-layout assertions don't depend on the
    `COLUMNS` env or the runner's TTY. Tests that exercise wrapping override it."""
    monkeypatch.setattr(
        furu_logging.shutil,
        "get_terminal_size",
        lambda fallback=(100, 24): os.terminal_size((100, 24)),
    )


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


# --- console renderer -------------------------------------------------------


def test_console_layout_time_level_message_and_caller() -> None:
    out = furu_logging._render_console(
        _record("loaded rows", pathname="/home/user/datasets.py", lineno=64),
        color=True,
    )
    plain = _strip_ansi(out)

    assert re.match(r"^\d{2}:\d{2}:\d{2} I ", plain)
    assert "loaded rows" in plain
    assert plain.rstrip().endswith("datasets.py:64")
    assert "\x1b[" in out  # colour present


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
    out = furu_logging._render_console(
        _record("msg", level=level, pathname=furu_logging.__file__), color=False
    )

    assert out.split()[1] == letter


def test_console_shows_component_only_when_scoped() -> None:
    internal = furu_logging.__file__

    with furu_logging._scoped_component("coord"):
        scoped = furu_logging._render_console(
            _record("leased it", pathname=internal), color=False
        )
    unscoped = furu_logging._render_console(
        _record("creating it", pathname=internal), color=False
    )

    assert re.match(r"^\d{2}:\d{2}:\d{2} I coord ", scoped)
    assert re.match(r"^\d{2}:\d{2}:\d{2} I creating", unscoped)


def test_console_omits_caller_for_furu_internal_code() -> None:
    out = furu_logging._render_console(
        _record("leased it", pathname=furu_logging.__file__), color=False
    )

    assert ".py:" not in out


def test_console_omits_caller_for_synthetic_paths() -> None:
    # `<stdin>`/`<string>`/`<frozen ...>` are not real files; tagging them as
    # "your code" would print a misleading file:line.
    out = furu_logging._render_console(
        _record("hi", pathname="<stdin>", lineno=1), color=False
    )

    assert ".py:" not in out
    assert "<stdin>" not in out


def test_console_highlights_artifact_id_and_ok_status() -> None:
    out = furu_logging._render_console(
        _record("finished RawData:9f2a1:8c3d2 ok", pathname=furu_logging.__file__),
        color=True,
    )

    assert "\x1b[32" in out  # green applied to the artifact id / ok status
    assert "RawData:9f2a1:8c3d2" in _strip_ansi(out)


def test_console_colours_error_message_red() -> None:
    out = furu_logging._render_console(
        _record("run failed", level=logging.ERROR, pathname=furu_logging.__file__),
        color=True,
    )

    assert "\x1b[31m" in out  # red message body (distinct from the level letter)


def test_console_colours_user_message_body_orange() -> None:
    out = furu_logging._render_console(
        _record("loaded 1,000 rows", pathname="/home/user/datasets.py", lineno=64),
        color=True,
    )

    assert "\x1b[38;5;208m" in out  # your own message body rendered in orange


def test_console_does_not_colour_furu_internal_message_orange() -> None:
    out = furu_logging._render_console(
        _record("leased RawData:9f2a1:8c3d2", pathname=furu_logging.__file__),
        color=True,
    )

    assert "\x1b[38;5;208m" not in out  # furu lines keep the default treatment


def test_console_user_warning_body_orange_with_level_letter_severity() -> None:
    out = furu_logging._render_console(
        _record(
            "validation AUC 0.81 below target",
            level=logging.WARNING,
            pathname="/home/user/models.py",
            lineno=155,
        ),
        color=True,
    )

    assert "\x1b[38;5;208m" in out  # body stays in the user colour at WARNING
    assert "\x1b[33;1m" in out  # the W letter still carries the warning colour


def test_console_renders_traceback_below_entry() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = furu_logging._render_console(
        _record(
            "failed it",
            level=logging.ERROR,
            pathname=furu_logging.__file__,
            exc_info=exc_info,
        ),
        color=False,
    )

    assert "failed it" in out
    assert "Traceback (most recent call last):" in out
    assert "ValueError: boom" in out


def test_console_wraps_long_message_with_hanging_indent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        furu_logging.shutil,
        "get_terminal_size",
        lambda fallback=(100, 24): os.terminal_size((40, 24)),
    )
    out = furu_logging._render_console(
        _record(
            " ".join(f"word{i}" for i in range(30)),
            pathname="/home/user/train.py",
            lineno=9,
        ),
        color=False,
    )

    lines = out.split("\n")

    assert len(lines) > 1
    prefix_width = len("00:00:00 I ")
    assert lines[1].startswith(" " * prefix_width)  # hanging indent
    assert lines[1].strip()  # message continues, not blank
    assert lines[0].rstrip().endswith("train.py:9")  # caller stays on row one


# --- logfmt renderer --------------------------------------------------------


def test_logfmt_starts_with_timestamp_level_and_message() -> None:
    out = furu_logging._render_logfmt(
        _record("leased it", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z level=info ", out)
    assert 'msg="leased it"' in out
    assert "\x1b[" not in out  # never coloured


def test_logfmt_appends_detail_fields() -> None:
    out = furu_logging._render_logfmt(
        _record(
            "leased it",
            pathname=furu_logging.__file__,
            detail={"lease": "L1", "ready": 2},
        )
    )

    assert "lease=L1" in out
    assert "ready=2" in out


def test_logfmt_includes_scoped_component() -> None:
    with furu_logging._scoped_component("wkr.1"):
        out = furu_logging._render_logfmt(
            _record("leased it", pathname=furu_logging.__file__)
        )

    assert "comp=wkr.1" in out


def test_logfmt_keeps_caller_for_user_code() -> None:
    out = furu_logging._render_logfmt(
        _record("loaded", pathname="/home/user/datasets.py", lineno=64)
    )

    assert "caller=datasets.py:64" in out


def test_logfmt_omits_caller_for_furu_internal_code() -> None:
    out = furu_logging._render_logfmt(
        _record("leased it", pathname=furu_logging.__file__)
    )

    assert "caller=" not in out


def test_logfmt_appends_exception_after_the_record() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = furu_logging._render_logfmt(
        _record(
            "pool stop failed",
            level=logging.ERROR,
            pathname=furu_logging.__file__,
            exc_info=exc_info,
        )
    )

    assert out.splitlines()[0].endswith('msg="pool stop failed"')
    assert "Traceback (most recent call last):" in out
    assert "ValueError: boom" in out


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


def test_logfmt_multiline_detail_value_stays_on_one_line() -> None:
    # A multi-line field value (e.g. a traceback string carried as a detail)
    # must be escaped so the logfmt record stays grep-/parse-stable on a single
    # physical line — the bug that record-splits unescaped newlines is avoided.
    out = furu_logging._render_logfmt(
        _record(
            "job failed",
            pathname=furu_logging.__file__,
            detail={"error": "Traceback\n  File x\nValueError: boom"},
        )
    )

    assert "\n" not in out  # no exc_info → the whole record is one line
    assert "error=" in out
    assert "\\n" in out  # embedded newlines escaped, not emitted raw


# --- helpers ----------------------------------------------------------------


def test_elide_leaves_short_names_unchanged() -> None:
    assert furu_logging._elide("datasets.py", furu_logging._MAX_CALLER_NAME) == (
        "datasets.py"
    )


def test_elide_middle_elides_long_names_keeping_both_ends() -> None:
    elided = furu_logging._elide(
        "gradient_boosting_trainer.py", furu_logging._MAX_CALLER_NAME
    )

    assert "…" in elided
    assert len(elided) <= furu_logging._MAX_CALLER_NAME
    assert elided.startswith("gradient_b")
    assert elided.endswith("trainer.py")


def test_caller_tag_distinguishes_user_code_from_furu() -> None:
    assert (
        furu_logging._caller_tag(_record("x", pathname=furu_logging.__file__)) is None
    )
    assert (
        furu_logging._caller_tag(
            _record("x", pathname="/home/me/datasets.py", lineno=64)
        )
        == "datasets.py:64"
    )


def test_caller_tag_ignores_synthetic_paths() -> None:
    assert furu_logging._caller_tag(_record("x", pathname="<stdin>")) is None
    assert furu_logging._caller_tag(_record("x", pathname="<string>")) is None


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


def test_formatter_falls_back_to_logfmt_when_stdout_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=False))

    out = furu_logging._FuruFormatter(console=True).format(
        _record("hi", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{4}-\d{2}-\d{2}T", out)  # logfmt


def test_formatter_uses_console_layout_for_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))

    out = furu_logging._FuruFormatter(console=True).format(
        _record("hi", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{2}:\d{2}:\d{2} I ", _strip_ansi(out))


def test_file_formatter_is_always_logfmt_even_on_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))

    out = furu_logging._FuruFormatter(console=False).format(
        _record("hi", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{4}-\d{2}-\d{2}T", out)


def test_log_detail_wraps_fields_under_the_detail_attr() -> None:
    assert furu_logging.log_detail(lease="L1", ready=2) == {
        furu_logging._DETAIL_ATTR: {"lease": "L1", "ready": 2}
    }


def test_scoped_component_sets_and_resets_the_context() -> None:
    assert furu_logging._CURRENT_COMPONENT.get() is None
    with furu_logging._scoped_component("coord"):
        assert furu_logging._CURRENT_COMPONENT.get() == "coord"
    assert furu_logging._CURRENT_COMPONENT.get() is None
