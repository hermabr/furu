import logging
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
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
    name: str = "furu",
    path: Path | None = None,
    exc_info: Any = None,
    stack_info: str | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=pathname,
        lineno=lineno,
        msg=msg,
        args=(),
        exc_info=exc_info,
        sinfo=stack_info,
    )
    if path is not None:
        record.path = path
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


def test_console_shows_logger_name_as_component() -> None:
    internal = furu_logging.__file__

    named = furu_logging._render_console(
        _record("leased it", pathname=internal, name="furu.coord"), color=False
    )
    base = furu_logging._render_console(
        _record("creating it", pathname=internal), color=False
    )
    slurm = furu_logging._render_console(
        _record("running it", pathname=internal, name="furu.1234567_10"),
        color=False,
    )

    assert re.match(r"^\d{2}:\d{2}:\d{2} I coord leased it", named)
    assert re.match(r"^\d{2}:\d{2}:\d{2} I creating", base)
    assert re.match(r"^\d{2}:\d{2}:\d{2} I 1234567_10 running it", slurm)


def test_console_shows_path_only_for_warnings_and_errors(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    internal = furu_logging.__file__

    info = furu_logging._render_console(
        _record("leased it", pathname=internal, path=run_log), color=False
    )
    warning = furu_logging._render_console(
        _record("failed it", level=logging.WARNING, pathname=internal, path=run_log),
        color=False,
    )

    assert str(run_log) not in info
    first, second = warning.split("\n")
    assert first.endswith("failed it")
    assert second == " " * len("00:00:00 W ") + f"→ {run_log}"


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


def test_console_leaves_traceback_to_default_error_output() -> None:
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
    assert "Traceback (most recent call last):" not in out
    assert "ValueError: boom" not in out


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


# --- plain renderer ---------------------------------------------------------


def test_plain_line_has_utc_timestamp_level_component_and_message() -> None:
    out = furu_logging._render_plain(
        _record("leased it", pathname=furu_logging.__file__, name="furu.w0")
    )

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z I w0    leased it", out
    )


def test_plain_line_omits_component_for_the_base_logger() -> None:
    out = furu_logging._render_plain(
        _record("creating it", pathname=furu_logging.__file__)
    )

    assert re.fullmatch(r"\S+ I creating it", out)


def test_plain_line_tags_user_code_relative_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    inside = furu_logging._render_plain(
        _record("loaded", pathname=str(tmp_path / "pkg" / "datasets.py"), lineno=64)
    )
    outside = furu_logging._render_plain(
        _record("loaded", pathname="/elsewhere/datasets.py", lineno=64)
    )

    assert inside.endswith("loaded  [pkg/datasets.py:64]")
    assert outside.endswith("loaded  [/elsewhere/datasets.py:64]")
    assert ".py:" not in furu_logging._render_plain(
        _record("leased it", pathname=furu_logging.__file__)
    )


def test_plain_line_appends_path_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    out = furu_logging._render_plain(
        _record(
            "leased it",
            pathname=furu_logging.__file__,
            name="furu.coord",
            path=tmp_path / "objects" / "run.log",
        )
    )

    assert out.endswith("I coord leased it → objects/run.log")


def test_plain_indents_continuation_lines_so_grep_can_drop_them() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = furu_logging._render_plain(
        _record(
            "create failed\nfor two reasons",
            level=logging.ERROR,
            pathname=furu_logging.__file__,
            exc_info=exc_info,
            stack_info="Stack (most recent call last):\n  File x",
        )
    )

    first, *rest = out.split("\n")
    assert re.match(r"^\S+ E create failed$", first)
    assert rest[0] == "    for two reasons"
    assert "    Traceback (most recent call last):" in rest
    assert "    ValueError: boom" in rest
    assert rest[-1] == "      File x"
    assert all(line.startswith("    ") for line in rest)
    assert "\x1b[" not in out


# --- run.log scope ------------------------------------------------------------


def test_run_log_scope_writes_records_with_one_open_per_scope(
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
    lines = run_log.read_text().splitlines()
    assert [line.split(" ", 2)[2].split("  [")[0] for line in lines] == [
        f"record {index}" for index in range(5)
    ]


def test_run_log_scope_captures_stdlib_logging_without_touching_root_level(
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
    assert re.match(r"^\S+ W some_library weights not used  \[", line)


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


def test_sections_frame_the_lead_and_point_batch_members_at_it(
    tmp_path: Path,
) -> None:
    lead_log, member_log = tmp_path / "a" / "run.log", tmp_path / "b" / "run.log"
    run_logs = [("A:00000:00000", lead_log), ("B:00000:00000", member_log)]

    lead_log.parent.mkdir()
    with lead_log.open("a") as lead:
        furu_logging._open_sections(run_logs, lead, "in-process", notes=["a note"])
        lead.write("output\n")
        furu_logging._close_sections(run_logs, lead, "ok · 3ms")

    header, note, output, footer, blank = lead_log.read_text().split("\n")[:5]
    assert re.fullmatch(
        r"── A:00000:00000 · in-process · host \S+ · pid \d+ · \S+Z", header
    )
    assert (note, output, footer, blank) == ("   a note", "output", "── ok · 3ms", "")
    member_lines = member_log.read_text().splitlines()
    assert member_lines[0].startswith("── B:00000:00000 · in-process · host ")
    assert member_lines[1:] == [
        f"   batched with A:00000:00000 → {lead_log}",
        "── ok · 3ms",
        "",
    ]


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


def test_formatter_falls_back_to_plain_when_stdout_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=False))

    out = furu_logging._Formatter(console=True).format(
        _record("hi", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{4}-\d{2}-\d{2}T", out)


def test_formatter_uses_console_layout_for_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))

    out = furu_logging._Formatter(console=True).format(
        _record("hi", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{2}:\d{2}:\d{2} I ", _strip_ansi(out))


def test_file_formatter_is_always_plain_even_on_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))

    out = furu_logging._Formatter(console=False).format(
        _record("hi", pathname=furu_logging.__file__)
    )

    assert re.match(r"^\d{4}-\d{2}-\d{2}T", out)
