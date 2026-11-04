import logging
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


def test_stdout_handler_logs_info(isolated_furu_logger: None) -> None:
    logger = furu_logging.get_logger()

    stdout_handlers = [
        handler
        for handler in logger.handlers
        # Exact type: pytest's LogCaptureHandler subclasses StreamHandler.
        if type(handler) is logging.StreamHandler
    ]
    assert len(stdout_handlers) == 1
    assert stdout_handlers[0].level == logging.INFO


# --- rendering --------------------------------------------------------------

_INTERNAL = furu_logging.__file__


def _plain(record: logging.LogRecord) -> str:
    return furu_logging._render(record, console=False, color=False)


def _console(record: logging.LogRecord, *, color: bool = False) -> str:
    return furu_logging._render(record, console=True, color=color)


@pytest.fixture(autouse=True)
def _not_a_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)


def test_plain_line_has_utc_timestamp_level_component_and_message() -> None:
    out = _plain(_record("leased it", pathname=_INTERNAL, name="furu.w0"))

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z I w0    leased it", out
    )


@pytest.mark.parametrize(
    ("level", "letter"),
    [
        (logging.DEBUG, "D"),
        (logging.INFO, "I"),
        (logging.WARNING, "W"),
        (logging.ERROR, "E"),
    ],
)
def test_lines_use_one_letter_levels(level: int, letter: str) -> None:
    record = _record("msg", level=level, pathname=_INTERNAL)

    assert _plain(record).split()[1] == letter
    assert _console(record).split()[1] == letter


def test_component_is_the_logger_name_minus_furu() -> None:
    def plain(name: str) -> str:
        return _plain(_record("ran it", pathname=_INTERNAL, name=name))

    assert re.fullmatch(r"\S+ I ran it", plain("furu"))
    assert re.fullmatch(r"\S+ I 1234567_10 ran it", plain("furu.1234567_10"))
    assert re.fullmatch(r"\S+ I some_library ran it", plain("some_library"))


def test_console_line_has_local_time_and_no_utc_suffix() -> None:
    out = _console(_record("leased it", pathname=_INTERNAL, name="furu.coord"))

    assert re.fullmatch(r"\d{2}:\d{2}:\d{2} I coord leased it", out)


def test_lines_tag_user_code_relative_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    inside = _record("loaded", pathname=str(tmp_path / "pkg" / "datasets.py"))
    outside = _record("loaded", pathname="/elsewhere/datasets.py")

    assert _plain(inside).endswith("loaded  [pkg/datasets.py:64]")
    assert _console(inside).endswith("loaded  [pkg/datasets.py:64]")
    assert _plain(outside).endswith("loaded  [/elsewhere/datasets.py:64]")


@pytest.mark.parametrize("pathname", [_INTERNAL, "<stdin>", "<string>"])
def test_lines_omit_caller_outside_user_files(pathname: str) -> None:
    out = _plain(_record("leased it", pathname=pathname))

    assert out.endswith("leased it")


def test_plain_line_appends_path_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    out = _plain(
        _record(
            "leased it",
            pathname=_INTERNAL,
            name="furu.coord",
            path=tmp_path / "objects" / "run.log",
        )
    )

    assert out.endswith("I coord leased it → objects/run.log")


def test_console_shows_path_only_for_warnings_and_errors(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"

    info = _console(_record("leased it", pathname=_INTERNAL, path=run_log))
    warning = _console(
        _record("failed it", level=logging.WARNING, pathname=_INTERNAL, path=run_log)
    )

    assert info.endswith("leased it")
    assert warning.endswith(f"failed it → {run_log}")


def test_plain_indents_continuation_lines_so_grep_can_drop_them() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = _plain(
        _record(
            "create failed\nfor two reasons",
            level=logging.ERROR,
            pathname=_INTERNAL,
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


def test_console_leaves_traceback_to_the_files() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = _console(
        _record("failed it", level=logging.ERROR, pathname=_INTERNAL, exc_info=exc_info)
    )

    assert out.endswith("E failed it")


def test_console_colours_user_lines_orange_until_they_are_errors() -> None:
    user = _console(_record("AUC 0.81", level=logging.WARNING), color=True)
    error = _console(_record("diverged", level=logging.ERROR), color=True)
    furu = _console(_record("leased it", pathname=_INTERNAL), color=True)

    assert "\x1b[38;5;208mAUC 0.81" in user
    assert "\x1b[33;1mW" in user  # the level letter still carries severity
    assert "\x1b[31mdiverged" in error
    assert "\x1b[38;5;208m" not in furu
    assert _strip_ansi(user) == _console(_record("AUC 0.81", level=logging.WARNING))


def test_ranks_of_a_torchrun_job_tag_their_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("loss 0.3", pathname=_INTERNAL, name="furu.train")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert " rank" not in _plain(record)

    monkeypatch.setenv("WORLD_SIZE", "8")
    assert _plain(record).split()[1:4] == ["I", "rank3", "train"]
    assert _console(record).split()[1:4] == ["I", "rank3", "train"]


@pytest.mark.parametrize(
    ("tty", "no_color", "force_color", "console", "color"),
    [
        (True, False, False, True, True),  # interactive terminal
        (True, True, False, True, False),  # NO_COLOR keeps the view, drops colour
        (False, False, False, False, False),  # piped / captured → plain
        (False, False, True, True, True),  # FORCE_COLOR forces view + colour
    ],
)
def test_stdout_formatter_follows_tty_and_env(
    monkeypatch: pytest.MonkeyPatch,
    tty: bool,
    no_color: bool,
    force_color: bool,
    console: bool,
    color: bool,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    if force_color:
        monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=tty))
    record = _record("hi", pathname=_INTERNAL)

    assert furu_logging._Formatter(console=True).format(record) == (
        furu_logging._render(record, console=console, color=color)
    )


def test_file_formatter_is_always_plain_even_on_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(furu_logging.sys, "stdout", _FakeStream(tty=True))
    record = _record("hi", pathname=_INTERNAL)

    assert furu_logging._Formatter().format(record) == _plain(record)


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
