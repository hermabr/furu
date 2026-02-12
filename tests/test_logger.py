import json
import logging
from pathlib import Path

import pytest

import furu
from furu.execution import run_local
from furu.runtime.logging import _FuruRichConsoleHandler


class InternetContent(furu.Furu[int]):
    def _create(self) -> int:
        logging.getLogger("internet").info("internet:download")
        (self.furu_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class Video(furu.Furu[int]):
    content: InternetContent = furu.chz.field(default_factory=InternetContent)

    def _create(self) -> int:
        logging.getLogger("video").info("video:before")
        self.content.get()
        logging.getLogger("video").info("video:after")
        (self.furu_dir / "value.json").write_text(json.dumps(2))
        return 2

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class SeparatorItem(furu.Furu[int]):
    def _create(self) -> int:
        (self.furu_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return json.loads((self.furu_dir / "value.json").read_text())


class LocalExecItem(furu.Furu[int]):
    value: int

    @property
    def _data_path(self) -> Path:
        return self.furu_dir / "value.txt"

    def _create(self) -> int:
        self._data_path.write_text(str(self.value))
        return self.value

    def _load(self) -> int:
        return int(self._data_path.read_text())


class LocalExecParent(furu.Furu[int]):
    child: LocalExecItem

    @property
    def _data_path(self) -> Path:
        return self.furu_dir / "parent.txt"

    def _create(self) -> int:
        result = self.child.get()
        self._data_path.write_text(str(result))
        return result

    def _load(self) -> int:
        return int(self._data_path.read_text())


def test_log_routes_to_current_holder_dir(furu_tmp_root) -> None:
    logging.getLogger("video").setLevel(logging.INFO)
    logging.getLogger("internet").setLevel(logging.INFO)

    obj = Video()
    obj.get()

    video_log = (obj.furu_dir / ".furu" / "furu.log").read_text()
    assert "[DEBUG]" in video_log
    assert "video:before" in video_log
    assert "video:after" in video_log
    assert "internet:download" not in video_log
    assert (
        f"dep: begin {obj.content.__class__.__name__} {obj.content.furu_hash}"
        in video_log
    )
    assert (
        f"dep: end {obj.content.__class__.__name__} {obj.content.furu_hash} (ok)"
        in video_log
    )
    assert video_log.index("video:before") < video_log.index("video:after")

    content_log = (obj.content.furu_dir / ".furu" / "furu.log").read_text()
    assert "[DEBUG]" in content_log
    assert "internet:download" in content_log
    assert "video:before" not in content_log
    assert "video:after" not in content_log


def test_log_without_holder_defaults_to_base_root(furu_tmp_root) -> None:
    log_path = furu.log("no-holder")
    assert log_path == furu.FURU_CONFIG.base_root / "furu.log"
    assert "no-holder" in log_path.read_text()


def test_configure_logging_rich_handler_is_idempotent(furu_tmp_root) -> None:
    root = logging.getLogger()
    before = sum(isinstance(h, _FuruRichConsoleHandler) for h in root.handlers)

    furu.configure_logging()
    after = sum(isinstance(h, _FuruRichConsoleHandler) for h in root.handlers)
    furu.configure_logging()
    after2 = sum(isinstance(h, _FuruRichConsoleHandler) for h in root.handlers)

    assert after >= before
    assert after2 == after


def test_get_does_not_log_on_cache_hit(
    furu_tmp_root,
) -> None:
    obj = SeparatorItem()
    obj.get()
    obj.get()

    text = (obj.furu_dir / ".furu" / "furu.log").read_text()
    assert text.count("------------------") == 1
    assert text.count("get ") == 1
    assert f"get {obj.__class__.__name__} {obj.furu_hash}" in text
    assert str(obj.furu_dir) in text
    assert text.count("_create: ok ") == 1


def test_run_local_logs_to_holder_dir(furu_tmp_root) -> None:
    obj = LocalExecItem(value=1)
    run_local([obj], max_workers=1)

    log_path = obj.furu_dir / ".furu" / "furu.log"
    assert log_path.exists()
    text = log_path.read_text()
    assert "_create: begin" in text
    assert "_create: ok" in text


def test_run_local_logs_dependency_access(furu_tmp_root) -> None:
    child = LocalExecItem(value=1)
    parent = LocalExecParent(child=child)
    run_local([parent], max_workers=1)

    text = (parent.furu_dir / ".furu" / "furu.log").read_text()
    assert f"dep: begin {child.__class__.__name__} {child.furu_hash}" in text
    assert f"dep: end {child.__class__.__name__} {child.furu_hash} (ok)" in text


def test_rich_console_colors_only_get_token() -> None:
    pytest.importorskip("rich")

    record = logging.LogRecord(
        name="furu",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="get Foo 123 /tmp (success->load)",
        args=(),
        exc_info=None,
    )
    record.furu_action_color = "green"

    text = _FuruRichConsoleHandler._format_message_text(record)
    assert text.plain == "get Foo 123 /tmp"
    assert len(text.spans) == 1
    span = text.spans[0]
    assert span.start == 0
    assert span.end == len("get")
    assert str(span.style) == "green"


def test_rich_console_wraps_location_in_brackets() -> None:
    pytest.importorskip("rich")

    record = logging.LogRecord(
        name="furu",
        level=logging.INFO,
        pathname=__file__,
        lineno=123,
        msg="hello",
        args=(),
        exc_info=None,
    )
    assert _FuruRichConsoleHandler._format_location(record) == "[test_logger.py:123]"
