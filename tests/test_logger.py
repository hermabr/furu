import json
import logging

import pytest

import huldra
from huldra.runtime.logging import _HuldraRichConsoleHandler


class InternetContent(huldra.Huldra[int]):
    def _create(self) -> int:
        logging.getLogger("internet").info("internet:download")
        (self.huldra_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "value.json").read_text())


class Video(huldra.Huldra[int]):
    content: InternetContent = huldra.chz.field(default_factory=InternetContent)

    def _create(self) -> int:
        logging.getLogger("video").info("video:before")
        self.content.load_or_create()
        logging.getLogger("video").info("video:after")
        (self.huldra_dir / "value.json").write_text(json.dumps(2))
        return 2

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "value.json").read_text())


class SeparatorItem(huldra.Huldra[int]):
    def _create(self) -> int:
        (self.huldra_dir / "value.json").write_text(json.dumps(1))
        return 1

    def _load(self) -> int:
        return json.loads((self.huldra_dir / "value.json").read_text())


def test_log_routes_to_current_holder_dir(huldra_tmp_root) -> None:
    logging.getLogger("video").setLevel(logging.INFO)
    logging.getLogger("internet").setLevel(logging.INFO)

    obj = Video()
    obj.load_or_create()

    video_log = (obj.huldra_dir / ".huldra" / "huldra.log").read_text()
    assert "[DEBUG]" in video_log
    assert "video:before" in video_log
    assert "video:after" in video_log
    assert "internet:download" not in video_log
    assert (
        f"dep: begin {obj.content.__class__.__name__} {obj.content.hexdigest}"
        in video_log
    )
    assert (
        f"dep: end {obj.content.__class__.__name__} {obj.content.hexdigest} (ok)"
        in video_log
    )
    assert video_log.index("video:before") < video_log.index("video:after")

    content_log = (obj.content.huldra_dir / ".huldra" / "huldra.log").read_text()
    assert "[DEBUG]" in content_log
    assert "internet:download" in content_log
    assert "video:before" not in content_log
    assert "video:after" not in content_log


def test_log_without_holder_defaults_to_base_root(huldra_tmp_root) -> None:
    log_path = huldra.log("no-holder")
    assert log_path == huldra.HULDRA_CONFIG.base_root / "huldra.log"
    assert "no-holder" in log_path.read_text()


def test_configure_logging_rich_handler_is_idempotent(huldra_tmp_root) -> None:
    root = logging.getLogger()
    before = sum(isinstance(h, _HuldraRichConsoleHandler) for h in root.handlers)

    huldra.configure_logging()
    after = sum(isinstance(h, _HuldraRichConsoleHandler) for h in root.handlers)
    huldra.configure_logging()
    after2 = sum(isinstance(h, _HuldraRichConsoleHandler) for h in root.handlers)

    assert after >= before
    assert after2 == after


def test_load_or_create_does_not_log_on_cache_hit(
    huldra_tmp_root,
) -> None:
    obj = SeparatorItem()
    obj.load_or_create()
    obj.load_or_create()

    text = (obj.huldra_dir / ".huldra" / "huldra.log").read_text()
    assert text.count("------------------") == 1
    assert text.count("load_or_create ") == 1
    assert f"load_or_create {obj.__class__.__name__} {obj.hexdigest}" in text
    assert str(obj.huldra_dir) in text
    assert text.count("_create: ok ") == 1


def test_rich_console_colors_only_load_or_create_token() -> None:
    pytest.importorskip("rich")

    record = logging.LogRecord(
        name="huldra",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="load_or_create Foo 123 /tmp (success->load)",
        args=(),
        exc_info=None,
    )
    record.huldra_action_color = "green"  # type: ignore[attr-defined]

    text = _HuldraRichConsoleHandler._format_message_text(record)
    assert text.plain == "load_or_create Foo 123 /tmp"
    assert len(text.spans) == 1
    span = text.spans[0]
    assert span.start == 0
    assert span.end == len("load_or_create")
    assert str(span.style) == "green"


def test_rich_console_wraps_location_in_brackets() -> None:
    pytest.importorskip("rich")

    record = logging.LogRecord(
        name="huldra",
        level=logging.INFO,
        pathname=__file__,
        lineno=123,
        msg="hello",
        args=(),
        exc_info=None,
    )
    assert _HuldraRichConsoleHandler._format_location(record) == "[test_logger.py:123]"
