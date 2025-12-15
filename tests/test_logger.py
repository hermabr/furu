import json
import logging

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


def test_log_routes_to_current_holder_dir(huldra_tmp_root) -> None:
    logging.getLogger("video").setLevel(logging.INFO)
    logging.getLogger("internet").setLevel(logging.INFO)

    obj = Video()
    obj.load_or_create()

    video_log = (obj.huldra_dir / "huldra.log").read_text()
    assert "[DEBUG]" in video_log
    assert "video:before" in video_log
    assert "video:after" in video_log
    assert "internet:download" not in video_log
    assert video_log.index("video:before") < video_log.index("video:after")

    content_log = (obj.content.huldra_dir / "huldra.log").read_text()
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
