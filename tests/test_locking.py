import os
import time
from pathlib import Path
from threading import Event, Thread

from furu.locking import short_lock


def test_short_lock_waits_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    started, entered = Event(), Event()

    def contender() -> None:
        started.set()
        with short_lock(path):
            entered.set()

    with short_lock(path):
        thread = Thread(target=contender)
        thread.start()
        assert started.wait(5)
        assert not entered.wait(0.1)
    thread.join(5)
    assert entered.is_set()
    assert not path.exists()


def test_short_lock_recovers_from_a_stale_writer(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    path.touch()
    past = time.time() - 61
    os.utime(path, (past, past))
    with short_lock(path):
        assert path.stat().st_mtime > past
    assert not path.exists()
