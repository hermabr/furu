"""short_lock: exclusive creation of a file, and nothing that keeps it alive."""

import os
import threading
import time
from pathlib import Path

import pytest

import furu.locking as locking_module
from furu.locking import short_lock


def test_the_lock_is_a_file_that_is_there_while_the_body_runs(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"

    with short_lock(path):
        assert path.exists()

    assert not path.exists()


def test_the_lock_is_released_when_the_body_raises(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"

    with pytest.raises(ValueError, match="boom"), short_lock(path):
        raise ValueError("boom")

    assert not path.exists()


def test_a_second_taker_waits_until_the_first_is_done(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    order: list[str] = []
    inside = threading.Event()

    def second() -> None:
        inside.wait()
        with short_lock(path):
            order.append("second")

    thread = threading.Thread(target=second)
    thread.start()
    with short_lock(path):
        inside.set()
        time.sleep(4 * locking_module._POLL_S)
        order.append("first")
    thread.join()

    assert order == ["first", "second"]


def test_a_lock_left_by_a_killed_holder_is_taken_once_it_is_old(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    path.touch()
    old = time.time() - locking_module._STALE_S - 1
    os.utime(path, (old, old))

    with short_lock(path):
        assert time.time() - path.stat().st_mtime < locking_module._STALE_S

    assert not path.exists()


def test_a_fresh_lock_is_not_taken_from_its_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "schema.lock"
    path.touch()
    polls = 0

    def release_on_the_third_poll(seconds: float) -> None:
        nonlocal polls
        polls += 1
        assert path.exists()  # still the holder's
        if polls == 3:
            path.unlink()

    monkeypatch.setattr(locking_module.time, "sleep", release_on_the_third_poll)

    with short_lock(path):
        assert polls == 3
