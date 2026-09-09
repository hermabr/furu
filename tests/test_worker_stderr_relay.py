from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator

import pytest

from furu import logging as furu_logging
from furu.worker import execute

# Well past the 64 KiB pipe buffer, so an undrained child blocks on write.
_LINES = 3000
_CHILD = f"""
import sys
for i in range({_LINES}):
    sys.stderr.write(f"line {{i:05d}} " + "x" * 60 + "\\n")
"""


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def stalled_file_handler(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Make every file-handler emit block until the returned event is set."""
    release = threading.Event()
    monkeypatch.setattr(
        furu_logging._ScopedFileHandler,
        "emit",
        lambda self, record: release.wait(),
    )
    try:
        yield release
    finally:
        release.set()


@pytest.fixture
def records() -> Iterator[_Records]:
    handler = _Records()
    execute.logger.addHandler(handler)
    try:
        yield handler
    finally:
        execute.logger.removeHandler(handler)


@pytest.fixture
def child() -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, "-c", _CHILD], stderr=subprocess.PIPE, text=True
    )
    try:
        yield process
    finally:
        process.kill()
        process.wait()


def _wait_for(predicate: Callable[[], object], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_pipe_drains_while_file_handler_is_stalled(
    child: subprocess.Popen[str],
    stalled_file_handler: threading.Event,
    records: _Records,
) -> None:
    tail: deque[str] = deque(maxlen=execute._STDERR_TAIL_LINES)
    consumer = execute._relay_stderr(child, tail)

    # The child can only finish if someone keeps reading its stderr.
    assert child.wait(timeout=30) == 0
    _wait_for(lambda: tail and tail[-1].startswith(f"line {_LINES - 1:05d} "))
    # The consumer took one line and is stuck in the file handler with it.
    assert len(records.records) <= 1
    assert consumer.is_alive()

    stalled_file_handler.set()
    consumer.join(timeout=30)
    assert not consumer.is_alive()


def test_dropped_lines_are_counted_once_logging_resumes(
    child: subprocess.Popen[str],
    stalled_file_handler: threading.Event,
    records: _Records,
) -> None:
    tail: deque[str] = deque(maxlen=execute._STDERR_TAIL_LINES)
    consumer = execute._relay_stderr(child, tail)
    assert child.wait(timeout=30) == 0
    _wait_for(lambda: tail and tail[-1].startswith(f"line {_LINES - 1:05d} "))

    stalled_file_handler.set()
    consumer.join(timeout=30)
    assert not consumer.is_alive()

    relayed = [r for r in records.records if r.levelno == logging.INFO]
    dropped = [r for r in records.records if r.levelno == logging.WARNING]
    # One report per stall; a consumer that starts mid-burst stalls on its own
    # first report and files a second one for what dropped meanwhile.
    assert 1 <= len(dropped) <= 2
    n_dropped = sum(
        int(m[1])
        for r in dropped
        if (m := re.fullmatch(r"child \d+: dropped (\d+) .*", r.getMessage()))
    )
    assert len(relayed) + n_dropped == _LINES  # nothing lost, nothing double counted
    assert len(relayed) <= execute._STDERR_QUEUE_LINES + 1
