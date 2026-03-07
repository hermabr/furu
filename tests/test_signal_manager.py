import os
import signal
import time
from multiprocessing import Process, Queue
from pathlib import Path

from furu.signal_manager import handle_termination_signals


def _nested_signal_cleanup(marker_dir: Path, queue: Queue) -> None:
    outer_marker = marker_dir / "outer.txt"
    inner_marker = marker_dir / "inner.txt"

    with handle_termination_signals():
        try:
            with handle_termination_signals():
                try:
                    queue.put(True)
                    time.sleep(30)
                finally:
                    inner_marker.write_text("inner", encoding="utf-8")
        finally:
            outer_marker.write_text("outer", encoding="utf-8")


def test_nested_signal_handlers_run_all_cleanup_layers(tmp_path: Path) -> None:
    queue: Queue = Queue()
    proc = Process(target=_nested_signal_cleanup, args=(tmp_path, queue))
    proc.start()

    try:
        queue.get(timeout=0.5)
        assert proc.pid is not None

        os.kill(proc.pid, signal.SIGTERM)
        proc.join(timeout=0.5)

        assert proc.exitcode == 128 + signal.SIGTERM
        assert (tmp_path / "inner.txt").read_text(encoding="utf-8") == "inner"
        assert (tmp_path / "outer.txt").read_text(encoding="utf-8") == "outer"
    finally:
        proc.join(timeout=0.5)
