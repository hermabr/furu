"""Running jobs outlive their script and their manager; queued work is resent.

Every test here runs a shared manager in this process, with local worker
threads, and makes scripts find it through address.json as Slurm runs do.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import pytest

import furu
import furu.execution.run as run_module
import furu.execution.run_on_workers as run_on_workers_module
from furu import Spec
from furu.config import get_config
from furu.execution.manager import Manager
from furu.storage._layout import run_log_path_in
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend

_serve = Manager.serve


class GatedLeaf(Spec[int]):
    """Notes each create() call, then waits until the test opens the gate."""

    gate: str
    value: int

    def create(self) -> int:
        gate = Path(self.gate)
        (gate / f"started-{os.getpid()}-{time.monotonic_ns()}").touch()
        while not (gate / "open").exists():
            time.sleep(0.02)
        return self.value


def _starts(gate: Path) -> int:
    return len(list(gate.glob("started-*")))


def _wait_until(condition: Callable[[], bool], *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.02)


@contextmanager
def _manager(manager_dir: Path) -> Generator[Manager]:
    with _serve(
        listen_host="127.0.0.1", connect_host="127.0.0.1", manager_dir=None
    ) as manager:
        address = get_config().manager_version_dir / "address.json"
        address.parent.mkdir(exist_ok=True)
        address.write_text(json.dumps({"url": manager.url}))
        yield manager


@pytest.fixture
def manager_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Scripts with local backends use the shared manager, not their own."""
    monkeypatch.setattr(Manager, "serve", lambda **_: nullcontext(None))
    monkeypatch.setattr(run_module, "_WAITING_POLL_INTERVAL_S", 0.1)
    monkeypatch.setattr(run_on_workers_module, "_MANAGER_RETRY_INTERVAL_S", 0.1)
    directory = tmp_path / "manager"
    directory.mkdir()
    with override_config(get_config().model_copy(update={"manager_dir": directory})):
        yield directory


@contextmanager
def _create_in_thread(leaf: GatedLeaf) -> Generator[list[object]]:
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(furu.create(leaf, on=[LocalThreadWorkerBackend()]))
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        yield outcome
    finally:
        thread.join(timeout=30)
        assert not thread.is_alive()


def test_script_that_loses_the_manager_waits_on_its_running_job(
    tmp_path: Path, manager_dir: Path
) -> None:
    leaf = GatedLeaf(gate=str(tmp_path), value=1)
    with _manager(manager_dir) as manager, _create_in_thread(leaf) as outcome:
        _wait_until(lambda: _starts(tmp_path) == 1)
        with manager.lock:
            (first,) = manager.scripts
        first.connection.close()
        # The script plans again from disk and finds its job's lock.
        _wait_until(lambda: [s for s in manager.scripts if s is not first] != [])
        (tmp_path / "open").touch()
    assert outcome == [1]
    assert _starts(tmp_path) == 1


def test_manager_restart_mid_job_lets_the_job_finish(
    tmp_path: Path, manager_dir: Path
) -> None:
    leaf = GatedLeaf(gate=str(tmp_path), value=2)
    with ExitStack() as stack, _create_in_thread(leaf) as outcome:
        first = stack.enter_context(_manager(manager_dir))
        _wait_until(lambda: _starts(tmp_path) == 1)
        # Its worker finishes the job after the manager is gone, so stopping
        # it waits on the gate; stop it in the background.
        stopping = threading.Thread(target=stack.close)
        stopping.start()
        _wait_until(lambda: first.stopping)
        with _manager(manager_dir) as second:
            _wait_until(lambda: second.scripts != [])
            (tmp_path / "open").touch()
            stopping.join(timeout=30)
    assert outcome == [2]
    assert _starts(tmp_path) == 1


_SCRIPT = """
import sys
from contextlib import nullcontext

import furu
import furu.provenance
from furu.execution.manager import Manager
from furu.worker.backends.local import LocalThreadWorkerBackend
from test_manager_restarts import GatedLeaf

furu.provenance._require_pushed_code = False
Manager.serve = lambda **_: nullcontext(None)  # use the test's manager
furu.create(GatedLeaf(gate=sys.argv[1], value=3), on=[LocalThreadWorkerBackend()])
"""


@pytest.mark.parametrize(
    ("sig", "finishes"), [(signal.SIGINT, False), (signal.SIGTERM, True)]
)
def test_ctrl_c_cancels_running_jobs_and_sigterm_leaves_them(
    tmp_path: Path, manager_dir: Path, sig: signal.Signals, finishes: bool
) -> None:
    leaf = GatedLeaf(gate=str(tmp_path), value=3)
    directories = get_config().directories.anchored()
    env = os.environ | {
        "FURU_MANAGER_DIR": str(manager_dir),
        **{
            f"FURU_DIRECTORIES__{name.upper()}": str(path) for name, path in directories
        },
    }
    with _manager(manager_dir) as manager:
        script = subprocess.Popen(
            [sys.executable, "-c", _SCRIPT, str(tmp_path)], env=env
        )
        try:
            _wait_until(lambda: _starts(tmp_path) == 1)
            script.send_signal(sig)
            assert script.wait(timeout=30) != 0
            _wait_until(lambda: not manager.scripts)
            if finishes:
                (tmp_path / "open").touch()
            _wait_until(lambda: not manager.leases)
        finally:
            script.kill()
    run_log = run_log_path_in(leaf._base_dir).read_text()
    if finishes:
        assert leaf.status == "done"
    else:  # the worker killed the job's child
        assert "── died · killed by signal 9" in run_log
    assert _starts(tmp_path) == 1
