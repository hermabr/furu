"""Real processes racing for the same results. Children are forked, so they
share this test's furu config and data directories."""

import json
import os
import time
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

import furu
import furu.execution.load_or_create as load_or_create_module
import furu.locking as locking_module
from furu import Spec
from furu.config import get_config
from furu.execution.load_or_create import _load_or_create
from furu.result.bundle import load_result_bundle
from furu.storage._layout import data_dir_in

OVERLAP_SLEEP_S = 0.01
POLL_INTERVAL_S = 0.005
# Only bounds a hung test; passing runs never wait for it, so keep it generous.
TIMEOUT_S = 10.0
ctx = get_context("fork")


@pytest.fixture(autouse=True)
def _fast_lock_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    # Forked children inherit this, so the loser notices the release quickly.
    monkeypatch.setattr(locking_module, "DEFAULT_ACQUIRE_POLL_INTERVAL_S", 0.005)
    monkeypatch.setattr(load_or_create_module, "_POLL_INTERVAL_S", 0.005)


class SlowProbe(Spec[int]):
    key: int
    marker_dir: str

    def create(self) -> int:
        marker_dir = Path(self.marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        time.sleep(OVERLAP_SLEEP_S)  # force overlap window
        return 42


class SlowBatchProbe(furu.Spec[int]):
    key: int
    marker_dir: str

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["SlowBatchProbe"]) -> list[int]:
        marker_dir = Path(objs[0].marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        for obj in objs:
            (
                marker_dir / f"{obj.key}-{os.getpid()}-{time.time_ns()}.marker"
            ).write_text("created")
        time.sleep(OVERLAP_SLEEP_S)
        return [obj.key * 10 for obj in objs]


class MidRunTakeoverProbe(Spec[int]):
    key: int
    entered_path: str
    release_path: str

    def create(self) -> int:
        Path(self.entered_path).touch()
        deadline = time.monotonic() + TIMEOUT_S
        while not Path(self.release_path).exists():
            assert time.monotonic() < deadline
            time.sleep(POLL_INTERVAL_S)
        return 42


def _worker(marker_dir: str, start_evt, out_q) -> None:
    obj = SlowProbe(key=1, marker_dir=marker_dir)
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        value = obj.create()
        out_q.put(("ok", os.getpid(), value))
    except RuntimeError as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__))


def _batch_worker(marker_dir: str, keys: list[int], start_evt, out_q) -> None:
    objs = [SlowBatchProbe(key=key, marker_dir=marker_dir) for key in keys]
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        values = _load_or_create(objs)
        out_q.put(("ok", os.getpid(), values))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _takeover_worker(entered_path: str, release_path: str, out_q) -> None:
    obj = MidRunTakeoverProbe(
        key=1,
        entered_path=entered_path,
        release_path=release_path,
    )
    try:
        value = obj.create()
        out_q.put(("ok", os.getpid(), value))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _start(target, *args) -> BaseProcess:
    # daemon: a failed test must not leave children that hang pytest at exit.
    proc = ctx.Process(target=target, args=args, daemon=True)
    proc.start()
    return proc


def _steal_lock(lock: Path) -> None:
    claim_path = lock.with_name(f"{lock.name}.stolen.{os.getpid()}.claim").resolve()
    manifest = {
        "claim_path": str(claim_path),
        "lock_paths": [str(lock.resolve())],
    }
    fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    lock.unlink()
    os.link(claim_path, lock)


def _race(target, *argss: tuple) -> list[tuple]:
    """Start one forked child per args, release them together, and collect
    what each puts on the queue after "ready"."""
    start_evt = ctx.Event()
    out_q = ctx.Queue()
    procs = [_start(target, *args, start_evt, out_q) for args in argss]
    assert all(out_q.get(timeout=TIMEOUT_S)[0] == "ready" for _ in procs)
    start_evt.set()
    results = [out_q.get(timeout=TIMEOUT_S) for _ in procs]
    for proc in procs:
        proc.join(timeout=TIMEOUT_S)
        assert proc.exitcode == 0
    return results


def test_two_processes_competing_for_same_furu_object(tmp_path):
    data_dir = get_config().directories.objects
    marker_dir = str(tmp_path / "markers")

    results = _race(_worker, (marker_dir,), (marker_dir,))

    assert [result[0] for result in results] == ["ok", "ok"], results
    assert len(list(Path(marker_dir).glob("*.marker"))) == 1
    manifest_paths = list(data_dir.glob("**/result/manifest.json"))
    assert len(manifest_paths) == 1
    bundle_dir = manifest_paths[0].parent
    assert (
        load_result_bundle(
            bundle_dir, data_dir=data_dir_in(bundle_dir.parent), declared_type=int
        )
        == 42
    )


def test_overlapping_batch_acquisitions_do_not_deadlock_or_duplicate_compute(tmp_path):
    data_dir = get_config().directories.objects
    marker_dir = tmp_path / "markers"

    results = _race(_batch_worker, (str(marker_dir), [1, 2]), (str(marker_dir), [2, 3]))

    assert [result[0] for result in results] == ["ok", "ok"], results
    assert sorted(result[2] for result in results) == [[10, 20], [20, 30]]
    for key in (1, 2, 3):
        assert len(list(marker_dir.glob(f"{key}-*.marker"))) == 1
    assert len(list(data_dir.glob("**/result/manifest.json"))) == 3


def test_lock_is_taken_over_mid_create(tmp_path):
    data_dir = get_config().directories.objects
    entered_path = tmp_path / "entered"
    release_path = tmp_path / "release"
    out_q = ctx.Queue()
    proc = _start(_takeover_worker, str(entered_path), str(release_path), out_q)

    deadline = time.monotonic() + TIMEOUT_S
    while not entered_path.exists():
        assert time.monotonic() < deadline
        time.sleep(POLL_INTERVAL_S)
    (lock_path,) = data_dir.glob("**/compute.lock")
    _steal_lock(lock_path)
    release_path.touch()

    result = out_q.get(timeout=TIMEOUT_S)
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0
    assert result[0] == "err"
    assert result[2] == RuntimeError.__name__
    assert result[3].endswith("before writing final result")
    assert list(data_dir.glob("**/result/manifest.json")) == []
    assert list(data_dir.glob("**/error-*.log")) == []
    (run_log,) = data_dir.glob("**/run.log")
    log_text = run_log.read_text(encoding="utf-8")
    assert "create failed" in log_text
    assert "before writing final result" in log_text
