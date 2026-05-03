# TODO: make this test not vibe coded
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

from furu import Furu, load_or_create
from furu.config import _FuruDirectories, config
from furu.locking import (
    DEFAULT_ACQUIRE_POLL_INTERVAL_S,
    LockAcquireError,
    LockLostError,
)
from furu.result import load_result_bundle

TEST_TIMING_SCALE = 4.0 if os.environ.get("GITHUB_ACTIONS") == "true" else 1.0
OVERLAP_SLEEP_S = 0.01 * TEST_TIMING_SCALE
POLL_INTERVAL_S = 0.005 * TEST_TIMING_SCALE
PROCESS_TIMEOUT_S = 0.5 * TEST_TIMING_SCALE
MID_CREATE_TIMEOUT_S = 1.0 * TEST_TIMING_SCALE
WAIT_FOR_LOCK_RESULT_TIMEOUT_S = DEFAULT_ACQUIRE_POLL_INTERVAL_S + PROCESS_TIMEOUT_S


class SlowProbe(Furu[int]):
    key: int

    def _create(self) -> int:
        marker_dir = Path(os.environ["FURU_TEST_MARKER_DIR"])
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        time.sleep(OVERLAP_SLEEP_S)  # force overlap window
        return 42


class SlowBatchProbe(Furu[int]):
    key: int

    @classmethod
    def _create_batched(cls, objs) -> list[int]:
        marker_dir = Path(os.environ["FURU_TEST_MARKER_DIR"])
        marker_dir.mkdir(parents=True, exist_ok=True)
        for obj in objs:
            (
                marker_dir / f"{obj.key}-{os.getpid()}-{time.time_ns()}.marker"
            ).write_text("created")
        time.sleep(OVERLAP_SLEEP_S)
        return [obj.key * 10 for obj in objs]


class MidRunTakeoverProbe(Furu[int]):
    key: int
    entered_path: str
    release_path: str

    def _create(self) -> int:
        Path(self.entered_path).touch()
        deadline = time.monotonic() + MID_CREATE_TIMEOUT_S
        while not Path(self.release_path).exists():
            assert time.monotonic() < deadline
            time.sleep(POLL_INTERVAL_S)
        return 42


def _worker(data_dir: str, start_evt, out_q) -> None:
    config.directories = _FuruDirectories(data=Path(data_dir))
    obj = SlowProbe(key=1)
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        value = obj.load_or_create()
        out_q.put(("ok", os.getpid(), value))
    except LockAcquireError as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__))


def _batch_worker(data_dir: str, keys: list[int], start_evt, out_q) -> None:
    config.directories = _FuruDirectories(data=Path(data_dir))
    objs = [SlowBatchProbe(key=key) for key in keys]
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        values = load_or_create(objs)
        out_q.put(("ok", os.getpid(), values))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _takeover_worker(
    data_dir: str,
    entered_path: str,
    release_path: str,
    out_q,
) -> None:
    config.directories = _FuruDirectories(data=Path(data_dir))
    obj = MidRunTakeoverProbe(
        key=1,
        entered_path=entered_path,
        release_path=release_path,
    )
    try:
        value = obj.load_or_create()
        out_q.put(("ok", os.getpid(), value))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _steal_lock(lock_path: str, out_q) -> None:
    lock = Path(lock_path)
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
    out_q.put(("stolen", os.getpid()))


def test_two_processes_competing_for_same_furu_object(tmp_path):
    data_dir = tmp_path / "data"
    marker_dir = tmp_path / "markers"
    os.environ["FURU_TEST_MARKER_DIR"] = str(marker_dir)
    ctx = get_context("spawn")
    start_evt = ctx.Event()
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(str(data_dir), start_evt, out_q))
        for _ in range(2)
    ]
    for proc in procs:
        proc.start()

    ready = [out_q.get(timeout=PROCESS_TIMEOUT_S), out_q.get(timeout=PROCESS_TIMEOUT_S)]
    assert all(tag == "ready" for tag, *_ in ready)
    start_evt.set()

    results = [
        out_q.get(timeout=WAIT_FOR_LOCK_RESULT_TIMEOUT_S),
        out_q.get(timeout=WAIT_FOR_LOCK_RESULT_TIMEOUT_S),
    ]
    for proc in procs:
        proc.join(timeout=WAIT_FOR_LOCK_RESULT_TIMEOUT_S)
        assert proc.exitcode == 0

    oks = [result for result in results if result[0] == "ok"]
    errs = [result for result in results if result[0] == "err"]
    assert len(oks) == 2
    assert errs == []
    assert len(list(marker_dir.glob("*.marker"))) == 1

    manifest_paths = list(data_dir.glob("**/result/manifest.json"))
    assert len(manifest_paths) == 1
    bundle_dir = manifest_paths[0].parent
    assert load_result_bundle(bundle_dir) == 42
    assert list(data_dir.glob("**/result.pkl")) == []


def test_overlapping_batch_acquisitions_do_not_deadlock_or_duplicate_compute(tmp_path):
    data_dir = tmp_path / "data"
    marker_dir = tmp_path / "markers"
    os.environ["FURU_TEST_MARKER_DIR"] = str(marker_dir)
    ctx = get_context("spawn")
    start_evt = ctx.Event()
    out_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_batch_worker,
            args=(str(data_dir), [1, 2], start_evt, out_q),
        ),
        ctx.Process(
            target=_batch_worker,
            args=(str(data_dir), [2, 3], start_evt, out_q),
        ),
    ]
    for proc in procs:
        proc.start()

    ready = [out_q.get(timeout=PROCESS_TIMEOUT_S), out_q.get(timeout=PROCESS_TIMEOUT_S)]
    assert all(tag == "ready" for tag, *_ in ready)
    start_evt.set()

    results = [
        out_q.get(timeout=WAIT_FOR_LOCK_RESULT_TIMEOUT_S),
        out_q.get(timeout=WAIT_FOR_LOCK_RESULT_TIMEOUT_S),
    ]
    for proc in procs:
        proc.join(timeout=WAIT_FOR_LOCK_RESULT_TIMEOUT_S)
        assert proc.exitcode == 0

    assert sorted(result[2] for result in results if result[0] == "ok") == [
        [10, 20],
        [20, 30],
    ]
    assert [result for result in results if result[0] == "err"] == []
    assert len(list(marker_dir.glob("1-*.marker"))) == 1
    assert len(list(marker_dir.glob("2-*.marker"))) == 1
    assert len(list(marker_dir.glob("3-*.marker"))) == 1
    assert len(list(data_dir.glob("**/result/manifest.json"))) == 3
    assert list(data_dir.glob("**/result.pkl")) == []


def test_lock_is_taken_over_mid_create(tmp_path):
    data_dir = tmp_path / "data"
    entered_path = tmp_path / "entered"
    release_path = tmp_path / "release"
    ctx = get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(
        target=_takeover_worker,
        args=(str(data_dir), str(entered_path), str(release_path), out_q),
    )
    proc.start()

    deadline = time.monotonic() + PROCESS_TIMEOUT_S
    while not entered_path.exists():
        assert time.monotonic() < deadline
        time.sleep(POLL_INTERVAL_S)

    lock_paths = list(data_dir.glob("**/compute.lock"))
    assert len(lock_paths) == 1

    steal_q = ctx.Queue()
    stealer = ctx.Process(target=_steal_lock, args=(str(lock_paths[0]), steal_q))
    stealer.start()
    assert steal_q.get(timeout=PROCESS_TIMEOUT_S)[0] == "stolen"
    stealer.join(timeout=PROCESS_TIMEOUT_S)
    assert stealer.exitcode == 0

    release_path.touch()

    result = out_q.get(timeout=PROCESS_TIMEOUT_S)
    proc.join(timeout=PROCESS_TIMEOUT_S)
    assert proc.exitcode == 0
    assert result[0] == "err"
    assert result[2] == LockLostError.__name__
    assert result[3].endswith("before writing final result")
    assert list(data_dir.glob("**/result/manifest.json")) == []
    assert list(data_dir.glob("**/result.pkl")) == []
    assert len(list(data_dir.glob("**/error-*.log"))) == 1
