# TODO: make this test not vibe coded
import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

from furu import Furu
from furu.config import _FuruDirectories, config
from furu.locking import LockAcquireError


class SlowProbe(Furu[int]):
    key: int

    def _create(self) -> int:
        marker_dir = Path(os.environ["FURU_TEST_MARKER_DIR"])
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        time.sleep(0.01)  # force overlap window
        # TODO: make teh sleep times even less if possible
        return 42


class MidRunTakeoverProbe(Furu[int]):
    key: int
    entered_path: str
    release_path: str

    def _create(self) -> int:
        Path(self.entered_path).touch()
        deadline = time.monotonic() + 1.0
        while not Path(self.release_path).exists():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        return 42


def _worker(
    data_dir: str, start_evt, out_q
) -> None:  # TODO: do i need to explicitly set the env variables here?
    config.directories = _FuruDirectories(data=Path(data_dir))
    obj = SlowProbe(key=1)
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        value = obj.load_or_create()
        out_q.put(("ok", os.getpid(), value))
    except LockAcquireError as e:
        out_q.put(("err", os.getpid(), type(e).__name__))


def _takeover_worker(
    data_dir: str, entered_path: str, release_path: str, out_q
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
    except BaseException as e:
        out_q.put(("err", os.getpid(), type(e).__name__, str(e)))


def _steal_lock(lock_path: str, out_q) -> None:
    lock = Path(lock_path)
    claim_path = lock.with_name(f"{lock.name}.stolen.{os.getpid()}.claim")
    fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(claim_path))
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
    for p in procs:
        p.start()
    # wait until both are ready, then release simultaneously
    ready = [
        out_q.get(timeout=0.5),
        out_q.get(timeout=0.5),
    ]  # timeout 0.25 will capture large regressions and should be strict enough
    assert all(tag == "ready" for tag, *_ in ready)
    start_evt.set()
    results = [out_q.get(timeout=0.5), out_q.get(timeout=0.5)]
    for p in procs:
        p.join(timeout=0.5)
        assert p.exitcode == 0
    oks = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    # current behavior: one winner, one loser
    assert len(oks) == 1
    assert len(errs) == 1
    assert errs[0][2] == "LockAcquireError"
    # proves _create ran once
    assert len(list(marker_dir.glob("*.marker"))) == 1
    # and winner persisted result
    result_paths = list(data_dir.glob("**/result.pkl"))
    assert len(result_paths) == 1
    with result_paths[0].open("rb") as f:
        assert pickle.load(f) == 42


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

    deadline = time.monotonic() + 0.5
    while not entered_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.005)

    lock_paths = list(data_dir.glob("**/compute.lock"))
    assert len(lock_paths) == 1

    steal_q = ctx.Queue()
    stealer = ctx.Process(target=_steal_lock, args=(str(lock_paths[0]), steal_q))
    stealer.start()
    assert steal_q.get(timeout=0.5)[0] == "stolen"
    stealer.join(timeout=0.5)
    assert stealer.exitcode == 0

    release_path.touch()

    result = out_q.get(timeout=0.5)
    proc.join(timeout=0.5)
    assert proc.exitcode == 0
    assert result[0] == "err"
    assert result[2] == "NotImplementedError"
    assert result[3] == "TODO: lost result before writing final result"
    assert list(data_dir.glob("**/result.pkl")) == []
    assert len(list(data_dir.glob("**/error-*.log"))) == 1
