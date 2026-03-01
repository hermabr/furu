# TODO: make this test not vibe coded
import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

from furu import Furu
from furu.config import _FuruDirectories, config


class SlowProbe(Furu[int]):
    key: int

    def _create(self) -> int:
        marker_dir = Path(os.environ["FURU_TEST_MARKER_DIR"])
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        time.sleep(0.01)  # force overlap window
        # TODO: make teh sleep times even less if possible
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
    except Exception as e:
        out_q.put(("err", os.getpid(), type(e).__name__))


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
        p.join(timeout=0.1)
        assert p.exitcode == 0
    oks = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    # current behavior: one winner, one loser
    assert len(oks) == 1
    assert len(errs) == 1
    assert errs[0][2] == "NotLockedError"
    # proves _create ran once
    assert len(list(marker_dir.glob("*.marker"))) == 1
    # and winner persisted result
    result_path = (
        data_dir
        / "test_locking"
        / "SlowProbe"
        / SlowProbe(key=1).schema_hash
        / SlowProbe(key=1).artifact_hash
        / "result.pkl"
    )
    assert result_path.exists()
    with result_path.open("rb") as f:
        assert pickle.load(f) == 42
