"""Real processes racing for the same results. Children are forked, so they
share this test's furu config and data directories."""

import os
import time
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path

import furu
from furu import Spec
from furu.config import get_config
from furu.core import attempt_parent
from furu.execution.load_or_create import _load_or_create
from furu.result.bundle import load_result_bundle
from furu.storage import attempt_lock
from furu.storage._layout import data_dir_in

OVERLAP_SLEEP_S = 0.01
POLL_INTERVAL_S = 0.005
# Only bounds a hung test; passing runs never wait for it, so keep it generous.
TIMEOUT_S = 10.0
ctx = get_context("fork")


class SlowBatchProbe(furu.Spec):
    key: int
    marker_dir: str

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(
        objs: list["SlowBatchProbe"], directory: furu.SpecDirectory
    ) -> list[int]:
        marker_dir = Path(objs[0].marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        for obj in objs:
            (
                marker_dir / f"{obj.key}-{os.getpid()}-{time.time_ns()}.marker"
            ).write_text("created")
        time.sleep(OVERLAP_SLEEP_S)
        return [obj.key * 10 for obj in objs]


class HoldsUntilReleased(Spec):
    key: int
    entered_path: str
    release_path: str

    def create(self) -> int:
        (Path(self.entered_path) / f"{os.getpid()}.marker").touch()
        deadline = time.monotonic() + TIMEOUT_S
        while not Path(self.release_path).exists():
            assert time.monotonic() < deadline
            time.sleep(POLL_INTERVAL_S)
        return 42


def _batch_worker(marker_dir: str, keys: list[int], start_evt, out_q) -> None:
    objs = [SlowBatchProbe(key=key, marker_dir=marker_dir) for key in keys]
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        values = _load_or_create(objs)
        out_q.put(("ok", os.getpid(), values))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _holding_worker(entered_path: str, release_path: str, out_q) -> None:
    obj = HoldsUntilReleased(
        key=1, entered_path=entered_path, release_path=release_path
    )
    try:
        out_q.put(("ok", os.getpid(), obj.create()))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not any(path.iterdir()):
        assert time.monotonic() < deadline
        time.sleep(POLL_INTERVAL_S)


def _start(target, *args) -> BaseProcess:
    # daemon: a failed test must not leave children that hang pytest at exit.
    proc = ctx.Process(target=target, args=args, daemon=True)
    proc.start()
    return proc


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


def test_a_second_process_is_told_who_holds_the_run_and_the_first_publishes_it(
    tmp_path,
):
    data_dir = get_config().directories.objects
    entered, release = tmp_path / "entered", tmp_path / "release"
    entered.mkdir()
    out_q = ctx.Queue()
    first = _start(_holding_worker, str(entered), str(release), out_q)
    _wait_for(entered)  # the first is inside create(), holding the folder

    second = _start(_holding_worker, str(entered), str(release), out_q)
    refused = out_q.get(timeout=TIMEOUT_S)
    release.touch()
    published = out_q.get(timeout=TIMEOUT_S)
    for proc in (first, second):
        proc.join(timeout=TIMEOUT_S)
        assert proc.exitcode == 0

    assert refused[:3] == ("err", second.pid, "HeldError")
    assert f"process {first.pid} on " in refused[3]
    assert published == ("ok", first.pid, 42)
    assert [path.name for path in entered.iterdir()] == [f"{first.pid}.marker"]
    (manifest,) = data_dir.glob("**/result/manifest.json")
    assert load_result_bundle(
        manifest.parent,
        [0],
        data_dir=data_dir_in(manifest.parent.parent),
        spec_classes=[HoldsUntilReleased],
    ) == [42]


def test_overlapping_batches_in_two_processes_each_publish_one_run(tmp_path):
    data_dir = get_config().directories.objects
    marker_dir = tmp_path / "markers"

    results = _race(_batch_worker, (str(marker_dir), [1, 2]), (str(marker_dir), [2, 3]))

    assert [result[0] for result in results] == ["ok", "ok"], results
    assert sorted(result[2] for result in results) == [[10, 20], [20, 30]]
    # Each batch has its own lock, so the spec they share may be computed twice.
    assert [len(list(marker_dir.glob(f"{key}-*.marker"))) for key in (1, 3)] == [1, 1]
    assert len(list(data_dir.glob("**/result/manifest.json"))) == 2


def test_a_folder_freed_mid_create_is_taken_by_the_next_and_its_holder_publishes_nothing(
    tmp_path,
):
    entered, release = tmp_path / "entered", tmp_path / "release"
    entered.mkdir()
    out_q = ctx.Queue()
    proc = _start(_holding_worker, str(entered), str(release), out_q)
    _wait_for(entered)

    # A person who took the holder for dead freed its folder, and another
    # process took it.
    obj = HoldsUntilReleased(
        key=1, entered_path=str(entered), release_path=str(release)
    )
    run = attempt_parent([obj])
    (held,) = attempt_lock.entries(run)
    (run / held).rename(run / "free__manual")
    claim = attempt_lock.acquire(run, lambda: False)
    assert claim is not None
    release.touch()

    result = out_q.get(timeout=TIMEOUT_S)
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0
    assert result[:3] == ("err", proc.pid, "FileNotFoundError")
    assert not obj.done
    assert attempt_lock.entries(run) == [claim.path.name]
    attempt_lock.release(claim.path)
