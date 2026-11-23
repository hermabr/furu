"""Real processes racing for the same results. Children are forked, so they
share this test's furu config and data directories."""

import os
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path

from runs import events_of

import furu
from furu import Spec
from furu.config import get_config
from furu.execution.load_or_create import _load_or_create
from furu.result.bundle import load_result_bundle
from furu.storage import attempt_lock
from furu.storage._layout import data_dir_in

# Only bounds a hung test; passing runs never wait for it, so keep it generous.
TIMEOUT_S = 10.0
ctx = get_context("fork")


class Probe(Spec):
    key: int
    marker_dir: str

    def create(self) -> int:
        marker_dir = Path(self.marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        return 42


class GatedProbe(Spec):
    key: int
    marker_dir: str
    entered_path: str
    release_path: str

    def create(self) -> int:
        marker_dir = Path(self.marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        Path(self.entered_path).touch()
        while not Path(self.release_path).exists():
            pass
        return 42


class BatchProbe(furu.Spec):
    key: int
    marker_dir: str

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["BatchProbe"], directory: furu.SpecDirectory) -> list[int]:
        marker_dir = Path(objs[0].marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        for obj in objs:
            (marker_dir / f"{obj.key}-{os.getpid()}.marker").write_text("created")
        return [obj.key * 10 for obj in objs]


class MidRunTakeoverProbe(Spec):
    key: int
    entered_path: str
    release_path: str
    recreate_folder: bool = False

    def create(self) -> int:
        data_path = self.directory.data
        Path(self.entered_path).touch()
        while not Path(self.release_path).exists():
            pass
        if self.recreate_folder:
            # Zombie user code recreates the vanished folder path with parents=True.
            (data_path / "zombie").mkdir(parents=True, exist_ok=True)
        return 42


def _worker(marker_dir: str, start_evt, out_q) -> None:
    obj = Probe(key=1, marker_dir=marker_dir)
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        value = obj.create()
        out_q.put(("ok", os.getpid(), value))
    except attempt_lock.HeldError as exc:
        out_q.put(("held", os.getpid(), str(exc)))


def _batch_worker(marker_dir: str, keys: list[int], start_evt, out_q) -> None:
    objs = [BatchProbe(key=key, marker_dir=marker_dir) for key in keys]
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        values = _load_or_create(objs)
        out_q.put(("ok", os.getpid(), values))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _takeover_worker(
    entered_path: str, release_path: str, recreate_folder: bool, out_q
) -> None:
    obj = MidRunTakeoverProbe(
        key=1,
        entered_path=entered_path,
        release_path=release_path,
        recreate_folder=recreate_folder,
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


def _race(target, *argss: tuple) -> list[tuple]:
    """Start one forked child per args, release them together, and collect
    what each puts on the queue after 'ready'."""
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


def test_two_processes_competing_for_same_furu_object(tmp_path: Path) -> None:
    data_dir = get_config().directories.objects
    marker_dir = str(tmp_path / "markers")

    results = _race(_worker, (marker_dir,), (marker_dir,))

    statuses = sorted(result[0] for result in results)
    # Without wait loops, either one wins and the loser fails fast with HeldError,
    # or the first finishes and publishes before the second checks.
    assert statuses in (["held", "ok"], ["ok", "ok"]), results
    assert len(list(Path(marker_dir).glob("*.marker"))) == 1
    manifest_paths = list(data_dir.glob("**/result/manifest.json"))
    assert len(manifest_paths) == 1
    bundle_dir = manifest_paths[0].parent
    assert load_result_bundle(
        bundle_dir,
        [0],
        data_dir=data_dir_in(bundle_dir.parent),
        spec_classes=[Probe],
    ) == [42]


def test_overlapping_batches_in_two_processes_each_publish_one_run(
    tmp_path: Path,
) -> None:
    data_dir = get_config().directories.objects
    marker_dir = tmp_path / "markers"

    results = _race(_batch_worker, (str(marker_dir), [1, 2]), (str(marker_dir), [2, 3]))

    assert [result[0] for result in results] == ["ok", "ok"], results
    assert sorted(result[2] for result in results) == [[10, 20], [20, 30]]
    # Each batch has its own attempt folder, so the spec they share may run twice.
    assert [len(list(marker_dir.glob(f"{key}-*.marker"))) for key in (1, 3)] == [1, 1]
    assert len(list(data_dir.glob("**/result/manifest.json"))) == 2


def test_folder_taken_over_mid_create_does_not_publish(tmp_path: Path) -> None:
    data_dir = get_config().directories.objects
    entered_path = tmp_path / "entered"
    release_path = tmp_path / "release"
    out_q = ctx.Queue()
    proc = _start(_takeover_worker, str(entered_path), str(release_path), False, out_q)

    while not entered_path.exists():
        pass
    (attempt_folder,) = data_dir.glob("**/attempts/*/local__*")
    attempt_folder.rename(attempt_folder.with_name("local__other__999__stolen"))
    release_path.touch()

    result = out_q.get(timeout=TIMEOUT_S)
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0
    assert result[0] == "err"
    assert result[2] == FileNotFoundError.__name__
    assert list(data_dir.glob("**/runs/*")) == []
    probe = MidRunTakeoverProbe(
        key=1, entered_path=str(entered_path), release_path=str(release_path)
    )
    assert [e["event"] for e in events_of(probe)] == ["opened", "start"]


def test_zombie_writer_recreating_folder_is_fenced_by_samestat_on_publish(
    tmp_path: Path,
) -> None:
    data_dir = get_config().directories.objects
    entered_path = tmp_path / "entered"
    release_path = tmp_path / "release"
    out_q = ctx.Queue()
    proc = _start(_takeover_worker, str(entered_path), str(release_path), True, out_q)

    while not entered_path.exists():
        pass
    (attempt_folder,) = data_dir.glob("**/attempts/*/local__*")
    attempt_folder.rename(attempt_folder.with_name("local__other__999__stolen"))
    release_path.touch()

    result = out_q.get(timeout=TIMEOUT_S)
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0
    assert result[0] == "err"
    assert result[2] == attempt_lock.HeldError.__name__
    assert "inode changed" in result[3]
    assert list(data_dir.glob("**/runs/*")) == []
