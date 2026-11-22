"""Real processes racing for the same results. Children are forked, so they
share this test's furu config and data directories."""

import os
import socket
import time
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import pytest

import furu
from furu import Spec
from furu.config import get_config
from furu.core import attempt_parent
from furu.execution.load_or_create import _load_or_create
from furu.result.bundle import load_result_bundle
from furu.storage._layout import data_dir_in
from furu.storage.attempt_lock import AttemptHeld

OVERLAP_SLEEP_S = 0.01
POLL_INTERVAL_S = 0.005
# Only bounds a hung test; passing runs never wait for it, so keep it generous.
TIMEOUT_S = 10.0
ctx = get_context("fork")


class SlowProbe(Spec):
    key: int
    marker_dir: str

    def create(self) -> int:
        marker_dir = Path(self.marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"{os.getpid()}.marker").write_text("created")
        time.sleep(OVERLAP_SLEEP_S)  # force overlap window
        return 42


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


class HeldProbe(Spec):
    """Its create() says that it started, then waits to be let go on."""

    key: int
    entered_path: str
    release_path: str

    def create(self) -> int:
        with Path(self.entered_path).open("a") as file:
            file.write(f"{os.getpid()}\n")
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
    except AttemptHeld as exc:
        out_q.put(("held", os.getpid(), str(exc)))


def _batch_worker(marker_dir: str, keys: list[int], start_evt, out_q) -> None:
    objs = [SlowBatchProbe(key=key, marker_dir=marker_dir) for key in keys]
    out_q.put(("ready", os.getpid()))
    start_evt.wait()
    try:
        values = _load_or_create(objs)
        out_q.put(("ok", os.getpid(), values))
    except BaseException as exc:
        out_q.put(("err", os.getpid(), type(exc).__name__, str(exc)))


def _held_worker(entered_path: str, release_path: str, out_q) -> None:
    obj = HeldProbe(key=1, entered_path=entered_path, release_path=release_path)
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


def _holding(tmp_path: Path) -> tuple[HeldProbe, BaseProcess, Any]:
    """A child process in the middle of HeldProbe's create(), with the spec
    and the queue its outcome comes on. Touching the spec's release_path lets
    it go on."""
    entered_path = tmp_path / "entered"
    spec = HeldProbe(
        key=1, entered_path=str(entered_path), release_path=str(tmp_path / "release")
    )
    out_q = ctx.Queue()
    proc = _start(_held_worker, spec.entered_path, spec.release_path, out_q)
    deadline = time.monotonic() + TIMEOUT_S
    while not entered_path.exists():
        assert time.monotonic() < deadline
        time.sleep(POLL_INTERVAL_S)
    return spec, proc, out_q


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

    # Nobody waits for a working folder: the process that finds it taken is
    # told so, unless the run was published before it looked.
    assert "ok" in [result[0] for result in results], results
    assert {result[0] for result in results} <= {"ok", "held"}, results
    assert {result[2] for result in results if result[0] == "ok"} == {42}
    assert list(data_dir.glob("**/attempts/*")) == []
    manifest_paths = list(data_dir.glob("**/result/manifest.json"))
    assert len(manifest_paths) == 1
    bundle_dir = manifest_paths[0].parent
    assert load_result_bundle(
        bundle_dir,
        [0],
        data_dir=data_dir_in(bundle_dir.parent),
        spec_classes=[SlowProbe],
    ) == [42]


def test_overlapping_batches_in_two_processes_each_publish_one_run(tmp_path):
    data_dir = get_config().directories.objects
    marker_dir = tmp_path / "markers"

    results = _race(_batch_worker, (str(marker_dir), [1, 2]), (str(marker_dir), [2, 3]))

    assert [result[0] for result in results] == ["ok", "ok"], results
    assert sorted(result[2] for result in results) == [[10, 20], [20, 30]]
    # Each batch has its own working folder, so the spec they share may be
    # computed twice.
    assert [len(list(marker_dir.glob(f"{key}-*.marker"))) for key in (1, 3)] == [1, 1]
    assert len(list(data_dir.glob("**/result/manifest.json"))) == 2


def test_a_run_another_process_is_working_on_is_refused_until_it_is_published(
    tmp_path,
):
    spec, proc, out_q = _holding(tmp_path)
    folder = attempt_parent([spec])
    (held,) = folder.iterdir()

    with pytest.raises(AttemptHeld) as refused:
        spec.create()

    assert held.name.startswith(f"local__{socket.gethostname()}__{proc.pid}__")
    assert f"{held} is the working folder of process {proc.pid}" in str(refused.value)
    assert f"`mv {held} {folder / 'free__manual'}`" in str(refused.value)
    # Refusing touched nothing of the other process's.
    assert list(folder.iterdir()) == [held]

    Path(spec.release_path).touch()
    assert out_q.get(timeout=TIMEOUT_S) == ("ok", proc.pid, 42)
    proc.join(timeout=TIMEOUT_S)
    assert spec.create() == 42
    assert Path(spec.entered_path).read_text().splitlines() == [str(proc.pid)]


def test_a_process_whose_folder_was_freed_by_hand_cannot_publish_over_its_successor(
    tmp_path,
):
    data_dir = get_config().directories.objects
    spec, proc, out_q = _holding(tmp_path)
    (held,) = attempt_parent([spec]).iterdir()

    # As told by the error for a folder whose process is thought dead; this
    # one is only slow.
    held.rename(held.with_name("free__manual"))
    Path(spec.release_path).touch()

    result = out_q.get(timeout=TIMEOUT_S)
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0
    assert result[0] == "err"
    assert result[2] == FileNotFoundError.__name__
    assert list(data_dir.glob("**/runs/*")) == []
    # What it wrote through the files it had open is in the folder still.
    assert [path.name for path in attempt_parent([spec]).iterdir()] == ["free__manual"]
    (log,) = data_dir.glob("**/attempts/*/free__manual/log")
    assert "create failed" in log.read_text(encoding="utf-8")

    assert spec.create() == 42
    assert Path(spec.entered_path).read_text().splitlines() == [
        str(proc.pid),
        str(os.getpid()),
    ]
    assert len(list(data_dir.glob("**/result/manifest.json"))) == 1
