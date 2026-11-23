"""A run does its work in attempts/ and is published into runs/ by one rename.

A batch is one run with a row per spec, found again by listing runs/.
"""

import errno
import os
import re
import shutil
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from runs import ends_of, events_of, fixed_run, run_of, working_folder

import furu
from furu import Spec, plan
from furu.config import get_config
from furu.core import attempt_parent
from furu.dependencies import creating
from furu.execution import load_or_create
from furu.storage import attempt_lock, stored
from furu.storage._layout import (
    FIXED,
    attempts_dir_in,
    events_path_in,
    log_path_in,
    run_name,
    runs_dir_in,
)


class Leaf(Spec):
    n: int

    def create(self) -> int:
        return self.n


class Train(Spec):
    seed: int

    @furu.batched(lambda spec, worker: (None, 100))
    def create(specs: list["Train"], directory: furu.SpecDirectory) -> list[float]:
        return [spec.seed / 10 for spec in specs]


class Weights(Spec):
    seed: int
    folders: ClassVar[list[str]] = []

    @furu.batched(lambda spec, worker: (None, 100))
    def create(specs: list["Weights"], directory: furu.SpecDirectory) -> list[Any]:
        Weights.folders += attempt_lock.entries(attempt_parent(specs))
        return [np.full(3, spec.seed) for spec in specs]


class Merge(Spec):
    n: int

    def create(self) -> int:
        return self.n


class BatchedMerge(Spec):
    """Merge once it has the decorator: the same class as far as ids go."""

    n: int

    @furu.batched(lambda spec, worker: (None, 100))
    def create(specs: list["BatchedMerge"], directory: furu.SpecDirectory) -> list[int]:
        return [spec.n for spec in specs]


BatchedMerge.__qualname__ = "Merge"


def _launch_plan() -> Path:
    return get_config().directories.anchored().plans / "launch.plan"


def _compute_again(specs: Sequence[Spec], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``specs`` as one batch, as a worker does that does not see them stored."""
    with monkeypatch.context() as patched, creating(plan=_launch_plan()):
        patched.setattr(stored.Stored, "row", lambda self, obj: None)
        load_or_create._ensure_group_result(specs)


class FailsAsABatch(Spec):
    n: int

    @furu.batched(lambda spec, worker: (None, 100))
    def create(
        specs: list["FailsAsABatch"], directory: furu.SpecDirectory
    ) -> list[int]:
        if len(specs) > 1:
            raise ValueError("boom")
        return [spec.n for spec in specs]


class LosesItsFolder(Spec):
    def create(self) -> int:
        # As if someone took the folder for dead and freed it by hand.
        folder = working_folder(self)
        folder.rename(folder.with_name("free__manual"))
        return 1


class LeavesScratch(Spec):
    def create(self) -> int:
        (self.directory.scratch / "download.csv").write_text("big")
        return 1


class BlocksThenFailsOnce(Spec):
    def create(self) -> int:
        value = Leaf(n=7).create()
        failed = self.directory.scratch / "failed"
        if not failed.exists():
            failed.touch()
            raise ValueError("bad shape\nthe second line stays in the log")
        return value


def test_one_specs_run_is_named_by_its_hash_and_a_batch_by_its_row_order() -> None:
    assert run_name(["03be"]) == "03be"
    assert run_name(["03be", "9c02"]) != run_name(["9c02", "03be"])
    assert run_name(["03be", "9c02"]).startswith("batch-")


def test_a_run_is_whole_and_still_its_workers_when_one_rename_publishes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=1)
    seen: dict[str, object] = {}
    rename = os.rename

    def spy(self: Path, target: Path) -> None:
        if target == fixed_run(spec):
            seen["files"] = sorted(path.name for path in self.iterdir())
            seen["ends"] = ends_of(spec)
            seen["holder"] = self.name.rpartition("__")[0]
            seen["free"] = attempt_lock.held(self.parent) is None
            seen["done"] = spec.done
        rename(self, target)

    monkeypatch.setattr(os, "rename", spy)

    assert spec.create() == 1

    files = ["events.jsonl", "log", "result"]
    assert seen == {
        "files": files,
        "ends": [{"outcome": "ok"}],
        "holder": attempt_lock.me(),
        "free": False,
        "done": False,
    }
    assert sorted(path.name for path in run_of(spec).iterdir()) == files
    # The working folder moved, and the folder that held it went with it.
    assert list(attempts_dir_in(spec._class_dir).iterdir()) == []


def test_a_crash_before_the_rename_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=2)

    rename = os.rename

    def crash(self: Path, target: Path) -> None:
        if target == fixed_run(spec):
            raise RuntimeError("killed before the rename")
        rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(os, "rename", crash)
        with pytest.raises(RuntimeError, match="killed before the rename"):
            spec.create()

    assert not spec.done
    assert not fixed_run(spec).exists()
    # Given up whole, for the next worker to take.
    assert working_folder(spec).name.startswith("free__")
    assert (working_folder(spec) / "result").is_dir()

    assert spec.create() == 2


@pytest.mark.parametrize(
    "code", [errno.ENOENT, errno.ESTALE, errno.EIO, errno.EDQUOT, errno.ENOTEMPTY]
)
def test_a_rename_that_fails_with_no_run_published_keeps_the_only_copy(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    spec = Leaf(n=5)
    rename = os.rename

    def fail(self: Path, target: Path) -> None:
        if target == fixed_run(spec):
            raise OSError(code, os.strerror(code))
        rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(os, "rename", fail)
        with pytest.raises(OSError, match=re.escape(os.strerror(code))) as failed:
            spec.create()

    assert failed.value.errno == code
    assert not spec.done
    kept = working_folder(spec)
    assert kept.name.startswith("free__")
    assert sorted(path.name for path in kept.iterdir()) == [
        "events.jsonl",
        "log",
        "result",
    ]
    assert ends_of(spec) == [{"outcome": "ok"}]

    assert spec.create() == 5
    assert list(attempts_dir_in(spec._class_dir).iterdir()) == []


def test_a_rename_that_happened_but_reported_an_error_has_published_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=6)
    rename = os.rename

    def answer_lost(self: Path, target: Path) -> None:
        rename(self, target)
        if target == fixed_run(spec):
            raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(os, "rename", answer_lost)

    assert spec.create() == 6

    assert run_of(spec) == fixed_run(spec)
    assert list(attempts_dir_in(spec._class_dir).iterdir()) == []


def test_scratch_is_moved_out_whole_before_the_run_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = LeavesScratch()
    seen: dict[str, object] = {}
    rename, rmtree = os.rename, shutil.rmtree

    def spy(self: Path, target: Path) -> None:
        if target == fixed_run(spec):
            seen["files"] = sorted(path.name for path in self.iterdir())
            # Beside the folders that hold working folders, not among them.
            assert list(self.parent.iterdir()) == [self]
            (trash,) = (
                path for path in self.parent.parent.iterdir() if path != self.parent
            )
            seen["trash"] = trash.name.partition("__")[0]
            seen["moved"] = (trash / "download.csv").read_text()
        rename(self, target)

    def no_scratch(path: Path, ignore_errors: bool = False) -> None:
        # Deleting it where it is could be cut short and leave half of it
        # for the next attempt.
        assert path.name != "scratch"
        rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(os, "rename", spy)
    monkeypatch.setattr(shutil, "rmtree", no_scratch)

    assert spec.create() == 1

    assert seen == {
        "files": ["events.jsonl", "log", "result"],
        "trash": ".trash",
        "moved": "big",
    }
    assert not (run_of(spec) / "scratch").exists()
    assert list(attempts_dir_in(spec._class_dir).iterdir()) == []


def test_a_batch_is_one_run_with_a_row_per_spec_and_one_file_per_array() -> None:
    specs = [Weights(seed=seed) for seed in range(5)]
    name = run_name([spec._artifact_hash for spec in specs])

    loaded = furu.create(specs)

    assert [row.tolist() for row in loaded] == [[seed] * 3 for seed in range(5)]
    run = runs_dir_in(specs[0]._class_dir / FIXED) / name
    assert {run_of(spec) for spec in specs} == {run}
    assert sorted(
        str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()
    ) == [
        "events.jsonl",
        "log",
        "result/artifacts/root/data.npy",
        "result/manifest.json",
    ]
    assert events_of(specs[0])[0]["specs"] == [spec._artifact_hash for spec in specs]
    # One working folder, named for the process that held it and gone once
    # the run is published.
    (held,) = Weights.folders
    assert held.rpartition("__")[0] == attempt_lock.me()
    assert list(attempts_dir_in(specs[0]._class_dir).iterdir()) == []


def test_a_published_run_is_never_replaced_and_another_order_is_another_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [Train(seed=1), Train(seed=2)]
    furu.create(specs)
    run = run_of(specs[0])
    published = events_path_in(run).read_text()

    # As the worker that lost a race to publish the run: what it computed is
    # dropped, and it has the run it wanted.
    _compute_again(specs, monkeypatch)
    assert list(attempts_dir_in(specs[0]._class_dir).iterdir()) == []
    _compute_again(specs[::-1], monkeypatch)

    assert events_path_in(run).read_text() == published
    hashes = [spec._artifact_hash for spec in specs]
    assert sorted(path.name for path in run.parent.iterdir()) == sorted(
        [run_name(hashes), run_name(hashes[::-1])]
    )


def test_a_spec_in_two_runs_is_read_from_the_one_published_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b, c = (Train(seed=seed) for seed in range(3))
    furu.create([a, b])
    first = run_of(b)
    _compute_again([b, c], monkeypatch)
    second = first.with_name(run_name([b._artifact_hash, c._artifact_hash]))
    # As if the second run had ended first, though this process read it last.
    events = events_path_in(second)
    events.write_text(re.sub(r"\d{4}-", "2000-", events.read_text()))

    assert [run_of(spec) for spec in (a, b, c)] == [first, second, second]
    assert furu.load_existing([c, b, a]) == [0.2, 0.1, 0.0]

    # Deleting one of the runs leaves the spec done.
    shutil.rmtree(second)
    assert run_of(b) == first
    assert not c.done


def test_a_batch_with_a_stored_spec_still_runs_whole_in_the_folder_named_for_it() -> (
    None
):
    specs = [FailsAsABatch(n=1), FailsAsABatch(n=2)]
    furu.create(specs[0])

    with pytest.raises(ValueError, match="boom"), creating(plan=_launch_plan()):
        load_or_create._ensure_group_result(specs)

    # Where the job that handed the batch out looks for its log.
    assert "boom" in log_path_in(working_folder(specs)).read_text()


def test_a_run_is_read_once_and_a_new_listing_reads_what_was_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read: list[str] = []
    events = stored._events

    def spy(runs: Sequence[Path]) -> list[bytes]:
        read.extend(run.name for run in runs)
        return events(runs)

    monkeypatch.setattr(stored, "_events", spy)
    old, new = Train(seed=1), Train(seed=2)

    furu.create([old, Train(seed=3)])
    assert read == [run_of(old).name]
    assert old.done
    assert not new.done

    furu.create(new)
    assert new.done
    assert old.done
    assert read == [run_of(old).name, new._artifact_hash]


def test_adding_the_decorator_finds_one_row_runs_and_removing_it_stops_the_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert furu.create([Merge(n=1), Merge(n=2)]) == [1, 2]

    # Nothing moves: a run with one row is found by the same listing.
    batch = [BatchedMerge(n=3), BatchedMerge(n=4)]
    assert furu.create([BatchedMerge(n=1), *batch]) == [1, 3, 4]
    assert run_of(BatchedMerge(n=1)) == fixed_run(Merge(n=1))
    assert events_of(batch[0])[0]["specs"] == [spec._artifact_hash for spec in batch]

    assert Merge(n=1).done
    assert not Merge(n=3).done
    held = run_of(batch[0])
    read: list[str] = []
    events = stored._events

    def spy(runs: Sequence[Path]) -> list[bytes]:
        read.extend(run.name for run in runs)
        return events(runs)

    monkeypatch.setattr(stored, "_events", spy)
    monkeypatch.setattr(stored, "_READ", {})  # as a new process
    with pytest.raises(furu.Missing, match=re.escape(str(held))):
        Merge(n=3).load_existing()

    with pytest.raises(RuntimeError, match="Put the decorator back") as error:
        furu.create([Merge(n=1), Merge(n=3)])
    assert f"Merge: {held}" in str(error.value)
    # Only the batch folder is read, once per process; never the one-row runs.
    assert read == [held.name]


def test_delete_raises_on_a_batched_spec_and_names_its_run() -> None:
    specs = [Train(seed=1), Train(seed=2)]
    furu.create(specs)

    with pytest.raises(RuntimeError, match=re.escape(str(run_of(specs[0])))):
        specs[0].delete(mode="force")

    assert all(spec.done for spec in specs)


def test_a_worker_whose_folder_was_taken_publishes_nothing_and_makes_no_folder() -> (
    None
):
    spec = LosesItsFolder()

    with pytest.raises(FileNotFoundError):
        spec.create()

    assert not spec.done
    # The folder is as it was taken: nothing was made again under the old
    # name, and the events are left to whoever has it now.
    assert attempt_lock.entries(attempt_parent([spec])) == ["free__manual"]
    assert ends_of(spec) == []
    assert not (working_folder(spec) / "result").exists()


def test_events_list_the_specs_once_and_each_attempts_start_and_end() -> None:
    spec = BlocksThenFailsOnce()

    with pytest.raises(ValueError, match="bad shape"):
        spec.create()  # blocks on its leaf, then fails
    assert spec.create() == 7

    lines = events_path_in(run_of(spec)).read_text().split("\n")
    # Each attempt begins after a line break.
    assert [bool(line) for line in lines] == [
        *(False, True, True, True),
        *(False, True, True),
        *(False, True, True),
        False,
    ]
    events = events_of(spec)
    times = [event.pop("at") for event in events]
    assert all(
        re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", t) for t in times
    )
    # Attempts in one second still sort by when they happened.
    assert times[1:] == sorted(set(times[1:]))
    plans = [Path(event.pop("plan")) for event in events if event["event"] == "start"]
    records = {event.pop("provenance") for event in events if "provenance" in event}
    host = socket.gethostname()
    assert events == [
        {"event": "opened", "specs": [spec._artifact_hash]},
        {"event": "start", "attempt": 1, "host": host},
        {"event": "end", "outcome": "blocked", "waiting_on": [Leaf(n=7).object_id]},
        {"event": "start", "attempt": 2, "host": host},
        {"event": "end", "outcome": "failed", "error": "ValueError: bad shape"},
        {"event": "start", "attempt": 3, "host": host},
        {"event": "end", "outcome": "ok"},
    ]
    # Each start names the record of the process that ran the attempt, and
    # the plan of its launch, which lists the spec.
    (record,) = records
    assert Path(record).parent == get_config().directories.anchored().provenance
    assert plans[0] == plans[1] != plans[2]
    listed = {"fields": {}, "runs": True, "spec": spec.object_id}
    assert plan.read(plans[0])[1:] == plan.read(plans[2])[1:] == [listed]
    # The leaf its create() asked for is listed in a file of the process that
    # ran the create(), in the folder of the launch.
    leaf = Leaf(n=7)
    found = Path(events_of(leaf)[1]["plan"])
    assert found.parent == plans[0].with_suffix("")


def test_an_attempt_killed_in_the_middle_of_a_line_costs_only_that_line() -> None:
    spec = Leaf(n=4)
    attempt = attempt_parent([spec]) / "free__9f3c2a7d51b0e6a4"
    attempt.mkdir(parents=True)
    events_path_in(attempt).write_text(
        f'\n{{"event":"opened","at":"2026-11-14T14:02:09Z","specs":["{spec._artifact_hash}"]}}\n'
        '{"event":"start","at":"2026-11-14T14:02:11Z","attempt":1,"host":"jagupard26"}\n'
        '{"event":"end","at":"2026-11-14T15:4'
    )

    assert spec.create() == 4

    assert [
        (event["event"], event.get("attempt"), event.get("outcome"))
        for event in events_of(spec)
    ] == [
        ("opened", None, None),
        ("start", 1, None),
        ("start", 2, None),
        ("end", None, "ok"),
    ]
