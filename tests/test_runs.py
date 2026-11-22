"""A run does its work in attempts/ and is published into runs/ by one rename.

A batch is one run with a row per spec, found again by listing runs/.
"""

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
from furu.dependencies import creating
from furu.execution import load_or_create
from furu.storage import stored
from furu.storage._layout import (
    FIXED,
    attempts_dir_in,
    events_path_in,
    log_path_in,
    run_name,
    runs_dir_in,
)
from furu.storage.attempt import owner


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
    locks: ClassVar[list[str]] = []

    @furu.batched(lambda spec, worker: (None, 100))
    def create(specs: list["Weights"], directory: furu.SpecDirectory) -> list[Any]:
        attempts = attempts_dir_in(specs[0]._class_dir)
        Weights.locks += [
            f"{run.name}/{folder.name.rpartition('__')[0]}"
            for run in attempts.iterdir()
            for folder in run.iterdir()
        ]
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
        # As if someone released it by hand, and another process claimed it.
        folder = working_folder(self)
        folder.rename(folder.with_name("free__taken"))
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


def _publishes(spec: Spec, folder: Path) -> bool:
    return folder.parent == spec._attempt_dir and not folder.name.startswith(".")


def test_a_run_is_whole_and_still_held_when_one_rename_publishes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=1)
    seen: dict[str, object] = {}
    rename = Path.rename

    def spy(self: Path, target: Path) -> Path:
        if _publishes(spec, self):
            seen["files"] = sorted(path.name for path in self.iterdir())
            seen["ends"] = ends_of(spec)
            seen["held"] = self.name.startswith(owner() + "__")
            seen["done"] = spec.done
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", spy)

    assert spec.create() == 1

    files = ["events.jsonl", "log", "result"]
    assert seen == {
        "files": files,
        "ends": [{"outcome": "ok"}],
        "held": True,
        "done": False,
    }
    assert sorted(path.name for path in run_of(spec).iterdir()) == files
    # The working folder moved, and the folder that held it went too.
    assert list(attempts_dir_in(spec._class_dir).iterdir()) == []


def test_a_crash_before_the_rename_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=2)

    rename = Path.rename

    def crash(self: Path, target: Path) -> Path:
        if _publishes(spec, self):
            raise RuntimeError("killed before the rename")
        return rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", crash)
        with pytest.raises(RuntimeError, match="killed before the rename"):
            spec.create()

    assert not spec.done
    assert not fixed_run(spec).exists()
    assert spec._attempt_dir.is_dir()

    assert spec.create() == 2


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
    # One working folder, named for this process, gone once it is published.
    assert Weights.locks == [f"{name}/{owner()}"]
    assert list(attempts_dir_in(specs[0]._class_dir).iterdir()) == []


def test_the_same_specs_in_the_same_order_are_dropped_and_another_order_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [Train(seed=1), Train(seed=2)]
    furu.create(specs)
    run = run_of(specs[0])
    published = events_path_in(run).read_text()

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

    # A run that is being deleted has a dotted name, and is not looked in.
    first.rename(first.with_name(f"{first.name}.host.1.deleting"))
    assert not a.done


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


def test_a_holder_that_lost_its_lock_leaves_the_events_to_its_successor() -> None:
    spec = LosesItsFolder()

    with pytest.raises(RuntimeError, match="lost .* before writing the result"):
        spec.create()

    assert ends_of(spec) == []


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
    killed = spec._attempt_dir / "free__killed"
    killed.mkdir(parents=True)
    events_path_in(killed).write_text(
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
