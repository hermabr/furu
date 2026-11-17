"""A run does its work in attempts/ and is published into runs/ by one rename."""

import re
import socket
from typing import Any

import pytest
from runs import ends_of, events_of, fixed_run, run_of

from furu import Spec
from furu.code_trace import Trace
from furu.execution import load_or_create
from furu.locking import is_active_lock
from furu.storage._layout import (
    attempts_dir_in,
    events_path_in,
    lock_path_in,
    run_name,
)


class Leaf(Spec[int]):
    n: int

    def create(self) -> int:
        return self.n


class LosesItsLock(Spec[int]):
    def create(self) -> int:
        # As if paused past the stale window, so that a successor took over.
        lock_path_in(self._class_dir, self._run_name).unlink()
        return 1


class BlocksThenFailsOnce(Spec[int]):
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
    assert Leaf(n=1)._run_name == Leaf(n=1)._artifact_hash


def test_a_run_is_whole_and_still_locked_when_one_rename_publishes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=1)
    seen: dict[str, object] = {}
    publish = load_or_create._publish

    def spy(obj: Spec, trace: Trace, **kwargs: Any) -> None:
        seen["files"] = sorted(path.name for path in obj._attempt_dir.iterdir())
        seen["ends"] = ends_of(obj)
        seen["locked"] = is_active_lock(lock_path_in(obj._class_dir, obj._run_name))
        seen["done"] = obj.done
        publish(obj, trace, **kwargs)

    monkeypatch.setattr(load_or_create, "_publish", spy)

    assert spec.create() == 1

    files = ["events.jsonl", "log", "provenance.json", "result"]
    assert seen == {
        "files": files,
        "ends": [{"outcome": "ok"}],
        "locked": True,
        "done": False,
    }
    assert sorted(path.name for path in run_of(spec).iterdir()) == files
    # The working folder moved, and its lock beside it was released.
    assert list(attempts_dir_in(spec._class_dir).iterdir()) == []


def test_a_crash_before_the_rename_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=2)

    def crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("killed before the rename")

    with monkeypatch.context() as patched:
        patched.setattr(load_or_create, "_publish", crash)
        with pytest.raises(RuntimeError, match="killed before the rename"):
            spec.create()

    assert not spec.done
    assert not fixed_run(spec).exists()
    assert spec._attempt_dir.is_dir()

    assert spec.create() == 2


def test_a_second_publish_under_the_same_name_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Leaf(n=3)
    spec.create()
    published = events_path_in(run_of(spec)).read_text()

    # A worker that does not see the run computes it again.
    monkeypatch.setattr(load_or_create, "stored_run", lambda obj: None)
    with pytest.raises(OSError, match="Directory not empty|File exists"):
        load_or_create._ensure_group_result(
            [spec], submit_provenance=load_or_create.capture_submit_provenance()
        )

    assert events_path_in(fixed_run(spec)).read_text() == published
    assert spec._attempt_dir.is_dir()


def test_a_holder_that_lost_its_lock_leaves_the_events_to_its_successor() -> None:
    spec = LosesItsLock()

    with pytest.raises(RuntimeError, match="lost the lock"):
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
    assert all(
        re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", event.pop("at"))
        for event in events
    )
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


def test_an_attempt_killed_in_the_middle_of_a_line_costs_only_that_line() -> None:
    spec = Leaf(n=4)
    spec._attempt_dir.mkdir(parents=True)
    events_path_in(spec._attempt_dir).write_text(
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
