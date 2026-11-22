"""attempts/<run>/ holds one working folder, whose name says who may take it."""

import errno
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest
from runs import fixed_run

from furu import Spec, slurm
from furu.locking import short_lock
from furu.migration.schema import _recorded_defaults
from furu.storage import attempt_lock
from furu.storage._layout import result_dir_in, scratch_dir_in
from furu.storage.attempt_lock import (
    HeldError,
    acquire,
    entries,
    me,
    owner,
    publish,
    release,
)

ctx = get_context("fork")


@pytest.fixture(autouse=True)
def _outside_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name folders as a process outside Slurm does, wherever the tests run."""
    for name in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_RESTART_COUNT"):
        monkeypatch.delenv(name, raising=False)


def _run(tmp_path: Path, *folders: str) -> Path:
    run = tmp_path / "attempts" / "03be"
    run.mkdir(parents=True)
    for folder in folders:
        (run / folder).mkdir()
    return run


# --- naming ------------------------------------------------------------------------


def test_a_process_names_its_folders_for_its_slurm_job_or_its_host_and_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert me() == f"local__{socket.gethostname()}__{os.getpid()}"

    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sphinx")
    assert me() == "slurm__sphinx__42__0"
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    assert me() == "slurm__sphinx__42__3"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("free__seed", "adoptable"),
        ("free__0f3a", "adoptable"),
        ("local__host__12__0f3a", "held"),
        # Malformed: never taken.
        ("free", "unknown"),
        ("free__", "unknown"),
        ("free__a__b", "unknown"),
        ("local__host__12", "unknown"),
        ("local____12__0f", "unknown"),
        ("slurm__c__42__0", "unknown"),
        ("whatever", "unknown"),
        ("03be", "unknown"),
    ],
)
def test_a_folder_name_says_whether_it_may_be_taken(name: str, expected: str) -> None:
    assert owner(name) == expected


# --- acquire and release -------------------------------------------------------------


def test_a_run_with_no_folder_gets_one_and_this_process_takes_it(
    tmp_path: Path,
) -> None:
    run = tmp_path / "attempts" / "03be"
    run.parent.mkdir()

    attempt = acquire(run)

    assert attempt is not None
    assert entries(run) == [attempt.name]
    held, _, token = attempt.name.rpartition("__")
    assert held == me()
    assert len(token) == 16


def test_a_released_folder_is_taken_with_what_it_holds(tmp_path: Path) -> None:
    run = _run(tmp_path)
    first = acquire(run)
    assert first is not None
    (first / "checkpoint").write_text("step 7")

    release(first)
    (freed,) = entries(run)
    assert freed.startswith("free__") and freed != "free__seed"
    again = acquire(run)

    assert again is not None and again != first
    assert (again / "checkpoint").read_text() == "step 7"
    assert entries(run) == [again.name]


def test_each_release_names_the_folder_anew(tmp_path: Path) -> None:
    run = _run(tmp_path)
    freed = set()
    for _ in range(3):
        attempt = acquire(run)
        assert attempt is not None
        release(attempt)
        freed.update(entries(run))
    assert len(freed) == 3


def test_a_release_after_publishing_does_nothing(tmp_path: Path) -> None:
    run = _run(tmp_path)
    attempt = acquire(run)
    assert attempt is not None
    published = tmp_path / "fixed" / "runs" / run.name
    publish(attempt, published)

    release(attempt)

    assert published.is_dir()
    assert not run.exists()


def test_a_published_run_is_not_taken_up_again(tmp_path: Path) -> None:
    run = tmp_path / "attempts" / "03be"
    published = tmp_path / "fixed" / "runs" / "03be"
    published.mkdir(parents=True)

    assert acquire(run, [published]) is None
    assert not run.exists()


def test_a_folder_a_local_process_holds_is_refused_with_how_to_free_it(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "local__node7__4242__0f3a")

    with pytest.raises(HeldError) as raised:
        acquire(run)

    message = str(raised.value)
    assert str(run / "local__node7__4242__0f3a") in message
    assert "process 4242 on node7" in message
    assert str(run / "free__manual") in message
    assert entries(run) == ["local__node7__4242__0f3a"]


@pytest.mark.parametrize("name", ["junk", "free__a__b", "slurm__other__42__0__0f"])
def test_a_folder_whose_holder_is_unknown_is_refused(
    tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sphinx")
    run = _run(tmp_path, name)

    with pytest.raises(HeldError, match="may be held"):
        acquire(run)
    assert entries(run) == [name]


def test_a_free_folder_beside_a_held_one_is_not_taken(tmp_path: Path) -> None:
    run = _run(tmp_path, "free__0f", "local__node7__1__0f")

    with pytest.raises(HeldError):
        acquire(run)
    assert entries(run) == ["free__0f", "local__node7__1__0f"]


def test_a_folder_with_checkpoints_is_taken_before_one_a_race_made(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "free__seed", "free__0f")
    (run / "free__0f" / "checkpoint").touch()

    attempt = acquire(run)

    assert attempt is not None
    assert (attempt / "checkpoint").exists()
    assert entries(run) == [attempt.name]


def test_a_worker_that_loses_the_rename_sees_the_winner_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path, "free__0f")
    rename = Path.rename

    def another_worker_first(self: Path, target: Path) -> Path:
        rename(self, self.with_name("local__node7__1__1a"))
        raise FileNotFoundError(errno.ENOENT, "taken", str(self))

    monkeypatch.setattr(Path, "rename", another_worker_first)

    with pytest.raises(HeldError, match="local__node7__1__1a"):
        acquire(run)


def test_a_rename_that_went_through_but_reported_an_error_is_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path, "free__0f")
    rename = Path.rename

    def retried(self: Path, target: Path) -> Path:
        rename(self, target)
        raise OSError(errno.ESTALE, "retransmitted", str(self))

    monkeypatch.setattr(Path, "rename", retried)

    attempt = acquire(run)
    assert attempt is not None
    assert entries(run) == [attempt.name]


def _take(run: str, start: Any, out: Any) -> None:
    start.wait()
    try:
        out.put(("took", str(acquire(Path(run)))))
    except HeldError:
        out.put(("held", ""))


def test_of_processes_racing_for_one_run_exactly_one_takes_it(tmp_path: Path) -> None:
    run = tmp_path / "attempts" / "03be"
    run.parent.mkdir()
    start, out = ctx.Event(), ctx.Queue()
    racers = [
        ctx.Process(target=_take, args=(str(run), start, out), daemon=True)
        for _ in range(6)
    ]
    for racer in racers:
        racer.start()
    start.set()
    results = [out.get(timeout=10) for _ in racers]
    for racer in racers:
        racer.join(timeout=10)

    took = [path for outcome, path in results if outcome == "took"]
    assert len(took) == 1
    assert entries(run) == [Path(took[0]).name]


# --- the Slurm owner check -----------------------------------------------------------


def _fake_slurm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    squeue: str | Exception = "",
    sacct: str | Exception = "",
) -> list[tuple[str, ...]]:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sphinx")
    calls: list[tuple[str, ...]] = []

    def run(*argv: str) -> str:
        calls.append(argv)
        answer = {"squeue": squeue, "sacct": sacct}[argv[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(slurm, "run", run)
    return calls


def _failed(stderr: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(1, "squeue", stderr=stderr)


GONE = "slurm__sphinx__42__0__0f"


def test_a_slurm_job_that_ended_cleanly_gives_its_folder_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_slurm(monkeypatch, sacct="42|FAILED|NonZeroExitCode\n")

    assert owner(GONE) == "adoptable"
    assert calls == [
        ("squeue", "--all", "--noheader", "-O", "JobIDRaw:32,State:32", "-j", "42"),
        (
            "sacct", "-D", "-X", "--noheader", "--parsable2",
            "-o", "JobIDRaw,State,Reason", "-j", "42",
        ),
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("squeue", "sacct", "expected"),
    [
        # Still queued, in any state.
        ("42  RUNNING\n", "42|RUNNING|None\n", "held"),
        ("42  PENDING\n", "", "held"),
        ("42  COMPLETING\n", "42|COMPLETED|None\n", "held"),
        # Purged from squeue, which older Slurm reports as an error.
        (_failed("slurm_load_jobs error: Invalid job id specified"), "42|TIMEOUT|None\n", "adoptable"),
        ("", "42|CANCELLED by 1234|None\n", "adoptable"),
        # Requeued once, then done.
        ("", "42|PREEMPTED|None\n42|COMPLETED|None\n", "adoptable"),
        # Its processes may still run.
        ("", "42|NODE_FAIL|None\n", "unknown"),
        ("", "42|BOOT_FAIL|None\n", "unknown"),
        ("", "42|CANCELLED|Kill task failed\n", "unknown"),
        ("", "42|COMPLETED|None\n42|NODE_FAIL|None\n", "unknown"),
        # Accounting has not caught up, or says something else.
        ("", "", "unknown"),
        ("", "43|COMPLETED|None\n", "unknown"),
        ("", "42|COMPLETED\n", "unknown"),
        ("", "42|SUSPENDED|None\n", "unknown"),
        # Slurm does not answer.
        (_failed("slurm_load_jobs error: Socket timed out"), "", "unknown"),
        (subprocess.TimeoutExpired("squeue", 60), "", "unknown"),
        (FileNotFoundError("squeue"), "", "unknown"),
        ("", subprocess.TimeoutExpired("sacct", 60), "unknown"),
        ("", _failed("sacct: error"), "unknown"),
    ],
)  # fmt: skip
def test_a_slurm_job_gives_its_folder_up_only_when_surely_gone(
    monkeypatch: pytest.MonkeyPatch,
    squeue: str | Exception,
    sacct: str | Exception,
    expected: str,
) -> None:
    _fake_slurm(monkeypatch, squeue=squeue, sacct=sacct)

    assert owner(GONE) == expected


def test_a_slurm_job_of_another_cluster_is_never_looked_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_slurm(monkeypatch, sacct="42|COMPLETED|None\n")

    assert owner("slurm__other__42__0__0f") == "unknown"
    assert calls == []


def test_a_folder_of_a_slurm_job_that_ended_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_slurm(monkeypatch, sacct="42|TIMEOUT|None\n")
    run = _run(tmp_path, GONE)
    (run / GONE / "checkpoint").touch()

    attempt = acquire(run)

    assert attempt is not None
    assert (attempt / "checkpoint").exists()


def test_a_folder_of_a_queued_slurm_job_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_slurm(monkeypatch, squeue="42  RUNNING\n")
    run = _run(tmp_path, GONE)

    with pytest.raises(HeldError, match="still queued"):
        acquire(run)


# --- publish -------------------------------------------------------------------------


def _finished(tmp_path: Path) -> tuple[Path, Path]:
    """A working folder with a result and scratch, and where it is published."""
    run = _run(tmp_path)
    attempt = acquire(run)
    assert attempt is not None
    result_dir_in(attempt).mkdir()
    (result_dir_in(attempt) / "manifest.json").write_text("[]")
    scratch_dir_in(attempt).mkdir()
    (scratch_dir_in(attempt) / "download.csv").touch()
    return attempt, tmp_path / "fixed" / "runs" / run.name


def test_publishing_moves_scratch_aside_and_leaves_attempts_empty(
    tmp_path: Path,
) -> None:
    attempt, published = _finished(tmp_path)

    publish(attempt, published)

    assert sorted(path.name for path in published.iterdir()) == ["result"]
    assert not attempt.parent.exists()


def test_a_publish_that_may_have_failed_keeps_the_only_finished_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, published = _finished(tmp_path)
    rename = Path.rename

    def stale(self: Path, target: Path) -> Path:
        if target == published:
            raise OSError(errno.ESTALE, "stale file handle", str(self))
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", stale)

    with pytest.raises(OSError, match="stale file handle"):
        publish(attempt, published)

    assert (result_dir_in(attempt) / "manifest.json").exists()
    assert not published.exists()


def test_a_publish_reported_failed_that_went_through_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, published = _finished(tmp_path)
    rename = Path.rename

    def retried(self: Path, target: Path) -> Path:
        rename(self, target)
        if target == published:
            raise FileNotFoundError(errno.ENOENT, "retransmitted", str(self))
        return target

    monkeypatch.setattr(Path, "rename", retried)

    publish(attempt, published)

    assert (result_dir_in(published) / "manifest.json").exists()
    assert not attempt.parent.exists()


def test_a_run_published_first_by_another_worker_is_kept_and_this_one_dropped(
    tmp_path: Path,
) -> None:
    attempt, published = _finished(tmp_path)
    published.mkdir(parents=True)
    (published / "winner").touch()

    publish(attempt, published)

    assert sorted(path.name for path in published.iterdir()) == ["winner"]
    assert not attempt.parent.exists()


# --- short_lock and schema.ndjson ----------------------------------------------------


def test_a_short_lock_waits_for_its_holder(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    order: list[str] = []

    def second() -> None:
        with short_lock(path):
            order.append("second")

    with short_lock(path):
        waiter = threading.Thread(target=second)
        waiter.start()
        time.sleep(0.2)
        order.append("first")
    waiter.join(timeout=5)

    assert order == ["first", "second"]
    assert not path.exists()


def test_a_short_lock_left_by_a_killed_holder_is_broken(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    path.touch()
    old = time.time() - 120
    os.utime(path, (old, old))

    with short_lock(path):
        assert path.exists()
    assert not path.exists()


def test_the_first_line_recorded_for_a_step_wins() -> None:
    lines = [
        {"names": ["lemma"]},
        {"added": "min_overlap", "default": 0.5},
        {"added": "min_overlap", "default": 0.6},
    ]

    assert _recorded_defaults(lines) == {(None, None, "min_overlap"): 0.5}


# --- Spec.delete ---------------------------------------------------------------------


class Deleted(Spec):
    n: int

    def create(self) -> int:
        return self.n


def _with_folder(name: str) -> Deleted:
    spec = Deleted(n=len(name))
    spec.create()
    (spec._attempt_dir / name).mkdir(parents=True)
    return spec


def test_delete_refuses_while_a_worker_holds_the_working_folder() -> None:
    spec = _with_folder("local__node7__4242__0f")

    with pytest.raises(HeldError, match="process 4242 on node7"):
        spec.delete(mode="force")

    assert spec.done
    assert entries(spec._attempt_dir) == ["local__node7__4242__0f"]


def test_delete_removes_a_free_working_folder_and_the_runs() -> None:
    spec = _with_folder("free__0f")

    assert spec.delete(mode="force")

    assert not spec.done
    assert not spec._attempt_dir.exists()
    assert not fixed_run(spec).exists()
    assert os.listdir(fixed_run(spec).parent) == []


def test_delete_fails_when_a_worker_takes_the_folder_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _with_folder("free__0f")
    holder: Callable[..., str | None] = attempt_lock.holder

    def taken_after_the_look(run: Path, found: list[str]) -> str | None:
        why = holder(run, found)
        (run / "free__0f").rename(run / "local__node7__1__1a")
        return why

    monkeypatch.setattr("furu.core.holder", taken_after_the_look)

    with pytest.raises(FileNotFoundError):
        spec.delete(mode="force")
    assert entries(spec._attempt_dir) == ["local__node7__1__1a"]
