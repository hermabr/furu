"""The schema's short lock, and the folder names that say who works on a run."""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from furu.locking import short_lock
from furu.storage import attempt
from furu.storage.attempt import (
    claim,
    claimed,
    free_after_death,
    owner,
    release,
    why_held,
)


@pytest.fixture(autouse=True)
def _outside_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def test_a_short_lock_is_held_by_one_holder_at_a_time(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    with short_lock(path):
        assert path.exists()
        with (
            pytest.raises(RuntimeError, match=f"{path} was held for over"),
            short_lock(path, timeout_s=0.05),
        ):
            pass
    assert not path.exists()
    with short_lock(path):
        pass


def test_a_claim_seeds_an_empty_run_and_names_the_folder_for_its_owner(
    tmp_path: Path,
) -> None:
    run = tmp_path / "attempts" / "03be"

    with claimed(run) as folder:
        assert os.listdir(run) == [folder.name]
        assert folder.name.startswith(f"{owner()}__")
        (folder / "log").write_text("half done")

    (free,) = os.listdir(run)
    assert free.startswith("free__") and free != attempt.SEED
    # The next claim takes it over, with what is in it.
    again = claim(run)
    assert (again / "log").read_text() == "half done"
    assert len({free, again.name, folder.name}) == 3


def test_releasing_an_empty_folder_removes_it_and_its_run(tmp_path: Path) -> None:
    run = tmp_path / "03be"
    release(claim(run))
    assert not run.exists()


@pytest.mark.parametrize(
    "names",
    [
        ["local__elsewhere__123__aa"],
        ["mystery"],
        ["slurm__sc__1"],
        ["free__a", "free__b"],
    ],
)
def test_a_claim_refuses_a_held_or_unknown_folder_without_adding_one(
    tmp_path: Path, names: list[str]
) -> None:
    for name in names:
        (tmp_path / name).mkdir()

    with pytest.raises(RuntimeError, match="free__released"):
        claim(tmp_path)

    assert sorted(os.listdir(tmp_path)) == names


def test_a_folder_stranded_by_a_dead_local_process_is_refused(tmp_path: Path) -> None:
    child = subprocess.Popen(["true"])
    child.wait()
    (tmp_path / f"{owner(child.pid)}__aa").mkdir()

    with pytest.raises(RuntimeError, match=f"process {child.pid} on"):
        claim(tmp_path)


def test_a_claim_waits_for_a_live_process_on_this_machine(tmp_path: Path) -> None:
    held = claim(tmp_path)
    timer = threading.Timer(0.2, release, [held])
    (held / "log").touch()
    timer.start()
    started = time.monotonic()

    with claimed(tmp_path) as folder:
        assert time.monotonic() - started >= 0.2
        assert (folder / "log").exists()


def test_a_parent_frees_only_the_folder_of_its_dead_child(tmp_path: Path) -> None:
    (tmp_path / "a" / f"{owner(111)}__aa").mkdir(parents=True)
    (tmp_path / "b" / f"{owner(222)}__bb").mkdir(parents=True)

    free_after_death(tmp_path / "a", owner(111))
    free_after_death(tmp_path / "b", owner(111))

    assert os.listdir(tmp_path / "a")[0].startswith("free__")
    assert os.listdir(tmp_path / "b") == [f"{owner(222)}__bb"]


def _slurm(
    monkeypatch: pytest.MonkeyPatch, squeue: tuple[int, str, str], sacct: str
) -> None:
    def run(argv: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        code, out, err = squeue if argv[0] == "squeue" else (0, sacct, "")
        return subprocess.CompletedProcess(argv, code, out, err)

    monkeypatch.setattr(subprocess, "run", run)


ENDED = (0, "", "")
PURGED = (1, "", "slurm_load_jobs error: Invalid job id specified")


@pytest.mark.parametrize(
    ("squeue", "sacct", "free"),
    [
        (ENDED, "sc|COMPLETED|None\n", True),
        (PURGED, "sc|CANCELLED by 24928|None\n", True),
        (ENDED, "sc|PREEMPTED|None\nsc|TIMEOUT|None\n", True),
        ((0, "42\n", ""), "sc|RUNNING|None\n", False),
        ((1, "", "slurm_load_jobs error: Socket timed out"), "", False),
        (ENDED, "", False),
        (ENDED, "other|COMPLETED|None\n", False),
        (ENDED, "sc|NODE_FAIL|None\n", False),
        (ENDED, "sc|BOOT_FAIL|None\n", False),
        (ENDED, "sc|FAILED|Kill task failed\n", False),
        (ENDED, "sc|REQUEUED|None\nsc|NODE_FAIL|None\n", False),
    ],
)
def test_a_slurm_jobs_folder_is_free_once_every_incarnation_ended_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    squeue: tuple[int, str, str],
    sacct: str,
    free: bool,
) -> None:
    _slurm(monkeypatch, squeue, sacct)
    assert (why_held("slurm__sc__42__0__aa") is None) == free


def test_without_slurm_a_slurm_jobs_folder_stays_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_: object, **__: object) -> None:
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(subprocess, "run", missing)
    assert why_held("slurm__sc__42__0__aa") is not None


def test_a_claim_waits_for_a_folder_of_its_own_slurm_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    test_a_claim_waits_for_a_live_process_on_this_machine(tmp_path)
