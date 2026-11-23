"""attempts/<run>/<name>/: the working folder whose name is the lock."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from furu import slurm
from furu.locking import STALE_S, short_lock
from furu.result.bundle import _save_result_bundle
from furu.storage import attempt_lock
from furu.storage.attempt_lock import HeldError, acquire, entries, publish
from furu.storage.directory import SpecDirectory


def _never() -> bool:
    return False


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return tmp_path / "attempts" / "r"


def _attempts(run: Path) -> list[str]:
    return sorted(os.listdir(run.parent))


# --- taking the folder ---


def test_a_run_with_no_folder_gets_one_named_for_this_process(run: Path) -> None:
    claim = acquire(run, _never, token="t")

    assert claim is not None
    assert entries(run) == [f"local__{os.uname().nodename}__{os.getpid()}__t"]
    assert claim.path == run / entries(run)[0]
    assert _attempts(run) == ["r"]  # the seed is gone


def test_a_seed_lands_on_a_run_folder_left_empty(run: Path) -> None:
    run.mkdir(parents=True)

    assert acquire(run, _never) is not None
    assert len(entries(run)) == 1


def test_a_worker_whose_seed_lands_second_is_told_who_took_the_run(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = attempt_lock._take

    def other_seeds_first(run: Path, dest: Path) -> bool:
        assert not take(run, dest)
        (run / "local__other__1__x").mkdir(parents=True)
        return False

    monkeypatch.setattr(attempt_lock, "_take", other_seeds_first)

    with pytest.raises(HeldError, match="took the working folder"):
        acquire(run, _never)
    assert entries(run) == ["local__other__1__x"]
    assert _attempts(run) == ["r"]


def test_a_free_folder_is_taken_with_what_it_holds(run: Path) -> None:
    (run / "free__old").mkdir(parents=True)
    (run / "free__old" / "partial").write_text("x")

    claim = acquire(run, _never)

    assert claim is not None
    assert (claim.path / "partial").read_text() == "x"
    assert entries(run) == [claim.path.name]


def test_a_listing_that_shows_two_folders_for_a_moment_is_taken_again(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "free__b").mkdir(parents=True)
    listings = iter([["free__a", "free__b"], ["free__a", "free__b"]])
    real = attempt_lock.list_names
    monkeypatch.setattr(
        attempt_lock, "list_names", lambda folder: next(listings, None) or real(folder)
    )

    assert acquire(run, _never) is not None


def test_a_run_that_keeps_two_folders_is_an_error_that_names_both(run: Path) -> None:
    (run / "free__a").mkdir(parents=True)
    (run / "free__b").mkdir()

    with pytest.raises(HeldError, match="free__a, free__b"):
        acquire(run, _never)


def test_a_take_whose_rename_reply_was_lost_still_holds_the_folder(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run / "free__old").mkdir(parents=True)
    real = os.rename

    def lost_reply(source: Any, dest: Any) -> None:
        real(source, dest)
        raise FileNotFoundError(source)

    monkeypatch.setattr(os, "rename", lost_reply)

    claim = acquire(run, _never)

    assert claim is not None
    assert entries(run) == [claim.path.name]


@pytest.mark.parametrize("holder", ["local__h__1__t", "slurm__c__5__0__t", "odd"])
def test_a_held_folder_is_an_error_with_a_way_out(run: Path, holder: str) -> None:
    (run / holder).mkdir(parents=True)

    with pytest.raises(HeldError) as excinfo:
        acquire(run, _never)
    assert f"`mv {run / holder} {run / 'free__manual'}`" in str(excinfo.value)
    assert entries(run) == [holder]


def test_a_run_published_before_the_claim_leaves_no_folder(run: Path) -> None:
    assert acquire(run, lambda: True) is None
    assert _attempts(run) == []


def test_a_taken_folder_of_a_run_published_since_is_given_back(run: Path) -> None:
    (run / "free__old").mkdir(parents=True)
    (run / "free__old" / "partial").write_text("x")

    assert acquire(run, lambda: True) is None
    (name,) = entries(run)
    assert name.startswith("free__")
    assert (run / name / "partial").read_text() == "x"


def test_a_dead_child_releases_only_the_folder_it_took(run: Path) -> None:
    (run / "slurm__c__5__0__sibling").mkdir(parents=True)

    attempt_lock.release_child(run, "mine")
    assert entries(run) == ["slurm__c__5__0__sibling"]
    attempt_lock.release_child(run, "sibling")
    assert entries(run)[0].startswith("free__")


def test_a_killed_process_leaves_a_folder_its_worker_frees_for_the_next(
    run: Path,
) -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                from pathlib import Path
                from furu.storage import attempt_lock
                claim = attempt_lock.acquire(Path(sys.argv[1]), lambda: False, token="t")
                (claim.path / "partial").write_text("x")
                print("ready", flush=True)
                sys.stdin.read()
                """
            ),
            str(run),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline() == "ready\n"
    child.kill()
    child.wait()

    # Nothing tells a process outside Slurm has stopped but its worker.
    assert attempt_lock.held(run) is not None
    attempt_lock.release_child(run, "t")
    claim = acquire(run, _never)
    assert claim is not None
    assert (claim.path / "partial").read_text() == "x"


def test_a_process_in_a_job_of_no_known_cluster_names_itself_by_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "5")
    monkeypatch.setattr(slurm, "cluster", lambda: None)
    assert attempt_lock.me().startswith("local__")
    monkeypatch.setattr(slurm, "cluster", lambda: "c")
    assert attempt_lock.me() == "slurm__c__5__0"


# --- publishing ---


def _claimed(run: Path) -> attempt_lock.Claim:
    claim = acquire(run, _never)
    assert claim is not None
    (claim.path / "result").mkdir()
    (claim.path / "scratch").mkdir()
    return claim


def test_publish_moves_the_folder_without_its_scratch(
    run: Path, tmp_path: Path
) -> None:
    claim = _claimed(run)
    dest = tmp_path / "runs" / "r"

    publish(claim, dest)

    assert sorted(os.listdir(dest)) == ["result"]
    assert _attempts(run) == []


def test_publish_that_fails_leaves_the_folder_with_its_scratch(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim = _claimed(run)
    dest = tmp_path / "runs" / "r"
    real = os.rename

    def refuse_publish(source: Any, target: Any) -> None:
        if Path(target) == dest:
            raise PermissionError(target)
        real(source, target)

    monkeypatch.setattr(os, "rename", refuse_publish)

    with pytest.raises(PermissionError):
        publish(claim, dest)
    assert sorted(os.listdir(claim.path)) == ["result", "scratch"]


def test_publish_whose_reply_was_lost_is_done(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim = _claimed(run)
    dest = tmp_path / "runs" / "r"
    real = os.rename

    def lost_reply(source: Any, target: Any) -> None:
        real(source, target)
        if Path(target) == dest:
            raise FileNotFoundError(source)

    monkeypatch.setattr(os, "rename", lost_reply)

    publish(claim, dest)
    assert sorted(os.listdir(dest)) == ["result"]
    assert _attempts(run) == []


def test_publish_after_another_worker_published_the_run_drops_this_attempt(
    run: Path, tmp_path: Path
) -> None:
    claim = _claimed(run)
    dest = tmp_path / "runs" / "r"
    (dest / "first").mkdir(parents=True)

    publish(claim, dest)

    assert os.listdir(dest) == ["first"]
    assert _attempts(run) == []


def test_a_folder_made_again_where_the_claim_was_is_not_published(
    run: Path, tmp_path: Path
) -> None:
    claim = _claimed(run)
    # Taken from this attempt, and a folder of its name made since.
    os.rename(claim.path, run / "free__y")
    claim.path.mkdir()

    with pytest.raises(HeldError, match="was taken from this attempt"):
        publish(claim, tmp_path / "runs" / "r")
    assert sorted(entries(run)) == sorted([claim.path.name, "free__y"])
    assert not (tmp_path / "runs").exists()


def test_files_of_an_attempt_whose_folder_was_taken_land_nowhere(
    tmp_path: Path,
) -> None:
    gone = tmp_path / "gone"
    with pytest.raises(FileNotFoundError):
        _ = SpecDirectory(gone).data
    with pytest.raises(FileNotFoundError):
        _ = SpecDirectory(gone).scratch
    with pytest.raises(FileNotFoundError):
        _save_result_bundle(
            [{"a": 1}],
            gone / "result",
            spec_classes=[dict],
            data_dir=gone / "data",
            label="x",
        )
    assert not gone.exists()


# --- whether a Slurm job has ended ---


def _slurm(
    monkeypatch: pytest.MonkeyPatch, squeue: str | Exception, sacct: str
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def run(*argv: str) -> str:
        calls.append(argv)
        if argv[0] == "scontrol":
            return "ClusterName          = c\n"
        out = squeue if argv[0] == "squeue" else sacct
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(slurm, "run", run)
    return calls


PURGED = subprocess.CalledProcessError(
    1, "squeue", output="", stderr="slurm_load_jobs error: Invalid job id specified"
)


@pytest.mark.parametrize(
    ("squeue", "sacct", "gone"),
    [
        pytest.param("5 RUNNING\n", "5|RUNNING|None\n", False, id="running"),
        pytest.param("5 COMPLETING\n", "5|COMPLETED|None\n", False, id="completing"),
        pytest.param("", "5|COMPLETED|None\n", True, id="ended"),
        pytest.param(PURGED, "5|CANCELLED by 1|None\n", True, id="purged"),
        pytest.param("", "", False, id="not-accounted-yet"),
        pytest.param("", "5|REQUEUED|None\n", False, id="requeued"),
        pytest.param(
            "", "5|REQUEUED|None\n5|PENDING|None\n", False, id="requeued-waits"
        ),
        pytest.param(
            "", "5|REQUEUED|None\n5|TIMEOUT|None\n", True, id="requeued-ended"
        ),
        pytest.param("", "5|NODE_FAIL|None\n", False, id="node-fail"),
        pytest.param("", "5|BOOT_FAIL|None\n", False, id="boot-fail"),
        pytest.param("", "5|CANCELLED|Kill task failed\n", False, id="kill-failed"),
        pytest.param(
            "6 RUNNING\n", "5|FAILED|None\n6|RUNNING|None\n", True, id="array"
        ),
        pytest.param("", "5|FAILED\n", False, id="malformed"),
        pytest.param(
            subprocess.CalledProcessError(1, "squeue", stderr="timeout"),
            "5|FAILED|None\n",
            False,
            id="squeue-fails",
        ),
        pytest.param(OSError("no squeue"), "5|FAILED|None\n", False, id="no-squeue"),
    ],
)
def test_a_slurm_job_is_gone_only_when_both_squeue_and_sacct_say_so(
    monkeypatch: pytest.MonkeyPatch, squeue: str | Exception, sacct: str, gone: bool
) -> None:
    _slurm(monkeypatch, squeue, sacct)

    assert slurm.gone("c", "5") is gone


def test_a_job_of_another_cluster_is_never_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _slurm(monkeypatch, "", "5|COMPLETED|None\n")

    assert not slurm.gone("other", "5")
    assert [call[0] for call in calls] == ["scontrol"]


def test_the_cluster_is_asked_of_scontrol_until_it_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: list[str | Exception] = [
        subprocess.TimeoutExpired("scontrol", 60),
        "ClusterName          = c\n",
    ]
    calls = []

    def run(*argv: str) -> str:
        calls.append(argv)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    monkeypatch.setattr(slurm, "run", run)

    assert slurm.cluster() is None
    assert slurm.cluster() == "c"
    assert slurm.cluster() == "c"
    assert len(calls) == 2
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "named")
    assert slurm.cluster() == "named"


def test_slurm_commands_ignore_the_users_output_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    def fake(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout="out")

    for name in ("SQUEUE_FORMAT", "SACCT_FORMAT", "SLURM_CLUSTERS"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("SLURM_CONF", "kept")
    monkeypatch.setattr(subprocess, "run", fake)

    assert slurm.run("squeue") == "out"
    assert not {"SQUEUE_FORMAT", "SACCT_FORMAT", "SLURM_CLUSTERS"} & seen.keys()
    assert seen["SLURM_CONF"] == "kept"


# --- short_lock ---


def _open_on(path: Path) -> bool:
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == str(path):
                return True
        except OSError:
            continue
    return False


def test_short_lock_is_held_with_no_file_open(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    with short_lock(path):
        assert path.exists()
        assert not _open_on(path)
    assert not path.exists()


def test_short_lock_breaks_a_lock_left_by_a_killed_holder(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    path.touch()
    old = path.stat().st_mtime - STALE_S - 1
    os.utime(path, (old, old))

    with short_lock(path):
        assert path.stat().st_mtime > old
    assert not path.exists()


def test_short_lock_leaves_the_lock_of_the_holder_that_broke_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.lock"
    with short_lock(path):
        path.unlink()
        path.touch()  # the next holder's
    assert path.exists()
