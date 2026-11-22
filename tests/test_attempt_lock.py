import errno
import os
import re
import subprocess
from pathlib import Path

import pytest

from furu.storage import attempt_lock as lock
from furu.storage.attempt_lock import State
from furu.storage.directory import SpecDirectory

TOKEN = "ab" * 8


def test_claim_release_and_manual_recovery(tmp_path: Path) -> None:
    parent = tmp_path / "a run"
    trash = parent / ".trash__garbage"
    trash.mkdir(parents=True)
    claim = lock.acquire(parent, lambda: False)
    assert claim is not None
    assert re.fullmatch(r"local__.+__[0-9]+__[0-9a-f]{16}", claim.name)
    (claim / "checkpoint").touch()
    with pytest.raises(RuntimeError) as error:
        lock.acquire(parent, lambda: False)
    assert str(claim) in str(error.value)
    assert "mv " in str(error.value) and "free__manual" in str(error.value)
    assert "dead" in str(error.value) and "remove" in str(error.value)
    claim.rename(parent / "free__manual")
    resumed = lock.acquire(parent, lambda: False)
    assert resumed is not None and resumed != claim
    assert (resumed / "checkpoint").exists()
    for child in ("data", "scratch"):
        with pytest.raises(FileNotFoundError):
            getattr(SpecDirectory(claim), child)
    lock.release(claim)
    assert lock.entries(parent) == [resumed]
    lock.release(resumed)
    (free,) = lock.entries(parent)
    assert re.fullmatch(r"free__[0-9a-f]{16}", free.name)
    again = lock.acquire(parent, lambda: False)
    assert again is not None
    lock.release(again)
    assert lock.entries(parent) != [free]
    assert trash.is_dir()


@pytest.mark.parametrize(
    "name",
    [
        "free",
        "free__",
        "free__bad",
        "free__manual__extra",
        "slurm__x",
        f"slurm__x__abc__0__{TOKEN}",
        f"slurm__x__123____{TOKEN}",
        f"local__x__NaN__{TOKEN}",
        f"local____3__{TOKEN}",
        "local__x__42__bad",
    ],
)
def test_unknown_entry_blocks_even_with_adoptable_folder(
    tmp_path: Path, name: str
) -> None:
    folder = tmp_path / name
    folder.mkdir()
    free = tmp_path / "free__manual"
    free.mkdir()
    assert lock.classify(folder) is State.UNKNOWN
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        lock.acquire(tmp_path, lambda: False)
    assert set(lock.entries(tmp_path)) == {free, folder}


def test_files_and_symlinks_cannot_be_claimed(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    parent.mkdir()
    entry = parent / "free__manual"
    entry.touch()
    assert lock.classify(entry) is State.UNKNOWN
    entry.unlink()
    entry.symlink_to(tmp_path, target_is_directory=True)
    assert lock.classify(entry) is State.UNKNOWN


def test_published_run_does_not_create_an_attempt(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    assert lock.acquire(parent, lambda: True) is None
    assert not parent.exists()
    checks = iter([False, True])
    assert lock.acquire(parent, lambda: next(checks)) is None
    assert not parent.exists()


def test_rename_rechecks_an_ambiguous_nfs_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = os.rename

    def lost_reply(src, dst):
        rename(src, dst)
        raise OSError(errno.EIO, "lost reply")

    monkeypatch.setattr(os, "rename", lost_reply)
    claim = lock.acquire(tmp_path, lambda: False)
    assert claim is not None
    assert lock.entries(tmp_path) == [claim]
    lock.release(claim)
    assert lock.entries(tmp_path)[0].name.startswith("free__")


def test_success_reply_without_claim_in_listing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "rename", lambda src, dst: None)
    with pytest.raises(RuntimeError, match="verify"):
        lock.acquire(tmp_path, lambda: False)
    assert lock.entries(tmp_path) == [tmp_path / "free__seed"]


def test_slurm_owner_uses_raw_job_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "one")
    monkeypatch.setenv("SLURM_JOB_ID", "1042")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    for restart in ("0", "7"):
        if restart != "0":
            monkeypatch.setenv("SLURM_RESTART_COUNT", restart)
        claim = lock.acquire(tmp_path, lambda: False)
        assert claim is not None
        assert re.fullmatch(rf"slurm__one__1042__{restart}__[0-9a-f]{{16}}", claim.name)
        lock.release(claim)


@pytest.mark.parametrize(
    ("queue", "accounts", "expected"),
    [
        *[
            (f"1042 {state}\n", None, State.HELD)
            for state in ("RUNNING", "PENDING", "COMPLETING", "STAGE_OUT")
        ],
        *[
            ("", f"1042|{state}|None\n", State.ADOPTABLE)
            for state in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "CANCELLED by 1001",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "PREEMPTED",
                "DEADLINE",
                "REVOKED",
            )
        ],
        ("", "1042|FAILED|None\n1042|COMPLETED|None", State.ADOPTABLE),
        ("", "1042|COMPLETED|None\n1042|NODE_FAIL|None", State.UNKNOWN),
        ("", "1042|BOOT_FAIL|None", State.UNKNOWN),
        ("", "1042|COMPLETED|Kill task failed", State.UNKNOWN),
        ("", "1042|CANCELLED pending|None", State.UNKNOWN),
        ("", "1042|RUNNING|None", State.UNKNOWN),
        ("", "", State.UNKNOWN),
        ("", "1042|FAILED", State.UNKNOWN),
        ("", "1000|COMPLETED|None", State.UNKNOWN),
        (
            subprocess.CalledProcessError(
                1, "squeue", stderr="slurm_load_jobs error: Invalid job id specified"
            ),
            "1042|FAILED|None",
            State.ADOPTABLE,
        ),
        (
            subprocess.CalledProcessError(
                1, "squeue", stderr="error contacting controller"
            ),
            None,
            State.UNKNOWN,
        ),
        (subprocess.TimeoutExpired("squeue", 60), None, State.UNKNOWN),
        (FileNotFoundError("squeue"), None, State.UNKNOWN),
        ("", subprocess.CalledProcessError(1, "sacct"), State.UNKNOWN),
        ("", subprocess.TimeoutExpired("sacct", 60), State.UNKNOWN),
    ],
)
def test_slurm_requires_both_gates(tmp_path, monkeypatch, queue, accounts, expected):
    claim = tmp_path / f"slurm__one__1042__0__{TOKEN}"
    claim.mkdir()
    calls = []

    def query(*args):
        calls.append(args)
        output = queue if args[0] == "squeue" else accounts
        if isinstance(output, BaseException):
            raise output
        assert output is not None
        return output

    monkeypatch.setattr(lock, "slurm", query)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "other")
    assert lock.classify(claim) is State.UNKNOWN
    assert not calls
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "one")
    assert lock.classify(claim) is expected
    assert calls[0] == (
        "squeue",
        "--all",
        "--noheader",
        "-O",
        "JobIDRaw:32,State:32",
        "-j",
        "1042",
    )
    if accounts is not None:
        assert calls[1:] == [
            (
                "sacct",
                "-D",
                "-X",
                "--noheader",
                "--parsable2",
                "-o",
                "JobIDRaw,State,Reason",
                "-j",
                "1042",
            )
        ]
    else:
        assert len(calls) == 1


def test_delayed_seed_does_not_grant_a_second_claim(tmp_path, monkeypatch):
    # Interleave a successful acquisition immediately before the first
    # contender's delayed seed mkdir, after both observed an empty directory.
    parent = tmp_path / "run"
    mkdir = Path.mkdir
    winner = None

    def late_mkdir(self, *args, **kwargs):
        nonlocal winner
        if self == parent / "free__seed":
            with monkeypatch.context() as inner:
                inner.setattr(Path, "mkdir", mkdir)
                winner = lock.acquire(parent, lambda: False)
        return mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", late_mkdir)
    with pytest.raises(lock.AttemptBusy):
        lock.acquire(parent, lambda: False)
    assert winner is not None
    assert lock.entries(parent) == [winner]
    lock.release(winner)


def test_a_seed_left_by_a_crashed_initializer_does_not_hide_checkpoints(
    tmp_path,
) -> None:
    free = tmp_path / "free__manual"
    free.mkdir()
    (free / "checkpoint").touch()
    (tmp_path / "free__seed").mkdir()
    claim = lock.acquire(tmp_path, lambda: False)
    assert claim is not None
    assert (claim / "checkpoint").exists()
    assert lock.entries(tmp_path) == [claim]


def test_enoent_with_source_still_listed_does_not_retry_forever(tmp_path, monkeypatch):
    def stale_lookup(src, dst):
        raise FileNotFoundError(errno.ENOENT, "negative lookup")

    monkeypatch.setattr(os, "rename", stale_lookup)
    with pytest.raises(FileNotFoundError):
        lock.acquire(tmp_path, lambda: False)
    assert lock.entries(tmp_path) == [tmp_path / "free__seed"]
