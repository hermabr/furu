from __future__ import annotations

import errno
import os
import socket
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from furu.locking import LockError
from furu.storage.attempt_lock import (
    EntryStatus,
    acquire,
    acquire_attempt,
    classify_entry,
    is_attempt_blocked,
)


def test_acquire_returns_none_when_already_published(tmp_path: Path) -> None:
    runs_dir = tmp_path / "fixed" / "runs"
    (runs_dir / "abc123").mkdir(parents=True)
    attempts_dir = tmp_path / "attempts" / "abc123"

    assert acquire(attempts_dir, "abc123", runs_dirs=[runs_dir]) is None
    assert not attempts_dir.exists()


def test_acquire_seeds_claims_verifies_and_releases_with_fresh_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    attempts_dir = tmp_path / "attempts" / "run1"
    (attempts_dir / ".trash__old").mkdir(parents=True)
    (attempts_dir / ".deleting__old").mkdir()

    released_names: list[str] = []
    for _ in range(2):
        with acquire_attempt(attempts_dir, "run1") as attempt_lock:
            assert attempt_lock is not None
            assert attempt_lock.is_held()
            parts = attempt_lock.claimed_name.split("__")
            assert len(parts) == 4
            assert parts[0] == "local"
            assert parts[1] == socket.gethostname().replace("__", "_")
            assert parts[2] == str(os.getpid())
            assert len(parts[3]) == 16
            # Dotted entries stay ignored alongside the single claimed dir.
            assert sorted(p.name for p in attempts_dir.iterdir()) == sorted(
                [".deleting__old", ".trash__old", attempt_lock.claimed_name]
            )
        non_dotted = [p.name for p in attempts_dir.iterdir() if not p.name.startswith(".")]
        assert len(non_dotted) == 1
        assert non_dotted[0].startswith("free__")
        assert len(non_dotted[0].split("__")[1]) == 16
        released_names.append(non_dotted[0])

    assert released_names[0] != released_names[1]


def test_acquire_mints_slurm_identity_when_in_slurm_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sphinx")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    attempts_dir = tmp_path / "attempts" / "run_slurm"

    with acquire_attempt(attempts_dir, "run_slurm") as attempt_lock:
        assert attempt_lock is not None
        parts = attempt_lock.claimed_name.split("__")
        assert parts[:4] == ["slurm", "sphinx", "4242", "0"]
        assert len(parts[4]) == 16


def test_release_runs_on_exception_and_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    attempts_dir = tmp_path / "attempts" / "run_exc"

    with pytest.raises(KeyboardInterrupt), acquire_attempt(attempts_dir, "run_exc"):
        raise KeyboardInterrupt

    entries = [p.name for p in attempts_dir.iterdir() if not p.name.startswith(".")]
    assert len(entries) == 1
    assert entries[0].startswith("free__")


@pytest.mark.parametrize(
    "bad_name",
    [
        "free__",
        "free__a__b",
        "local__host__notpid__tok",
        "local__host__0__tok",
        "local__host__123",
        "local__host__123__tok__extra",
        "slurm__cluster__123",
        "slurm__cluster__notjobid__0__tok",
        "slurm__cluster__123__badrestart__tok",
        "slurm__cluster__123__0__",
        "unexpected",
    ],
)
def test_malformed_names_classify_as_unknown_and_never_mkdir_seed(
    tmp_path: Path, bad_name: str
) -> None:
    status, _ = classify_entry(bad_name)
    assert status is EntryStatus.UNKNOWN

    attempts_dir = tmp_path / "attempts" / "run_bad"
    (attempts_dir / bad_name).mkdir(parents=True)

    with pytest.raises(LockError, match="unknown"):
        acquire(attempts_dir, "run_bad")

    assert sorted(p.name for p in attempts_dir.iterdir()) == [bad_name]


def test_blocked_local_folder_raises_actionable_error_with_free_manual(
    tmp_path: Path,
) -> None:
    attempts_dir = tmp_path / "attempts" / "run_local"
    held = attempts_dir / "local__other-node__98765__abcdef0123456789"
    held.mkdir(parents=True)

    with pytest.raises(LockError) as excinfo:
        acquire(attempts_dir, "run_local")

    message = str(excinfo.value)
    assert str(held) in message
    assert "free__manual" in message
    assert "dead" in message
    assert "remove" in message
    assert not (attempts_dir / "free__seed").exists()


def test_live_local_pid_on_same_host_blocks_and_dead_pid_is_adoptable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    host = socket.gethostname().replace("__", "_")
    attempts_dir = tmp_path / "attempts" / "run_pid"

    live_dir = attempts_dir / f"local__{host}__{os.getpid()}__0123456789abcdef"
    live_dir.mkdir(parents=True)
    with pytest.raises(LockError, match="free__manual"):
        acquire(attempts_dir, "run_pid")
    live_dir.rmdir()

    # Use a PID that os.kill reports as ESRCH.
    dead_name = f"local__{host}__999999999__0123456789abcdef"
    (attempts_dir / dead_name).mkdir()
    with patch("furu.storage.attempt_lock._is_pid_alive", return_value=False):
        status, _ = classify_entry(dead_name)
        assert status is EntryStatus.ADOPTABLE
        lock = acquire(attempts_dir, "run_pid")
        assert lock is not None
        lock.release()


def test_acquire_handles_ambiguous_nfs_oserror_when_dst_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    attempts_dir = tmp_path / "attempts" / "run_nfs"
    (attempts_dir / "free__seed").mkdir(parents=True)
    real_rename = os.rename

    def flaky_rename(src: Path, dst: Path) -> None:
        real_rename(src, dst)
        raise OSError(errno.ESTALE, "ambiguous NFS rename reply")

    monkeypatch.setattr("furu.storage.attempt_lock.os.rename", flaky_rename)
    attempt_lock = acquire(attempts_dir, "run_nfs")
    assert attempt_lock is not None
    assert attempt_lock.is_held()
    monkeypatch.undo()
    attempt_lock.release()


def test_acquire_fails_when_verification_listing_missing_dst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    attempts_dir = tmp_path / "attempts" / "run_lost"
    (attempts_dir / "free__seed").mkdir(parents=True)
    real_rename = os.rename

    def vanish_after_rename(src: Path, dst: Path) -> None:
        real_rename(src, dst)
        real_rename(dst, attempts_dir / "local__other__1__stolen")

    monkeypatch.setattr("furu.storage.attempt_lock.os.rename", vanish_after_rename)
    with pytest.raises(LockError, match="lost ownership"):
        acquire(attempts_dir, "run_lost")


# --- 2-gate Slurm owner check ------------------------------------------------


def test_slurm_owner_cluster_mismatch_is_unknown() -> None:
    status, _ = classify_entry(
        "slurm__other__100__0__abcdef0123456789",
        current_cluster="sphinx",
        slurm_run=lambda *_: pytest.fail("should not query slurm on cluster mismatch"),
    )
    assert status is EntryStatus.UNKNOWN


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETING", "STAGE_OUT"])
def test_slurm_owner_in_squeue_is_held(state: str) -> None:
    def fake_slurm(*argv: str) -> str:
        assert argv[0] == "squeue"
        return f"100                              {state}\n"

    status, _ = classify_entry(
        "slurm__sphinx__100__0__abcdef0123456789",
        current_cluster="sphinx",
        slurm_run=fake_slurm,
    )
    assert status is EntryStatus.HELD


def test_slurm_owner_squeue_unexpected_error_is_unknown() -> None:
    def fake_slurm(*argv: str) -> str:
        raise subprocess.CalledProcessError(1, argv, stderr="slurmctld down")

    status, _ = classify_entry(
        "slurm__sphinx__100__0__abcdef0123456789",
        current_cluster="sphinx",
        slurm_run=fake_slurm,
    )
    assert status is EntryStatus.UNKNOWN


@pytest.mark.parametrize(
    "sacct_state",
    [
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "CANCELLED by 1001",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "DEADLINE",
        "REVOKED",
    ],
)
def test_slurm_owner_clean_terminal_sacct_is_adoptable(sacct_state: str) -> None:
    def fake_slurm(*argv: str) -> str:
        if argv[0] == "squeue":
            raise subprocess.CalledProcessError(
                1, argv, stderr="slurm_load_jobs error: Invalid job id specified"
            )
        assert argv == (
            "sacct",
            "-D",
            "-X",
            "--noheader",
            "--parsable2",
            "-o",
            "JobIDRaw,State,Reason",
            "-j",
            "100",
        )
        return f"100|{sacct_state}|None\n"

    status, _ = classify_entry(
        "slurm__sphinx__100__0__abcdef0123456789",
        current_cluster="sphinx",
        slurm_run=fake_slurm,
    )
    assert status is EntryStatus.ADOPTABLE


@pytest.mark.parametrize(
    "sacct_output",
    [
        "100|NODE_FAIL|Node failure\n",
        "100|BOOT_FAIL|Boot failure\n",
        "100|FAILED|Kill task failed\n",
        "100|CANCELLED by 0|Kill task failed\n",
        "",
        "999|COMPLETED|None\n",
        "malformed-line\n",
    ],
)
def test_slurm_owner_unsafe_or_missing_sacct_fails_closed_as_unknown(
    sacct_output: str,
) -> None:
    def fake_slurm(*argv: str) -> str:
        if argv[0] == "squeue":
            return ""
        return sacct_output

    status, _ = classify_entry(
        "slurm__sphinx__100__0__abcdef0123456789",
        current_cluster="sphinx",
        slurm_run=fake_slurm,
    )
    assert status is EntryStatus.UNKNOWN


def test_slurm_owner_sacct_timeout_fails_closed_as_unknown() -> None:
    def fake_slurm(*argv: str) -> str:
        if argv[0] == "squeue":
            return ""
        raise subprocess.TimeoutExpired(argv, 60.0)

    status, _ = classify_entry(
        "slurm__sphinx__100__0__abcdef0123456789",
        current_cluster="sphinx",
        slurm_run=fake_slurm,
    )
    assert status is EntryStatus.UNKNOWN


def test_is_attempt_blocked_checks_entries(tmp_path: Path) -> None:
    attempts_dir = tmp_path / "attempts" / "r1"
    assert not is_attempt_blocked(attempts_dir)
    (attempts_dir / "free__seed").mkdir(parents=True)
    assert not is_attempt_blocked(attempts_dir)
    (attempts_dir / "free__seed").rename(attempts_dir / "local__other__1__abcd")
    assert is_attempt_blocked(attempts_dir)
