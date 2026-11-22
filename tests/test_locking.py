from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import furu.locking as locking_module
from furu.locking import (
    LockError,
    active_attempt_dir,
    attempt_lock,
    attempt_log_path,
    is_attempt_held,
    owns_attempt,
    short_lock,
)

SHORT_POLL_S = 0.01
SHORT_TIMEOUT_S = 0.1


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


# --- short_lock ----------------------------------------------------------------


def test_short_lock_acquires_excludes_and_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with short_lock(lock_path, timeout_s=1.0):
        assert lock_path.exists()
        assert not any(
            t.name.startswith("lock-heartbeat:") for t in threading.enumerate()
        )
        with (
            pytest.raises(LockError, match="could not acquire lock"),
            short_lock(lock_path, timeout_s=0.05, poll_s=SHORT_POLL_S),
        ):
            pass

    assert not lock_path.exists()


def test_short_lock_breaks_stale_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    lock_path.write_text("", encoding="utf-8")
    old = time.time() - 120.0
    os.utime(lock_path, (old, old))

    with short_lock(lock_path, stale_s=30.0, timeout_s=1.0):
        assert lock_path.exists()

    assert not lock_path.exists()


# --- attempt_lock: local and free ---------------------------------------------


def test_attempt_lock_seeds_claims_and_releases_with_fresh_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    parent = tmp_path / "attempts" / "run1"

    with attempt_lock(parent) as claimed:
        assert claimed is not None
        assert claimed.parent == parent
        assert re.fullmatch(
            rf"local__{re.escape(locking_module._local_host())}__{os.getpid()}__[0-9a-f]{{16}}",
            claimed.name,
        )
        assert owns_attempt(claimed)
        assert is_attempt_held(parent)
        assert active_attempt_dir(parent) == claimed
        assert attempt_log_path(parent) == claimed / "log"
        # Write a file so the attempt is kept and released to free__<fresh_tok>.
        (claimed / "events.jsonl").write_text("{}\n", encoding="utf-8")
        claimed_tok = claimed.name.rsplit("__", 1)[1]

    assert not is_attempt_held(parent)
    released = active_attempt_dir(parent)
    assert released is not None
    assert re.fullmatch(r"free__[0-9a-f]{16}", released.name)
    assert released.name.rsplit("__", 1)[1] != claimed_tok
    assert (released / "events.jsonl").exists()


def test_attempt_lock_cleans_up_empty_seeded_folder_when_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "attempts" / "run_empty"

    with attempt_lock(parent) as claimed:
        assert claimed is not None and claimed.is_dir()

    assert not parent.exists()


def test_attempt_lock_skips_when_is_published_is_true(tmp_path: Path) -> None:
    parent = tmp_path / "attempts" / "already_done"

    with attempt_lock(parent, is_published=lambda: True) as claimed:
        assert claimed is None

    assert not parent.exists()


def test_attempt_lock_adopts_free_and_dead_local_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "attempts" / "run_adopt"
    free_dir = parent / "free__0123456789abcdef"
    free_dir.mkdir(parents=True)
    (free_dir / "marker").write_text("kept", encoding="utf-8")

    with attempt_lock(parent) as claimed:
        assert claimed is not None
        assert (claimed / "marker").read_text(encoding="utf-8") == "kept"
        assert not free_dir.exists()

    # Simulate a dead process on this same host.
    dead_dir = parent / f"local__{locking_module._local_host()}__99999999__deadbeef01234567"
    released = active_attempt_dir(parent)
    assert released is not None
    released.rename(dead_dir)
    monkeypatch.setattr(locking_module, "_pid_is_alive", lambda pid: False)
    assert not is_attempt_held(parent)

    with attempt_lock(parent) as claimed:
        assert claimed is not None
        assert (claimed / "marker").read_text(encoding="utf-8") == "kept"
        assert not dead_dir.exists()


def test_attempt_lock_ignores_dotted_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "attempts" / "run_dotted"
    (parent / ".trash__abcdef").mkdir(parents=True)
    (parent / ".nfs00001").write_text("nfs", encoding="utf-8")

    assert not is_attempt_held(parent)
    with attempt_lock(parent) as claimed:
        assert claimed is not None
        assert owns_attempt(claimed)
        (claimed / "marker").write_text("ok", encoding="utf-8")

    assert active_attempt_dir(parent) is not None


def test_attempt_lock_refuses_held_live_local_without_creating_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "attempts" / "run_live"
    live_dir = parent / f"local__{locking_module._local_host()}__{os.getpid()}__0123456789abcdef"
    live_dir.mkdir(parents=True)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError, match=re.escape(str(live_dir))),
        attempt_lock(parent, acquire_timeout_s=0.05, acquire_poll_interval_s=SHORT_POLL_S),
    ):
        pass

    assert sorted(p.name for p in parent.iterdir()) == [live_dir.name]


def test_attempt_lock_refuses_stranded_remote_local_with_actionable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "attempts" / "run_remote"
    remote_dir = parent / "local__other-host.example__4242__0123456789abcdef"
    remote_dir.mkdir(parents=True)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError) as excinfo,
        attempt_lock(parent, acquire_timeout_s=10.0),
    ):
        pass

    msg = str(excinfo.value)
    assert str(remote_dir) in msg
    assert "other-host.example" in msg
    assert "4242" in msg
    assert "free__manual" in msg
    assert sorted(p.name for p in parent.iterdir()) == [remote_dir.name]


@pytest.mark.parametrize(
    "bad_name",
    [
        "unknown_folder",
        "free__",
        "local__host__notapid__tok",
        "local__host__0__tok",
        "slurm__cluster__123__notanint__tok",
        "slurm__cluster__123",
    ],
)
def test_attempt_lock_refuses_unknown_or_malformed_entry_without_sibling(
    tmp_path: Path, bad_name: str
) -> None:
    parent = tmp_path / "attempts" / "run_bad"
    bad_dir = parent / bad_name
    bad_dir.mkdir(parents=True)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError, match=re.escape(str(bad_dir))),
        attempt_lock(parent, acquire_timeout_s=10.0),
    ):
        pass

    assert sorted(p.name for p in parent.iterdir()) == [bad_name]


def test_attempt_lock_refuses_multiple_non_dotted_entries(tmp_path: Path) -> None:
    parent = tmp_path / "attempts" / "run_multi"
    (parent / "free__aaaaaaaaaaaaaaaa").mkdir(parents=True)
    (parent / "free__bbbbbbbbbbbbbbbb").mkdir(parents=True)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError, match="multiple active entries"),
        attempt_lock(parent, acquire_timeout_s=10.0),
    ):
        pass

    assert sorted(p.name for p in parent.iterdir()) == [
        "free__aaaaaaaaaaaaaaaa",
        "free__bbbbbbbbbbbbbbbb",
    ]


def test_attempt_lock_post_rename_verification_rejects_concurrent_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "attempts" / "run_race"
    existing = parent / "free__0123456789abcdef"
    existing.mkdir(parents=True)
    (existing / "marker").write_text("data", encoding="utf-8")

    real_rename = os.rename
    injected = False

    def rename_and_inject_sibling(src: Path, dst: Path) -> None:
        nonlocal injected
        real_rename(src, dst)
        if not injected:
            injected = True
            (parent / "local__other__1__sibling01234567").mkdir()

    with (
        patch.object(locking_module.os, "rename", side_effect=rename_and_inject_sibling),
        pytest.raises(LockError, match="unexpected concurrent entries"),
        attempt_lock(parent, acquire_timeout_s=0.0),
    ):
        pass

    # Our adopted folder was released back to free__<tok>, preserving its data.
    names = sorted(p.name for p in parent.iterdir())
    assert len(names) == 2
    free_names = [n for n in names if n.startswith("free__")]
    assert len(free_names) == 1
    assert (parent / free_names[0] / "marker").read_text(encoding="utf-8") == "data"


# --- attempt_lock: Slurm naming & 2-gate ownership check ----------------------


def test_attempt_lock_uses_raw_slurm_job_id_and_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "987600")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "54")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sail_cluster")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    parent = tmp_path / "attempts" / "run_slurm"

    with attempt_lock(parent) as claimed:
        assert claimed is not None
        assert re.fullmatch(
            r"slurm__sail_cluster__987654__2__[0-9a-f]{16}", claimed.name
        )


@pytest.mark.parametrize(
    ("squeue_res", "sacct_stdout"),
    [
        (_completed(stdout=""), "mycluster|COMPLETED|None\n"),
        (
            _completed(stderr="slurm_load_jobs error: Invalid job id specified\n", returncode=1),
            "mycluster|FAILED|NonZeroExitCode\n",
        ),
        (_completed(stdout=""), "mycluster|CANCELLED by 1000|None\n"),
        (_completed(stdout=""), "mycluster|TIMEOUT|TimeLimit\n"),
        (_completed(stdout=""), "mycluster|OUT_OF_MEMORY|OutOfMemory\n"),
        (_completed(stdout=""), "mycluster|DEADLINE|None\n"),
        (_completed(stdout=""), "mycluster|REVOKED|None\n"),
        (
            _completed(stdout=""),
            "mycluster|TIMEOUT|TimeLimit\nmycluster|COMPLETED|None\n",
        ),
    ],
)
def test_slurm_2gate_adopts_clean_terminal_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    squeue_res: subprocess.CompletedProcess[str],
    sacct_stdout: str,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mycluster")
    parent = tmp_path / "attempts" / "run_slurm_adopt"
    old_dir = parent / "slurm__mycluster__12345__0__0123456789abcdef"
    old_dir.mkdir(parents=True)
    (old_dir / "marker").write_text("checkpoint", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "squeue":
            return squeue_res
        assert argv[:5] == ["sacct", "-D", "-X", "--noheader", "--parsable2"]
        return _completed(stdout=sacct_stdout)

    monkeypatch.setattr(locking_module.subprocess, "run", fake_run)

    with attempt_lock(parent, acquire_timeout_s=0.0) as claimed:
        assert claimed is not None
        assert (claimed / "marker").read_text(encoding="utf-8") == "checkpoint"

    assert [c[0] for c in calls] == ["squeue", "sacct"]


@pytest.mark.parametrize(
    ("env_cluster", "squeue_res", "sacct_res", "expected_match"),
    [
        # Different cluster via env -> refuses before calling squeue/sacct.
        ("othercluster", None, None, "belongs to Slurm cluster"),
        # Active in squeue -> held.
        ("mycluster", _completed(stdout="12345 RUNNING\n"), None, "active Slurm job"),
        # squeue controller error -> fail-closed.
        (
            "mycluster",
            _completed(stderr="Unable to contact slurm controller", returncode=1),
            None,
            "via squeue",
        ),
        # sacct returns 0 rows -> fail-closed.
        ("mycluster", _completed(stdout=""), _completed(stdout=""), "returned no rows"),
        # sacct error -> fail-closed.
        (
            "mycluster",
            _completed(stdout=""),
            _completed(stderr="slurmdbd down", returncode=1),
            "via sacct",
        ),
        # sacct reports different cluster (even when env_cluster unset) -> refuse.
        (
            "",
            _completed(stdout=""),
            _completed(stdout="othercluster|COMPLETED|None\n"),
            "sacct reported cluster",
        ),
        # NODE_FAIL excluded -> refuse.
        (
            "mycluster",
            _completed(stdout=""),
            _completed(stdout="mycluster|NODE_FAIL|Node failure\n"),
            "unsafe state",
        ),
        # BOOT_FAIL excluded -> refuse.
        (
            "mycluster",
            _completed(stdout=""),
            _completed(stdout="mycluster|BOOT_FAIL|None\n"),
            "unsafe state",
        ),
        # Kill task failed excluded even on CANCELLED -> refuse.
        (
            "mycluster",
            _completed(stdout=""),
            _completed(stdout="mycluster|CANCELLED|Kill task failed\n"),
            "Kill task failed",
        ),
        # Duplicate rows where one is NODE_FAIL -> refuse (every row must be clean).
        (
            "mycluster",
            _completed(stdout=""),
            _completed(
                stdout="mycluster|COMPLETED|None\nmycluster|NODE_FAIL|None\n"
            ),
            "unsafe state",
        ),
        # Non-terminal / PREEMPTED state -> fail-closed.
        (
            "mycluster",
            _completed(stdout=""),
            _completed(stdout="mycluster|PREEMPTED|None\n"),
            "non-terminal state",
        ),
        (
            "mycluster",
            _completed(stdout=""),
            _completed(stdout="mycluster|COMPLETING|None\n"),
            "non-terminal state",
        ),
    ],
)
def test_slurm_2gate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_cluster: str,
    squeue_res: subprocess.CompletedProcess[str] | None,
    sacct_res: subprocess.CompletedProcess[str] | None,
    expected_match: str,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if env_cluster:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", env_cluster)
    else:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)

    parent = tmp_path / "attempts" / "run_slurm_closed"
    old_dir = parent / "slurm__mycluster__12345__0__0123456789abcdef"
    old_dir.mkdir(parents=True)

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == "squeue":
            assert squeue_res is not None
            return squeue_res
        assert argv[0] == "sacct"
        assert sacct_res is not None
        return sacct_res

    monkeypatch.setattr(locking_module.subprocess, "run", fake_run)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError, match=expected_match),
        attempt_lock(parent, acquire_timeout_s=0.0),
    ):
        pass

    assert sorted(p.name for p in parent.iterdir()) == [old_dir.name]


def test_slurm_2gate_fails_closed_on_subprocess_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mycluster")
    parent = tmp_path / "attempts" / "run_slurm_timeout"
    old_dir = parent / "slurm__mycluster__12345__0__0123456789abcdef"
    old_dir.mkdir(parents=True)

    def timeout_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 30.0)

    monkeypatch.setattr(locking_module.subprocess, "run", timeout_run)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError, match="via squeue"),
        attempt_lock(parent, acquire_timeout_s=0.0),
    ):
        pass
