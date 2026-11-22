from __future__ import annotations

import errno
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
from multiprocessing import Process
from pathlib import Path
from unittest.mock import patch

import pytest

import furu.storage.attempts as attempts_module
from furu.locking import LockError, short_lock
from furu.storage._layout import (
    FIXED,
    attempt_parent_in,
    run_dir_in,
    trace_path_in,
)
from furu.storage.attempts import (
    acquire_attempt,
    has_published_run,
    is_attempt_held,
    resolve_attempt_dir,
    revalidated_listdir,
)

TIMEOUT_S = 10.0


def _child_claim_and_exit_abruptly(
    class_dir: Path, run: str, folder_out: Path
) -> None:
    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        folder_out.write_text(str(claimed.path), encoding="utf-8")
        os._exit(0)


def _child_hold_and_release(
    class_dir: Path, run: str, entered: Path, release: Path
) -> None:
    with acquire_attempt(class_dir, run, check_published=False):
        entered.touch()
        deadline = time.monotonic() + TIMEOUT_S
        while not release.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)


# --- short_lock (schema.ndjson) -----------------------------------------------


def test_short_lock_uses_o_excl_without_heartbeat_thread(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"

    with short_lock(lock_path):
        assert lock_path.exists()
        assert not any(
            "heartbeat" in thread.name for thread in threading.enumerate()
        )
        with pytest.raises(LockError, match="could not acquire lock"), short_lock(
            lock_path, stale_after_s=60.0, timeout_s=0.05, poll_interval_s=0.01
        ):
            pass

    assert not lock_path.exists()


def test_short_lock_breaks_stale_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    lock_path.write_text("", encoding="utf-8")
    stale_time = time.time() - 60.0
    os.utime(lock_path, (stale_time, stale_time))

    with short_lock(lock_path, stale_after_s=10.0, timeout_s=1.0):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_short_lock_raises_unexpected_open_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "schema.lock"
    with (
        patch("os.open", side_effect=OSError(errno.EINVAL, "bad open")),
        pytest.raises(OSError, match="bad open"),
        short_lock(lock_path),
    ):
        pass


# --- folder-name attempt lock --------------------------------------------------


def test_acquire_seeds_claims_and_releases_to_free_with_fresh_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)

    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        assert claimed.was_created
        assert claimed.is_held()
        assert claimed.path.parent == parent
        parts = claimed.name.split("__")
        assert parts[0] == "local"
        assert parts[2] == str(os.getpid())
        assert len(parts[3]) == 16
        assert [p.name for p in parent.iterdir()] == [claimed.name]
        first_token = parts[3]

    released = [p.name for p in parent.iterdir()]
    assert len(released) == 1
    assert released[0].startswith("free__")
    assert released[0].removeprefix("free__") != first_token
    assert len(released[0].removeprefix("free__")) == 16
    assert not is_attempt_held(parent)

    with acquire_attempt(class_dir, run, check_published=False) as reacquired:
        assert reacquired is not None
        assert not reacquired.was_created
        assert reacquired.name.split("__")[3] != first_token


def test_acquire_ignores_dotted_entries_in_attempt_parent(tmp_path: Path) -> None:
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    (parent / ".trash__deadbeef").mkdir(parents=True)

    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        assert resolve_attempt_dir(parent) == claimed.path


@pytest.mark.parametrize(
    "bad_name",
    [
        "bogus",
        "free__",
        "local__onlytwo",
        "local__host__notpid__abcd1234abcd1234",
        "local__host__0__abcd1234abcd1234",
        "slurm__cluster__123__0",
    ],
)
def test_acquire_refuses_malformed_entry_without_creating_sibling(
    tmp_path: Path, bad_name: str
) -> None:
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    (parent / bad_name).mkdir(parents=True)

    with (
        pytest.raises(LockError, match="malformed entry"),
        acquire_attempt(class_dir, run, check_published=False),
    ):
        pass

    assert [p.name for p in parent.iterdir()] == [bad_name]
    assert is_attempt_held(parent)
    assert resolve_attempt_dir(parent) == parent / bad_name


def test_acquire_refuses_multiple_active_entries_without_creating_sibling(
    tmp_path: Path,
) -> None:
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    (parent / "free__a").mkdir(parents=True)
    (parent / "free__b").mkdir(parents=True)

    with (
        pytest.raises(LockError, match="multiple active entries"),
        acquire_attempt(class_dir, run, check_published=False),
    ):
        pass

    assert sorted(p.name for p in parent.iterdir()) == ["free__a", "free__b"]
    assert is_attempt_held(parent)
    assert resolve_attempt_dir(parent) is None


def test_acquire_refuses_stranded_other_host_local_with_actionable_path_error(
    tmp_path: Path,
) -> None:
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    stranded = parent / "local__other-node__4242__0123456789abcdef"
    stranded.mkdir(parents=True)

    with (
        pytest.raises(LockError, match=str(stranded)) as excinfo,
        acquire_attempt(class_dir, run, check_published=False),
    ):
        pass

    assert "free__manual" in str(excinfo.value)
    assert "other-node" in str(excinfo.value)
    assert [p.name for p in parent.iterdir()] == [stranded.name]
    assert is_attempt_held(parent)


def test_acquire_adopts_dead_local_pid_on_same_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    folder_out = tmp_path / "claimed.txt"

    proc = Process(
        daemon=True,
        target=_child_claim_and_exit_abruptly,
        args=(class_dir, run, folder_out),
    )
    proc.start()
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0

    abandoned = Path(folder_out.read_text(encoding="utf-8"))
    assert abandoned.is_dir()
    assert not is_attempt_held(abandoned.parent)

    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        assert claimed.path != abandoned
        assert not abandoned.exists()


def test_acquire_waits_for_live_local_holder_and_times_out_if_still_held(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    entered = tmp_path / "entered"
    release = tmp_path / "release"
    proc = Process(
        daemon=True,
        target=_child_hold_and_release,
        args=(class_dir, run, entered, release),
    )
    proc.start()
    try:
        deadline = time.monotonic() + TIMEOUT_S
        while not entered.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)

        parent = attempt_parent_in(class_dir, run)
        assert is_attempt_held(parent)

        monkeypatch.setattr(attempts_module, "DEFAULT_LOCK_WAIT_LOG_INTERVAL_S", 0.02)
        furu_logger = logging.getLogger("furu")
        furu_logger.addHandler(caplog.handler)
        try:
            caplog.set_level(logging.INFO, logger="furu")
            with (
                pytest.raises(LockError, match="could not acquire attempt"),
                acquire_attempt(
                    class_dir,
                    run,
                    check_published=False,
                    acquire_timeout_s=0.08,
                    acquire_poll_interval_s=0.01,
                ),
            ):
                pass
        finally:
            furu_logger.removeHandler(caplog.handler)

        assert any("waiting for attempt at" in m for m in caplog.messages)

        # Now release and verify waiter acquires it.
        release.touch()
        with acquire_attempt(
            class_dir,
            run,
            check_published=False,
            acquire_timeout_s=TIMEOUT_S,
            acquire_poll_interval_s=0.01,
        ) as claimed:
            assert claimed is not None
            assert claimed.is_held()
    finally:
        release.touch()
        proc.join(timeout=TIMEOUT_S)


def test_acquire_raises_lost_lock_when_renamed_mid_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    class_dir = tmp_path / "cls"
    run = "abcd1234"

    with (
        pytest.raises(LockError, match="lost lock"),
        acquire_attempt(class_dir, run, check_published=False) as claimed,
    ):
        assert claimed is not None
        claimed.path.rename(claimed.path.with_name(f"free__{secrets.token_hex(8)}"))
        assert not claimed.is_held()


def test_acquire_skips_when_published_run_exists_via_revalidated_listing(
    tmp_path: Path,
) -> None:
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    published = run_dir_in(class_dir / FIXED, run)
    published.mkdir(parents=True)

    assert has_published_run(class_dir, run, code_version="fixed")
    with acquire_attempt(class_dir, run, code_version="fixed") as claimed:
        assert claimed is None
    assert not attempt_parent_in(class_dir, run).exists()


def test_revalidated_listdir_refreshes_from_root_and_handles_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    child = root / "a" / "b"
    assert revalidated_listdir(child, root=root) == []
    child.mkdir(parents=True)
    (child / "item").mkdir()
    assert revalidated_listdir(child, root=root) == ["item"]


# --- Slurm ownership and 2-gate check -----------------------------------------


def test_acquire_encodes_slurm_identity_with_raw_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "98765_4")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sherlock")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    class_dir = tmp_path / "cls"
    run = "abcd1234"

    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        parts = claimed.name.split("__")
        assert parts[:4] == ["slurm", "sherlock", "98765_4", "2"]
        assert len(parts[4]) == 16


def test_acquire_falls_back_to_local_when_slurm_cluster_name_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    class_dir = tmp_path / "cls"
    run = "abcd1234"

    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        assert claimed.name.startswith(f"local__{socket.gethostname()}__")


@pytest.mark.parametrize(
    "state",
    [
        "CANCELLED",
        "CANCELLED by 12345",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "OUT_OF_MEMORY",
        "REVOKED",
        "TIMEOUT",
    ],
)
@pytest.mark.parametrize(
    ("squeue_rc", "squeue_stderr"),
    [(0, ""), (1, "slurm_load_jobs error: Invalid job id specified")],
)
def test_slurm_two_gate_adopts_clean_terminal_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    squeue_rc: int,
    squeue_stderr: str,
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sherlock")
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    old = parent / "slurm__sherlock__100__0__0123456789abcdef"
    old.mkdir(parents=True)

    def fake_slurm(*argv: str) -> subprocess.CompletedProcess[str]:
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, squeue_rc, "", squeue_stderr)
        if argv[0] == "sacct":
            assert "-D" in argv and "-X" in argv
            return subprocess.CompletedProcess(argv, 0, f"100|{state}|None\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr(attempts_module, "_run_slurm", fake_slurm)

    assert not is_attempt_held(parent)
    with acquire_attempt(class_dir, run, check_published=False) as claimed:
        assert claimed is not None
        assert claimed.name.startswith("slurm__sherlock__200__0__")
        assert not old.exists()


@pytest.mark.parametrize(
    ("cluster_env", "squeue_res", "sacct_res"),
    [
        # Different or missing cluster fails closed without even calling squeue.
        ("other-cluster", (0, "", ""), (0, "100|COMPLETED|None\n", "")),
        (None, (0, "", ""), (0, "100|COMPLETED|None\n", "")),
        # Present in squeue -> held.
        ("sherlock", (0, "100 RUNNING\n", ""), (0, "100|COMPLETED|None\n", "")),
        # squeue failure -> fail closed.
        ("sherlock", (1, "", "slurm_load_jobs error: Socket timed out"), (0, "100|COMPLETED|None\n", "")),
        # sacct empty or failed -> fail closed.
        ("sherlock", (0, "", ""), (0, "", "")),
        ("sherlock", (0, "", ""), (1, "", "sacct error")),
        # Excluded terminal states & non-terminal states -> fail closed.
        ("sherlock", (0, "", ""), (0, "100|NODE_FAIL|None\n", "")),
        ("sherlock", (0, "", ""), (0, "100|BOOT_FAIL|None\n", "")),
        ("sherlock", (0, "", ""), (0, "100|PREEMPTED|None\n", "")),
        ("sherlock", (0, "", ""), (0, "100|REQUEUED|None\n", "")),
        ("sherlock", (0, "", ""), (0, "100|RUNNING|None\n", "")),
        ("sherlock", (0, "", ""), (0, "100|COMPLETING|None\n", "")),
        # Kill task failed -> fail closed.
        ("sherlock", (0, "", ""), (0, "100|FAILED|Kill task failed\n", "")),
        # Duplicate (-D) rows where one row is unclean -> fail closed.
        (
            "sherlock",
            (0, "", ""),
            (0, "100|COMPLETED|None\n100|NODE_FAIL|None\n", ""),
        ),
    ],
)
def test_slurm_two_gate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cluster_env: str | None,
    squeue_res: tuple[int, str, str],
    sacct_res: tuple[int, str, str],
) -> None:
    if cluster_env is None:
        monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    else:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", cluster_env)

    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    held = parent / "slurm__sherlock__100__0__0123456789abcdef"
    held.mkdir(parents=True)

    def fake_slurm(*argv: str) -> subprocess.CompletedProcess[str]:
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, *squeue_res)
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, *sacct_res)
        raise AssertionError(argv)

    monkeypatch.setattr(attempts_module, "_run_slurm", fake_slurm)

    assert is_attempt_held(parent)
    with (
        pytest.raises(LockError, match="could not acquire attempt"),
        acquire_attempt(
            class_dir,
            run,
            check_published=False,
            acquire_timeout_s=0.0,
        ),
    ):
        pass
    assert [p.name for p in parent.iterdir()] == [held.name]


def test_slurm_two_gate_fails_closed_on_subprocess_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sherlock")
    class_dir = tmp_path / "cls"
    run = "abcd1234"
    parent = attempt_parent_in(class_dir, run)
    (parent / "slurm__sherlock__100__0__0123456789abcdef").mkdir(parents=True)

    def boom(*argv: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 60.0)

    monkeypatch.setattr(attempts_module, "_run_slurm", boom)

    assert is_attempt_held(parent)


def test_traced_has_published_run_ignores_outdated_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_dir = tmp_path / "storage" / "Mod.Cls" / "part"
    outdated_ver = class_dir / "oldver"
    run_dir_in(outdated_ver, "abcd1234").mkdir(parents=True)
    trace_path_in(outdated_ver).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        attempts_module,
        "versions",
        lambda _dir: [(outdated_ver, "code changed")],
    )
    assert not has_published_run(class_dir, "abcd1234", code_version="traced")

    live_ver = class_dir / "livever"
    run_dir_in(live_ver, "abcd1234").mkdir(parents=True)
    monkeypatch.setattr(
        attempts_module,
        "versions",
        lambda _dir: [(outdated_ver, "code changed"), (live_ver, None)],
    )
    assert has_published_run(class_dir, "abcd1234", code_version="traced")
