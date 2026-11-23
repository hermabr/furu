"""Deterministic, sleep-free tests for Slurm helper, folder-name attempt locks,
SpecDirectory/bundle fencing, Spec.delete() races, and Manager lock-free probes."""

from __future__ import annotations

import errno
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from runs import fixed_run, run_of
from subprocess_objects import SubprocessCrashLeaf
from websockets.protocol import State
from websockets.sync.server import ServerConnection

import furu.execution.manager as manager_module
from furu import Spec, plan, slurm
from furu.config import get_config
from furu.core import working_folder
from furu.execution.manager import Manager, _Script
from furu.execution.run import Run
from furu.result.bundle import _mkdir_under, _save_result_bundle
from furu.snapshot import CodeLocation
from furu.storage import attempt_lock, events
from furu.storage._layout import (
    attempt_parent_in,
    attempts_dir_in,
    data_dir_in,
    list_names,
    scratch_dir_in,
)
from furu.storage.directory import SpecDirectory
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.execute import ChildSlot
from furu.worker.messages import (
    Job,
    JobFailedResult,
    Launch,
    LocalPool,
    ProcessSettings,
    WarningMessage,
)


class SimpleLeaf(Spec):
    key: int

    def create(self) -> int:
        return self.key * 10


# --- Slurm helper (furu.slurm) -------------------------------------------------


def test_slurm_run_scrubs_formatting_and_cluster_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SQUEUE_FORMAT", "%i")
    monkeypatch.setenv("SQUEUE_ALL", "1")
    monkeypatch.setenv("SACCT_FORMAT", "JobID")
    monkeypatch.setenv("SLURM_CLUSTERS", "other")
    monkeypatch.setenv("KEEP_ME", "yes")

    captured: dict[str, Any] = {}

    def fake_run(
        argv: tuple[str, ...], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)

    assert slurm.run("squeue", "-j", "1") == "ok\n"
    env = captured["env"]
    assert "SQUEUE_FORMAT" not in env
    assert "SQUEUE_ALL" not in env
    assert "SACCT_FORMAT" not in env
    assert "SLURM_CLUSTERS" not in env
    assert env["KEEP_ME"] == "yes"


def test_slurm_cluster_uses_env_and_caches_only_non_none_scontrol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(*argv: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FileNotFoundError("scontrol not installed")
        return "AuthType = auth/munge\nClusterName             = sc\n"

    monkeypatch.setattr(slurm, "run", fake_run)

    # Failed scontrol returns None and is NOT cached.
    assert slurm.cluster() is None
    assert calls == 1

    # Subsequent working scontrol succeeds and IS cached.
    assert slurm.cluster() == "sc"
    assert slurm.cluster() == "sc"
    assert calls == 2

    # Explicit SLURM_CLUSTER_NAME takes precedence over the cache.
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "override")
    assert slurm.cluster() == "override"
    assert calls == 2

    # Clearing cache forces a fresh probe when env is unset.
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    slurm.clear_cluster_cache()
    assert slurm.cluster() == "sc"
    assert calls == 3


@pytest.mark.parametrize(
    ("squeue_result", "sacct_result", "expected"),
    [
        # Live jobs in squeue stay False even if sacct had a terminal row.
        ("100 RUNNING\n", "100|COMPLETED|None\n", False),
        ("100 PENDING\n", "100|COMPLETED|None\n", False),
        ("100 COMPLETING\n", "100|COMPLETED|None\n", False),
        # Purged job: absent from squeue, empty sacct -> fail-closed False.
        ("INVALID_JOB", "", False),
        # Clean terminal states in _GONE -> True.
        ("INVALID_JOB", "100|COMPLETED|None\n", True),
        ("100 COMPLETED\n", "100|COMPLETED|None\n", True),
        ("INVALID_JOB", "100|FAILED|NonZeroExitCode\n", True),
        ("INVALID_JOB", "100|CANCELLED by 1234|None\n", True),
        ("INVALID_JOB", "100|TIMEOUT|TimeLimit\n", True),
        ("INVALID_JOB", "100|OUT_OF_MEMORY|OutOfMemory\n", True),
        ("INVALID_JOB", "100|PREEMPTED|Preempted\n", True),
        ("INVALID_JOB", "100|DEADLINE|None\n", True),
        ("INVALID_JOB", "100|REVOKED|None\n", True),
        # REQUEUED with a final terminal row -> True; without -> False (sacct lag).
        ("INVALID_JOB", "100|REQUEUED|Preempted\n100|COMPLETED|None\n", True),
        ("INVALID_JOB", "100|REQUEUED|Preempted\n", False),
        # NODE_FAIL and BOOT_FAIL stay fail-closed False.
        ("INVALID_JOB", "100|NODE_FAIL|None\n", False),
        ("INVALID_JOB", "100|BOOT_FAIL|None\n", False),
        # "Kill task failed" in Reason stays fail-closed False (case-insensitive).
        ("INVALID_JOB", "100|FAILED|Kill task failed\n", False),
        ("INVALID_JOB", "100|CANCELLED by 0|epilog: KILL TASK FAILED\n", False),
        # Sibling array rows are ignored in both squeue and sacct.
        (
            "101 RUNNING\n100 COMPLETED\n",
            "101|RUNNING|None\n100|COMPLETED|None\n",
            True,
        ),
        ("101 RUNNING\n", "101|NODE_FAIL|None\n100|COMPLETED|None\n", True),
        ("INVALID_JOB", "101|COMPLETED|None\n", False),
        # Malformed sacct line (len != 3) -> fail-closed False.
        ("INVALID_JOB", "100|COMPLETED\n", False),
        # squeue error without "Invalid job id" or with stdout -> fail-closed False.
        ("SQUEUE_BAD_ERR", "100|COMPLETED|None\n", False),
        ("SQUEUE_ERR_WITH_OUT", "100|COMPLETED|None\n", False),
        # sacct error -> fail-closed False.
        ("INVALID_JOB", "SACCT_ERR", False),
    ],
)
def test_slurm_gone_two_gate_table(
    monkeypatch: pytest.MonkeyPatch,
    squeue_result: str,
    sacct_result: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    seen_argvs: list[tuple[str, ...]] = []

    def fake_run(*argv: str) -> str:
        seen_argvs.append(argv)
        if argv[0] == "squeue":
            if squeue_result == "INVALID_JOB":
                raise subprocess.CalledProcessError(
                    1,
                    argv,
                    output="",
                    stderr="slurm_load_jobs error: Invalid job id specified",
                )
            if squeue_result == "SQUEUE_BAD_ERR":
                raise subprocess.CalledProcessError(
                    1, argv, output="", stderr="Unable to contact slurm controller"
                )
            if squeue_result == "SQUEUE_ERR_WITH_OUT":
                raise subprocess.CalledProcessError(
                    1, argv, output="100 RUNNING\n", stderr="Invalid job id"
                )
            return squeue_result
        if argv[0] == "sacct":
            if sacct_result == "SACCT_ERR":
                raise subprocess.TimeoutExpired(argv, 60.0)
            return sacct_result
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(slurm, "run", fake_run)

    assert slurm.gone("sc", "100") is expected
    assert seen_argvs[0] == (
        "squeue",
        "--all",
        "--array",
        "--noheader",
        "--format=%A %T",
        "-j",
        "100",
    )
    if len(seen_argvs) > 1:
        assert seen_argvs[1] == (
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


def test_slurm_gone_fails_closed_on_cluster_mismatch_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*argv: str) -> str:
        nonlocal called
        called = True
        return ""

    monkeypatch.setattr(slurm, "run", fail_if_called)
    monkeypatch.setattr(slurm, "cluster", lambda: None)
    assert slurm.gone("sc", "100") is False
    assert not called

    monkeypatch.setattr(slurm, "cluster", lambda: "other")
    assert slurm.gone("sc", "100") is False
    assert not called


# --- attempt_lock --------------------------------------------------------------


def test_me_falls_back_to_local_without_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setattr(slurm, "cluster", lambda: None)
    assert attempt_lock.me().startswith("local__")

    monkeypatch.setattr(slurm, "cluster", lambda: "sc")
    assert attempt_lock.me() == "slurm__sc__4242__0"

    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    assert attempt_lock.me() == "slurm__sc__4242__2"


def test_atomic_parent_install_contention_elects_one_winner(tmp_path: Path) -> None:
    run = tmp_path / "attempts" / "run1"
    barrier = threading.Barrier(4)
    wins: list[attempt_lock.Claim] = []
    errors: list[attempt_lock.HeldError] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            claim = attempt_lock.acquire(run, lambda: False)
            assert claim is not None
            with lock:
                wins.append(claim)
        except attempt_lock.HeldError as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(wins) == 1
    assert len(errors) == 3
    assert attempt_lock.entries(run) == [wins[0].path.name]
    # No stray .seed__* left in attempts/.
    assert list_names(run.parent) == ["run1"]


def test_delayed_seed_install_onto_existing_folder_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "attempts" / "run1"
    real_rename = os.rename
    injected = False

    def race_before_install(
        src: os.PathLike[str] | str, dst: os.PathLike[str] | str
    ) -> None:
        nonlocal injected
        if not injected and Path(dst) == run:
            injected = True
            # Another process wins and installs run/local__other__1__tok first.
            (run / "local__other__1__tok").mkdir(parents=True)
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", race_before_install)

    with pytest.raises(attempt_lock.HeldError, match="local process 1 on other"):
        attempt_lock.acquire(run, lambda: False)

    assert injected
    assert attempt_lock.entries(run) == ["local__other__1__tok"]
    assert list_names(run.parent) == ["run1"]


def test_transient_two_entry_listing_retries_and_persistent_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "attempts" / "run1"
    (run / "free__a").mkdir(parents=True)
    real_entries = attempt_lock.entries
    calls = 0

    def two_then_one(path: Path) -> list[str]:
        nonlocal calls
        if path == run:
            calls += 1
            if calls <= 2:
                return ["free__a", "free__b"]
        return real_entries(path)

    monkeypatch.setattr(attempt_lock, "entries", two_then_one)
    claim = attempt_lock.acquire(run, lambda: False)
    assert claim is not None
    assert calls >= 3

    # Persistent 2-entry listing fails closed.
    monkeypatch.setattr(attempt_lock, "entries", lambda path: ["free__a", "free__b"])
    assert "multiple entries" in (attempt_lock.held(run) or "")
    with pytest.raises(attempt_lock.HeldError, match="multiple entries"):
        attempt_lock.acquire(run, lambda: False)


@pytest.mark.parametrize("err_no", [errno.ENOENT, errno.ESTALE])
def test_lost_nfs_rename_reply_in_take_and_publish_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err_no: int
) -> None:
    run = tmp_path / "attempts" / "run1"
    (run / "free__prev").mkdir(parents=True)
    dest = tmp_path / "fixed" / "runs" / "run1"
    real_rename = os.rename

    def drop_reply(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        real_rename(src, dst)
        if Path(src).parent == run or Path(dst) == dest:
            raise OSError(err_no, "lost NFS reply")

    monkeypatch.setattr(os, "rename", drop_reply)

    claim = attempt_lock.acquire(run, lambda: False)
    assert claim is not None
    assert claim.path.is_dir()

    (claim.path / "marker").write_text("ok")
    attempt_lock.publish(claim.path, dest, claim.stat)
    assert (dest / "marker").read_text() == "ok"
    assert not run.exists()


def test_post_claim_published_removes_fresh_folder_and_preserves_adopted(
    tmp_path: Path,
) -> None:
    # Fresh claim: empty working folder + empty parent are both cleaned up.
    fresh_run = tmp_path / "attempts" / "fresh"
    assert attempt_lock.acquire(fresh_run, lambda: True) is None
    assert not fresh_run.exists()

    # Adopted claim: checkpoints inside free__prev are preserved under a new free__ name.
    adopted_run = tmp_path / "attempts" / "adopted"
    prev = adopted_run / "free__old"
    prev.mkdir(parents=True)
    (prev / "checkpoint.pt").write_text("weights")

    assert attempt_lock.acquire(adopted_run, lambda: True) is None
    remaining = attempt_lock.entries(adopted_run)
    assert len(remaining) == 1
    assert remaining[0].startswith("free__")
    assert (adopted_run / remaining[0] / "checkpoint.pt").read_text() == "weights"


def test_release_child_only_releases_matching_token_and_recovers_after_sigkill(
    tmp_path: Path,
) -> None:
    run = tmp_path / "attempts" / "run1"
    sibling = run / "slurm__sc__500__0__other_tok"
    sibling.mkdir(parents=True)

    # Different token: untouched!
    attempt_lock.release_child(run, "my_tok")
    assert attempt_lock.entries(run) == ["slurm__sc__500__0__other_tok"]

    # Already free__ with same token: untouched!
    sibling.rename(run / "free__my_tok")
    attempt_lock.release_child(run, "my_tok")
    assert attempt_lock.entries(run) == ["free__my_tok"]

    # Matching non-free token: released to a fresh free__<tok>.
    (run / "free__my_tok").rename(run / "slurm__sc__500__0__my_tok")
    attempt_lock.release_child(run, "my_tok")
    (released,) = attempt_lock.entries(run)
    assert released.startswith("free__") and released != "free__my_tok"

    # End-to-end subprocess SIGKILL recovery through ChildSlot.
    crashing = SubprocessCrashLeaf()
    slot = ChildSlot(worker="w0", backend="test")
    try:
        job = Job(
            specs=[crashing.object_id],
            lines=list(map(plan.dump, plan.lines([crashing]))),
            plan=get_config().directories.anchored().plans / "e1.plan",
            class_dir=crashing._class_dir,
            execution_log=get_config().directories.anchored().executions
            / "e1"
            / "execution.log",
            process=ProcessSettings.from_metadata(crashing._metadata),
            config=get_config().anchored(),
            code=CodeLocation.here(),
        )
        result = slot.run(job, cancelled=threading.Event())
    finally:
        slot.close()

    assert isinstance(result, JobFailedResult)
    assert "crash-leaf about to die" in result.error
    assert "SIGKILL" in result.error
    # ChildSlot released the dead child's attempt folder immediately.
    assert attempt_lock.held(crashing._attempt_dir) is None
    (after_kill,) = attempt_lock.entries(crashing._attempt_dir)
    assert after_kill.startswith("free__")


def test_publish_restores_scratch_on_transient_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "attempts" / "run1"
    dest = tmp_path / "fixed" / "runs" / "run1"
    claim = attempt_lock.acquire(run, lambda: False)
    assert claim is not None
    scratch = scratch_dir_in(claim.path)
    scratch.mkdir()
    (scratch / "ckpt").write_text("epoch1")

    real_rename = os.rename

    def fail_publish(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        if Path(dst) == dest:
            raise OSError(errno.EIO, "transient I/O error")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", fail_publish)

    with pytest.raises(OSError, match="transient I/O error"):
        attempt_lock.publish(claim.path, dest, claim.stat)

    assert not dest.exists()
    assert (scratch_dir_in(claim.path) / "ckpt").read_text() == "epoch1"
    assert [n for n in list_names(run.parent) if n.startswith(".trash__")] == []


def test_publish_winner_race_preserves_existing_and_cleans_loser(
    tmp_path: Path,
) -> None:
    run = tmp_path / "attempts" / "run1"
    dest = tmp_path / "fixed" / "runs" / "run1"
    dest.mkdir(parents=True)
    (dest / "winner").write_text("first")

    claim = attempt_lock.acquire(run, lambda: False)
    assert claim is not None
    (claim.path / "loser").write_text("second")
    (scratch_dir_in(claim.path)).mkdir()

    attempt_lock.publish(claim.path, dest, claim.stat)

    assert (dest / "winner").read_text() == "first"
    assert not (dest / "loser").exists()
    assert not run.exists()
    assert list_names(run.parent) == []


def test_publish_fences_recreated_folder_via_samestat(tmp_path: Path) -> None:
    run = tmp_path / "attempts" / "run1"
    dest = tmp_path / "fixed" / "runs" / "run1"
    claim = attempt_lock.acquire(run, lambda: False)
    assert claim is not None

    # Folder is renamed away and recreated at the same path (new inode).
    claim.path.rename(run / "slurm__sc__999__0__successor")
    claim.path.mkdir()

    with pytest.raises(attempt_lock.HeldError, match="inode changed"):
        attempt_lock.publish(claim.path, dest, claim.stat)

    assert not dest.exists()


# --- SpecDirectory & bundle fencing -------------------------------------------


def test_spec_directory_and_bundle_do_not_recreate_vanished_parents(
    tmp_path: Path,
) -> None:
    vanished = tmp_path / "attempts" / "run1" / "local__host__1__tok"
    directory = SpecDirectory(vanished)

    with pytest.raises(FileNotFoundError):
        _ = directory.data
    with pytest.raises(FileNotFoundError):
        _ = directory.scratch
    assert not vanished.exists()

    with pytest.raises(FileNotFoundError):
        _save_result_bundle(
            [1],
            vanished / "result",
            spec_classes=[SimpleLeaf],
            data_dir=data_dir_in(vanished),
            label="SimpleLeaf",
        )
    assert not vanished.exists()

    with pytest.raises(FileNotFoundError):
        _mkdir_under(vanished / "result", Path("artifacts", "a", "b"))
    assert not vanished.exists()


# --- Spec.delete() refusal & race ---------------------------------------------


def test_delete_refuses_before_prompt_when_attempt_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = SimpleLeaf(key=1)
    assert spec.create() == 10
    claim = attempt_lock.acquire(spec._attempt_dir, lambda: False)
    assert claim is not None

    prompted = False

    def fail_if_prompted(_prompt: str) -> str:
        nonlocal prompted
        prompted = True
        return "y"

    monkeypatch.setattr("builtins.input", fail_if_prompted)

    with pytest.raises(attempt_lock.HeldError, match="free__manual"):
        spec.delete(mode="prompt")

    assert not prompted
    assert spec.done


def test_delete_race_raises_held_error_without_touching_published_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = SimpleLeaf(key=2)
    assert spec.create() == 20
    published = run_of(spec)
    assert published.is_dir()

    # Prompt sees no held attempt, then another worker claims the attempt before discard().
    def claim_during_prompt(_prompt: str) -> str:
        claim = attempt_lock.acquire(spec._attempt_dir, lambda: False)
        assert claim is not None
        return "y"

    monkeypatch.setattr("builtins.input", claim_during_prompt)

    with pytest.raises(attempt_lock.HeldError):
        spec.delete(mode="prompt")

    # Published run is completely untouched!
    assert published.is_dir()
    assert spec.done


def test_delete_tombstones_via_attempts_trash_and_cleans_parent() -> None:
    spec = SimpleLeaf(key=3)
    assert spec.create() == 30
    (spec._attempt_dir / "free__leftover").mkdir(parents=True)

    assert spec.delete(mode="force") is True
    assert not spec.done
    assert not fixed_run(spec).exists()
    assert not spec._attempt_dir.exists()
    assert list_names(attempts_dir_in(spec._class_dir)) == []


# --- Manager lock-free probe, 30s throttle, warning dedup ----------------------


def test_manager_probes_outside_lock_throttles_30s_and_dedups_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sc")
    now = [1000.0]
    monkeypatch.setattr(
        manager_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )

    spec = SimpleLeaf(key=4)
    parent = attempt_parent_in(spec._class_dir, spec._artifact_hash)
    slurm_folder = parent / "slurm__sc__777__0__tok1"
    slurm_folder.mkdir(parents=True)
    events.start(
        slurm_folder,
        [spec._artifact_hash],
        get_config().directories.anchored().plans / "e1.plan",
        get_config().directories.anchored().plans / "e1.plan",
    )

    pool = LocalThreadWorkerBackend()
    run = Run(
        backends={pool.pool_key: pool},
        launch=Launch(
            plan=get_config().directories.anchored().plans / "e1.plan",
            execution_log=get_config().directories.anchored().executions
            / "e1"
            / "execution.log",
            config=get_config().anchored(),
            code=CodeLocation.here(),
        ),
    )
    groups = run.place([spec])
    submit = run.submit(
        groups, list(map(plan.dump, plan.lines([spec]))), run.launch.plan
    )

    manager = Manager(url="ws://unused", manager_dir=None)
    sent_messages: list[object] = []
    script = _Script(
        connection=cast(ServerConnection, SimpleNamespace(send=sent_messages.append)),
        pools={pool.pool_key: LocalPool(max_workers=1)},
    )
    manager.scripts.append(script)
    manager._add(script, submit)

    probe_calls = 0
    lock_free_during_probe: list[bool] = []
    squeue_state = ["777 RUNNING\n"]

    def fake_slurm_run(*argv: str) -> str:
        nonlocal probe_calls
        if argv[0] == "squeue":
            probe_calls += 1
            # Prove manager.lock is NOT held by checking from another thread.
            acquired: list[bool] = []

            def try_acquire() -> None:
                got = manager.lock.acquire(blocking=False)
                if got:
                    manager.lock.release()
                acquired.append(got)

            checker = threading.Thread(target=try_acquire)
            checker.start()
            checker.join()
            lock_free_during_probe.append(acquired[0])
            return squeue_state[0]
        if argv[0] == "sacct":
            return "777|COMPLETED|None\n"
        raise AssertionError(argv)

    monkeypatch.setattr(slurm, "run", fake_slurm_run)

    # First probe through _lease (close fake connection after first wait).
    class OneShotConnection:
        state = State.OPEN

    conn = OneShotConnection()
    real_wait = manager.lock.wait

    def stop_on_wait(timeout: float | None = None) -> bool:
        conn.state = State.CLOSED
        return False

    monkeypatch.setattr(manager.lock, "wait", stop_on_wait)
    assert manager._lease(cast(ServerConnection, conn), pool=pool.pool_key) is None
    assert probe_calls == 1
    assert lock_free_during_probe == [True]
    # _flush() ran outside self.lock and delivered exactly 1 WarningMessage.
    assert len(sent_messages) == 1
    first_warning = WarningMessage.model_validate_json(cast(str, sent_messages[0]))
    assert "777" in first_warning.reason

    # Within 30s: throttled (no second probe, no extra warning).
    now[0] += 29.0
    conn.state = State.OPEN
    assert manager._lease(cast(ServerConnection, conn), pool=pool.pool_key) is None
    assert probe_calls == 1
    assert len(sent_messages) == 1

    # At +30s: probes again; same why -> NO duplicate WarningMessage!
    now[0] += 1.0
    conn.state = State.OPEN
    assert manager._lease(cast(ServerConnection, conn), pool=pool.pool_key) is None
    assert probe_calls == 2
    assert lock_free_during_probe == [True, True]
    assert len(sent_messages) == 1

    # Folder owner changes -> different why -> emits 1 NEW WarningMessage!
    slurm_folder.rename(parent / "local__node1__42__tok2")
    now[0] += 30.0
    conn.state = State.OPEN
    assert manager._lease(cast(ServerConnection, conn), pool=pool.pool_key) is None
    assert len(sent_messages) == 2
    second_warning = WarningMessage.model_validate_json(cast(str, sent_messages[1]))
    assert "node1" in second_warning.reason

    # Once released and 30s elapsed, _lease hands out the job immediately!
    monkeypatch.setattr(manager.lock, "wait", real_wait)
    (parent / "local__node1__42__tok2").rename(parent / "free__done")
    now[0] += 30.0
    conn.state = State.OPEN
    lease = manager._lease(cast(ServerConnection, conn), pool=pool.pool_key)
    assert lease is not None
    assert lease.specs == [spec.object_id]
    assert lease.job.folder == working_folder([spec])
