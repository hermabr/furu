from __future__ import annotations

import errno
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

import furu
from furu import Spec
from furu.storage._layout import data_dir_in, scratch_dir_in
from furu.storage.attempt_lock import (
    AttemptLockedError,
    AttemptState,
    Owner,
    SlurmQuery,
    acquire,
    acquire_attempt,
    active_attempt_dir,
    classify_entry,
    is_attempt_blocked,
    list_attempt_entries,
)
from furu.storage.directory import SpecDirectory


class _DeleteProbe(Spec):
    name: str

    def create(self) -> str:
        return f"val:{self.name}"


# --- owner identity & entry classification -------------------------------------


def test_owner_current_prefers_slurm_when_job_id_and_cluster_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "sphinx")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    assert Owner.current() == Owner("slurm__sphinx__98765__0")

    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    assert Owner.current() == Owner("slurm__sphinx__98765__3")

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert Owner.current() == Owner(f"local__{socket.gethostname()}__{os.getpid()}")


@pytest.mark.parametrize(
    "malformed",
    [
        "free",
        "free__",
        "free__a__b",
        "local__host__123",
        "local__host__0__tok",
        "local__host__notapid__tok",
        "local____123__tok",
        "local__host__123__",
        "local__host__123__tok__extra",
        "slurm__cluster__123__0",
        "slurm__cluster__abc__0__tok",
        "slurm__cluster__123__abc__tok",
        "slurm____123__0__tok",
        "slurm__cluster__123__0__",
        "slurm__cluster__123__0__tok__extra",
        "unknown__foo__bar",
        "stray_dir",
    ],
)
def test_malformed_names_classify_as_unknown_and_are_never_adoptable(
    tmp_path: Path, malformed: str
) -> None:
    assert classify_entry(malformed) is AttemptState.UNKNOWN

    attempt_root = tmp_path / "attempts" / "run1"
    (attempt_root / malformed).mkdir(parents=True)

    with pytest.raises(AttemptLockedError, match="unrecognized folder"):
        acquire(attempt_root)

    # Never falls through to mkdir("free__seed").
    assert list_attempt_entries(attempt_root) == [malformed]


def test_local_classification_and_actionable_error_message(tmp_path: Path) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    # Live PID on same host -> HELD.
    same_host_live = f"local__myhost__{os.getpid()}__deadbeefcafebabe"
    assert (
        classify_entry(same_host_live, local_host="myhost", is_pid_alive=lambda _: True)
        is AttemptState.HELD
    )
    # Dead PID on same host -> ADOPTABLE.
    assert (
        classify_entry(
            same_host_live, local_host="myhost", is_pid_alive=lambda _: False
        )
        is AttemptState.ADOPTABLE
    )
    # Different host -> UNKNOWN (no cross-host assumption or boot_id auto-takeover).
    other_host = "local__otherhost__4242__deadbeefcafebabe"
    assert (
        classify_entry(other_host, local_host="myhost", is_pid_alive=lambda _: False)
        is AttemptState.UNKNOWN
    )

    blocked = attempt_root / other_host
    blocked.mkdir(parents=True)
    with pytest.raises(AttemptLockedError) as excinfo:
        acquire(attempt_root, local_host="myhost")

    message = str(excinfo.value)
    assert str(blocked) in message
    assert f"mv {blocked} {attempt_root / 'free__manual'}" in message
    assert "resume its checkpoints" in message
    assert "remove" in message
    assert list_attempt_entries(attempt_root) == [other_host]


# --- acquire / release / contention / NFS verification -------------------------


def test_acquire_seeds_claims_releases_and_reclaims_with_fresh_tokens(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    # Dotted entries are ignored throughout.
    (attempt_root / ".trash__old").mkdir(parents=True)
    (attempt_root / ".deleting__old").mkdir(parents=True)

    owner = Owner("local__myhost__12345")
    with acquire_attempt(attempt_root, owner=owner) as first:
        assert first is not None
        assert re.fullmatch(r"local__myhost__12345__[0-9a-f]{16}", first.path.name)
        assert first.still_owned()
        (first.path / "checkpoint.bin").write_text("epoch1")
        first_tok = first.path.name.rsplit("__", maxsplit=1)[1]

    # Clean exit without publishing releases to free__<fresh_tok>.
    (released_name,) = list_attempt_entries(attempt_root)
    assert re.fullmatch(r"free__[0-9a-f]{16}", released_name)
    released_tok = released_name.rsplit("__", maxsplit=1)[1]
    assert released_tok != first_tok
    assert (attempt_root / released_name / "checkpoint.bin").read_text() == "epoch1"

    # Re-acquiring adopts the released folder and preserves checkpoints.
    with acquire_attempt(attempt_root, owner=owner) as second:
        assert second is not None
        assert (second.path / "checkpoint.bin").read_text() == "epoch1"
        second_tok = second.path.name.rsplit("__", maxsplit=1)[1]
        assert second_tok not in (first_tok, released_tok)


def test_acquire_returns_none_when_already_published_without_creating_attempt_dir(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"

    claimed = acquire(attempt_root, is_published=lambda: True)

    assert claimed is None
    assert not attempt_root.exists()


def test_acquire_refuses_when_any_entry_is_held_even_if_another_is_free(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    (attempt_root / "free__seed").mkdir(parents=True)
    (attempt_root / f"local__myhost__{os.getpid()}__abcd1234abcd1234").mkdir()

    with pytest.raises(AttemptLockedError, match="free__manual"):
        acquire(attempt_root, local_host="myhost", is_pid_alive=lambda _: True)


def test_acquire_recovers_from_ambiguous_nfs_rename_error_when_dst_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    real_rename = os.rename

    def ambiguous_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        real_rename(src, dst)
        raise OSError(errno.ESTALE, "simulated NFS retry error after server rename")

    monkeypatch.setattr(os, "rename", ambiguous_rename)

    claimed = acquire(attempt_root, owner=Owner("local__myhost__777"))
    assert claimed is not None
    assert claimed.path.is_dir()
    assert claimed.still_owned()


def test_acquire_retries_rename_race_and_sees_winner_or_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    real_rename = os.rename
    raced = False

    def racing_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal raced
        if not raced:
            raced = True
            # Another worker renamed free__seed first.
            real_rename(src, attempt_root / "local__other__555__1111222233334444")
            raise FileNotFoundError(errno.ENOENT, "already moved")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", racing_rename)

    with pytest.raises(AttemptLockedError, match="local__other__555__1111222233334444"):
        acquire(attempt_root, owner=Owner("local__myhost__777"), local_host="myhost")


# --- 2-gate conservative Slurm owner check -------------------------------------


def _fake_slurm(
    *,
    cluster: str | None = "sphinx",
    squeue: str | Exception = "",
    sacct: str | Exception = "",
    scontrol: str | Exception | None = None,
    calls: list[tuple[str, ...]] | None = None,
) -> SlurmQuery:
    def run(*argv: str) -> str:
        if calls is not None:
            calls.append(argv)
        cmd = argv[0]
        if cmd == "squeue":
            if isinstance(squeue, Exception):
                raise squeue
            return squeue
        if cmd == "sacct":
            if isinstance(sacct, Exception):
                raise sacct
            return sacct
        if cmd == "scontrol":
            if isinstance(scontrol, Exception):
                raise scontrol
            if scontrol is not None:
                return scontrol
        raise AssertionError(f"unexpected Slurm command: {argv}")

    return SlurmQuery(run=run, cluster=cluster)


def test_slurm_cluster_mismatch_fails_closed_without_calling_cli() -> None:
    calls: list[tuple[str, ...]] = []
    slurm = _fake_slurm(cluster="other", calls=calls)

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.UNKNOWN
    )
    assert calls == []


def test_slurm_cluster_discovered_from_scontrol_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    calls: list[tuple[str, ...]] = []
    slurm = _fake_slurm(
        cluster=None,
        scontrol="ClusterName             = sphinx\n",
        squeue="",
        sacct="12345|COMPLETED|None\n",
        calls=calls,
    )

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.ADOPTABLE
    )
    assert calls == [
        ("scontrol", "show", "config"),
        (
            "squeue",
            "--all",
            "--noheader",
            "-O",
            "JobIDRaw:32,State:32",
            "-j",
            "12345",
        ),
        (
            "sacct",
            "-D",
            "-X",
            "--noheader",
            "--parsable2",
            "-o",
            "JobIDRaw,State,Reason",
            "-j",
            "12345",
        ),
    ]


@pytest.mark.parametrize(
    "squeue_state", ["RUNNING", "PENDING", "COMPLETING", "STAGE_OUT"]
)
def test_slurm_any_squeue_row_is_held(squeue_state: str) -> None:
    calls: list[tuple[str, ...]] = []
    slurm = _fake_slurm(
        squeue=f"12345                           {squeue_state}\n",
        sacct="12345|COMPLETED|None\n",
        calls=calls,
    )

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.HELD
    )
    assert [argv[0] for argv in calls] == ["squeue"]


def test_slurm_squeue_invalid_job_id_proceeds_to_sacct() -> None:
    slurm = _fake_slurm(
        squeue=subprocess.CalledProcessError(
            1, ["squeue"], stderr="slurm_load_jobs error: Invalid job id specified\n"
        ),
        sacct="12345|COMPLETED|None\n",
    )

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.ADOPTABLE
    )


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(
            1, ["squeue"], stderr="slurm_load_jobs error: Unable to contact controller"
        ),
        subprocess.TimeoutExpired(["squeue"], 60.0),
        FileNotFoundError("squeue"),
    ],
)
def test_slurm_squeue_errors_fail_closed_as_unknown(error: Exception) -> None:
    slurm = _fake_slurm(squeue=error, sacct="12345|COMPLETED|None\n")

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.UNKNOWN
    )


@pytest.mark.parametrize(
    "sacct_state",
    [
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "CANCELLED by 12345",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "DEADLINE",
        "REVOKED",
    ],
)
def test_slurm_clean_terminal_sacct_states_are_adoptable(sacct_state: str) -> None:
    slurm = _fake_slurm(squeue="", sacct=f"12345|{sacct_state}|None\n")

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.ADOPTABLE
    )


@pytest.mark.parametrize(
    "sacct_out",
    [
        "",  # accounting lag
        "99999|COMPLETED|None\n",  # wrong jobid
        "12345|NODE_FAIL|Node failure\n",
        "12345|BOOT_FAIL|Boot failure\n",
        "12345|RUNNING|None\n",
        "12345|COMPLETING|None\n",
        "12345|FAILED|Kill task failed\n",
        "12345|CANCELLED by 0|Kill task failed\n",
        "12345|PREEMPTED|None\n12345|RUNNING|None\n",  # requeued duplicate still active
        "12345|PREEMPTED|None\n12345|NODE_FAIL|None\n",
        "malformed-line\n",
    ],
)
def test_slurm_unclean_or_ambiguous_sacct_fails_closed_as_unknown(
    sacct_out: str,
) -> None:
    slurm = _fake_slurm(squeue="", sacct=sacct_out)

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.UNKNOWN
    )


def test_slurm_sacct_cli_error_fails_closed_as_unknown() -> None:
    slurm = _fake_slurm(
        squeue="",
        sacct=subprocess.TimeoutExpired(["sacct"], 60.0),
    )

    assert (
        classify_entry("slurm__sphinx__12345__0__0123456789abcdef", slurm=slurm)
        is AttemptState.UNKNOWN
    )


# --- crash-safe publish() ------------------------------------------------------


def test_publish_moves_scratch_to_trash_before_rename_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    published = tmp_path / "fixed" / "runs" / "run1"
    seen_during_rename: dict[str, object] = {}
    real_rename = Path.rename

    with acquire_attempt(attempt_root) as claimed:
        assert claimed is not None
        scratch_dir_in(claimed.path).mkdir()
        (scratch_dir_in(claimed.path) / "ckpt.pt").write_text("tmp")
        (claimed.path / "result").mkdir()
        (claimed.path / "result" / "manifest.json").write_text("[]")

        def spy_rename(self: Path, target: Path) -> Path:
            if self == claimed.path and target == published:
                seen_during_rename["attempt_entries"] = sorted(
                    p.name for p in self.iterdir()
                )
                seen_during_rename["trash_entries"] = [
                    name
                    for name in os.listdir(attempt_root)
                    if name.startswith(".trash__")
                ]
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", spy_rename)
        claimed.publish(published)

    assert seen_during_rename["attempt_entries"] == ["result"]
    assert len(seen_during_rename["trash_entries"]) == 1  # ty: ignore[invalid-argument-type]
    assert (published / "result" / "manifest.json").read_text() == "[]"
    assert not scratch_dir_in(published).exists()
    assert not attempt_root.exists()


@pytest.mark.parametrize(
    "err",
    [errno.EIO, errno.EDQUOT, errno.ESTALE, errno.ENOENT],
)
def test_publish_transient_oserror_preserves_attempt_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    published = tmp_path / "fixed" / "runs" / "run1"

    def fail_rename(self: Path, target: Path) -> Path:
        raise OSError(err, os.strerror(err))

    monkeypatch.setattr(Path, "rename", fail_rename)

    with pytest.raises(OSError) as excinfo, acquire_attempt(attempt_root) as claimed:
        assert claimed is not None
        (claimed.path / "result").mkdir()
        (claimed.path / "result" / "manifest.json").write_text("finished")
        claimed.publish(published)

    assert excinfo.value.errno == err
    assert not published.exists()
    # Released cleanly to free__<tok> with the finished copy preserved intact.
    active = active_attempt_dir(attempt_root)
    assert active is not None
    assert active.name.startswith("free__")
    assert (active / "result" / "manifest.json").read_text() == "finished"


def test_publish_winner_race_trashes_loser_and_preserves_winner(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempts" / "run1"
    published = tmp_path / "fixed" / "runs" / "run1"
    published.mkdir(parents=True)
    (published / "winner.txt").write_text("winner")

    with acquire_attempt(attempt_root) as claimed:
        assert claimed is not None
        scratch_dir_in(claimed.path).mkdir()
        (claimed.path / "loser.txt").write_text("loser")
        claimed.publish(published)

    assert (published / "winner.txt").read_text() == "winner"
    assert not (published / "loser.txt").exists()
    assert not attempt_root.exists()


# --- SpecDirectory non-parents=True & Spec.delete() ----------------------------


def test_spec_directory_does_not_recreate_renamed_attempt_folder(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempts" / "run1" / "local__h__1__tok"
    attempt.mkdir(parents=True)
    directory = SpecDirectory(attempt)

    # Simulate another owner renaming our folder away.
    attempt.rename(attempt.with_name("free__manual"))

    with pytest.raises(FileNotFoundError):
        _ = directory.data
    with pytest.raises(FileNotFoundError):
        _ = directory.scratch
    assert not data_dir_in(attempt).exists()
    assert not attempt.exists()


def test_spec_delete_refuses_when_attempt_is_held_or_unknown() -> None:
    spec = _DeleteProbe(name="blocked")
    assert furu.create(spec) == "val:blocked"

    held_dir = spec._attempt_dir / "local__otherhost__9999__abcd1234abcd1234"
    held_dir.mkdir(parents=True)
    assert is_attempt_blocked(spec._attempt_dir)

    with pytest.raises(AttemptLockedError, match="free__manual"):
        spec.delete(mode="force")

    # Published run and held attempt remain untouched.
    assert spec.done
    assert held_dir.is_dir()

    # Once moved to free__manual, delete removes both the attempt and published run.
    held_dir.rename(spec._attempt_dir / "free__manual")
    assert spec.delete(mode="force")
    assert not spec.done
    assert not spec._attempt_dir.exists()
