import errno
import os
from pathlib import Path

import pytest

from furu.locking import short_lock
from furu.storage import attempt
from furu.worker.backends.slurm import commands


def test_short_lock_is_exclusive_and_released_after_error(tmp_path: Path) -> None:
    path = tmp_path / "schema.lock"
    with pytest.raises(ValueError), short_lock(path):
        with pytest.raises(FileExistsError):
            path.open("x")
        raise ValueError
    assert not path.exists()


def test_claim_and_release_preserve_the_folder_with_fresh_names(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    parent = tmp_path / "run"
    with attempt.claim_attempt(parent) as first:
        assert first and first.name.startswith("local__")
        (first / "checkpoint").touch()
    (free,) = parent.iterdir()
    assert free.name.startswith("free__") and free.name != "free__seed"
    with attempt.claim_attempt(parent) as second:
        assert second and second != first
        assert (second / "checkpoint").is_file()
    (again,) = parent.iterdir()
    assert again != free


@pytest.mark.parametrize(
    "names",
    [
        ["local__host__42__0123456789abcdef"],
        ["unknown"],
        ["free__oops"],
        ["free__seed", "free__0123456789abcdef"],
        ["free__seed", "unknown"],
        ["slurm__other__123__0__0123456789abcdef"],
    ],
)
def test_refuses_occupied_and_ambiguous_parents(tmp_path, monkeypatch, names):
    parent = tmp_path / "run"
    parent.mkdir()
    for name in names:
        (parent / name).mkdir()
    monkeypatch.setattr(attempt, "_WAIT_SECONDS", 0)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "ours")
    with pytest.raises(RuntimeError, match=str(parent)), attempt.claim_attempt(parent):
        pytest.fail("entered held folder")
    assert sorted(os.listdir(parent)) == sorted(names)
    assert attempt.held(parent)


def test_free_name_must_be_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(attempt, "_WAIT_SECONDS", 0)
    (tmp_path / "free__seed").symlink_to(tmp_path, target_is_directory=True)
    with (
        pytest.raises(RuntimeError, match="Cannot claim"),
        attempt.claim_attempt(tmp_path),
    ):
        pytest.fail("followed symlink")


def test_seed_cannot_install_a_sibling_after_a_stale_empty_listing(tmp_path):
    parent = tmp_path / "run"
    with attempt.claim_attempt(parent) as owner:
        attempt._seed(parent)
        assert list(parent.iterdir()) == [owner]


def test_ambiguous_rename_reply_is_resolved_by_listing(tmp_path, monkeypatch):
    rename = os.rename

    def lost_reply(src, dst):
        rename(src, dst)
        if Path(src).name == "free__seed":
            raise OSError(errno.EIO, "reply lost")

    monkeypatch.setattr(os, "rename", lost_reply)
    with attempt.claim_attempt(tmp_path / "run") as owner:
        assert owner and owner.is_dir()


@pytest.mark.parametrize(
    ("queue", "extra", "expected"),
    [
        ("", "", True),
        ("117\n", "", False),
        ("bogus\n", "", False),
        ("", "117|0|CANCELLED by 100|None|\n", True),
        ("", "117|0|NODE_FAIL|None|\n", False),
        ("", "117|0|BOOT_FAIL|None|\n", False),
        ("", "117|0|FAILED|None|Kill task failed\n", False),
        ("", "117|0|FAILED|Kill task failed|\n", False),
        ("", "117|0|COMPLETING|None|\n", False),
        ("", "missing fields\n", False),
    ],
)
def test_slurm_requires_absent_job_and_every_accounting_row_clean(
    monkeypatch, queue, extra, expected
):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "ours")
    calls = []

    def query(command, *args):
        calls.append(command)
        if command == "squeue":
            return queue
        assert "-D" in args and "-X" in args and "117" in args
        return "117|1|COMPLETED|None|\n" + extra

    monkeypatch.setattr(commands, "slurm", query)
    assert commands.clean_exit("ours", "117", "1") is expected
    assert calls == (["squeue"] if queue else ["squeue", "sacct"])


def test_slurm_claim_uses_raw_job_id_and_requires_positive_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "ours")
    monkeypatch.setenv("SLURM_JOB_ID", "117")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "100")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "9")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    parent = tmp_path / "run"
    prior = parent / "slurm__ours__116__1__0123456789abcdef"
    prior.mkdir(parents=True)
    for output in ("", "116|0|COMPLETED|None|", "444|1|COMPLETED|None|"):
        monkeypatch.setattr(
            commands,
            "slurm",
            lambda *args, output=output: "" if args[0] == "squeue" else output,
        )
        assert attempt.held(parent)

    def unavailable(*args):
        raise FileNotFoundError

    monkeypatch.setattr(commands, "slurm", unavailable)
    assert attempt.held(parent)
    monkeypatch.setattr(
        commands,
        "slurm",
        lambda *args: "" if args[0] == "squeue" else "116|1|FAILED|None|",
    )
    with attempt.claim_attempt(parent) as mine:
        assert mine and mine != prior and mine.name.startswith("slurm__ours__117__1__")


def test_new_version_can_seed_after_a_crash_left_only_detached_scratch(tmp_path):
    trash = tmp_path / "run/.trash__0123456789abcdef"
    trash.mkdir(parents=True)
    (trash / "old").touch()
    with attempt.claim_attempt(trash.parent) as mine:
        assert mine and not list(mine.iterdir())
    assert not trash.exists()
