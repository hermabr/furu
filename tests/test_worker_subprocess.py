from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from subprocess_objects import (
    OtherSubprocessEnvLeaf,
    SubprocessBlockedParent,
    SubprocessCrashLeaf,
    SubprocessCwdLeaf,
    SubprocessDependencyLeaf,
    SubprocessEnvLeaf,
)

import furu
from furu import Spec
from furu.config import get_config
from furu.metadata import ArtifactSpec
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.snapshot import create_snapshot
from furu.worker._child import _execute
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.context import worker_execution_context
from furu.worker.execute import ChildSlot
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    ProcessSettings,
)


@pytest.fixture
def child_slot() -> Iterator[ChildSlot]:
    slot = ChildSlot(backend="test", materialize_snapshot=False)
    try:
        yield slot
    finally:
        slot.close()


def _submit_provenance() -> SubmitProvenance:
    # Real environment identity so the child's lock-hash verification passes;
    # the git half is a stub.
    return SubmitProvenance(
        git=GitIdentity(
            commit="0" * 40,
            branch=None,
            remote=None,
            repo_root=".",
            dirty=False,
            diff_stats=None,
        ),
        environment=EnvironmentIdentity.capture(),
        snapshot_id=None,
        submitted=SubmitContext.capture(),
    )


def _job(obj: Spec) -> Job:
    return Job(
        artifacts=[ArtifactSpec.from_furu(obj)],
        provenance=_submit_provenance(),
        process=ProcessSettings.from_metadata(obj._metadata),
    )


def _run(slot: ChildSlot, obj: Spec) -> JobResult:
    return slot.run(_job(obj), cancelled=threading.Event())


def _pid_and_value(obj: Spec[str]) -> tuple[int, str]:
    pid, _, value = obj.load_existing().partition(":")
    return int(pid), value


def test_subprocess_child_gets_the_spec_environment(
    monkeypatch: pytest.MonkeyPatch, child_slot: ChildSlot
) -> None:
    monkeypatch.setenv("FURU_TEST_REQUIRED", "from-parent")
    monkeypatch.delenv("FURU_TEST_VARIABLE", raising=False)
    overridden = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
        # Satisfied by the parent and by the override, respectively.
        required_environment_variables=("FURU_TEST_REQUIRED", "FURU_TEST_VARIABLE"),
    )
    unset = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value=None,
    )

    assert isinstance(_run(child_slot, overridden), JobCompletedResult)
    monkeypatch.setenv("FURU_TEST_VARIABLE", "from-parent")
    assert isinstance(_run(child_slot, unset), JobCompletedResult)

    overridden_pid, overridden_value = _pid_and_value(overridden)
    unset_pid, unset_value = _pid_and_value(unset)
    assert overridden_value == "from-override"
    assert unset_value == "None"
    assert overridden_pid != os.getpid()
    # A different environment needs a different child.
    assert unset_pid != overridden_pid

    provenance = overridden.provenance()
    assert provenance.executed.worker_backend == "test"
    assert provenance.executed.pid == overridden_pid
    assert provenance.submitted.hostname == provenance.executed.hostname


def test_subprocess_missing_required_environment_variables_fails_before_spawn(
    monkeypatch: pytest.MonkeyPatch, child_slot: ChildSlot
) -> None:
    monkeypatch.delenv("FURU_TEST_REQUIRED", raising=False)
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="irrelevant",
        required_environment_variables=("FURU_TEST_REQUIRED",),
    )

    with pytest.raises(RuntimeError, match="FURU_TEST_REQUIRED"):
        _run(child_slot, leaf)
    assert child_slot._child is None


def test_subprocess_child_reuse_follows_the_reuse_policy(
    child_slot: ChildSlot,
) -> None:
    looser = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=1,
    )
    strict = OtherSubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        reuse="same_environment_same_spec",
        marker=1,
    )
    strict_again = OtherSubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        reuse="same_environment_same_spec",
        marker=2,
    )
    looser_after = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=2,
    )
    pristine = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        reuse="never",
    )

    for obj in (looser, strict, strict_again, looser_after, pristine):
        assert isinstance(_run(child_slot, obj), JobCompletedResult)

    looser_pid = _pid_and_value(looser)[0]
    strict_pid = _pid_and_value(strict)[0]
    assert strict_pid != looser_pid
    # Same class and mapping: the strict child stays warm.
    assert _pid_and_value(strict_again)[0] == strict_pid
    # A looser job may reuse a strict job's leftover child; it opted into sharing.
    assert _pid_and_value(looser_after)[0] == strict_pid
    # A never-reuse job gets a fresh interpreter and leaves nothing warm.
    assert _pid_and_value(pristine)[0] != strict_pid
    assert child_slot._child is None


def test_subprocess_crash_becomes_job_failed_result_and_slot_survives(
    child_slot: ChildSlot,
) -> None:
    crashing = SubprocessCrashLeaf()

    result = _run(child_slot, crashing)

    assert isinstance(result, JobFailedResult)
    assert "subprocess died: signal 9 (SIGKILL)" in result.error
    assert "crash-leaf about to die" in result.error
    assert child_slot._child is None

    follow_up = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="after-crash",
    )
    assert isinstance(_run(child_slot, follow_up), JobCompletedResult)
    assert _pid_and_value(follow_up)[1] == "after-crash"


def test_child_relays_blocked_dependency() -> None:
    with worker_execution_context():
        result = _execute(_job(SubprocessBlockedParent()))

    assert isinstance(result, JobBlockedResult)
    assert [artifact.object_id for artifact in result.dependencies] == [
        SubprocessDependencyLeaf().object_id
    ]


class _CountingLeaf(Spec[int]):
    calls_file: str

    def create(self) -> int:
        with open(self.calls_file, "a", encoding="utf-8") as f:
            f.write("x")
        return 1


def test_child_completes_cache_hit_without_recreating(tmp_path: Path) -> None:
    calls_file = tmp_path / "calls"
    leaf = _CountingLeaf(calls_file=str(calls_file))
    leaf.create()

    with worker_execution_context():
        result = _execute(_job(leaf))

    assert isinstance(result, JobCompletedResult)
    assert calls_file.read_text(encoding="utf-8") == "x"


def _snapshot_repo(repo: Path, marker: str) -> str:
    """Commit a repo whose checked-in venv python shims this interpreter, so a
    child spawned from the extracted snapshot actually runs."""
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python.chmod(0o755)
    (repo / "marker.txt").write_text(marker)
    for args in (
        ["init", "-q", "-b", "main"],
        ["add", "-Af"],  # -f: the user's global excludes may ignore .venv
        ["commit", "-qm", "snap"],
    ):
        subprocess.run(
            ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    return create_snapshot(repo)


def test_materializing_slot_runs_each_job_from_its_snapshot(tmp_path: Path) -> None:
    slot = ChildSlot(backend="test", materialize_snapshot=True)
    pids_and_cwds: list[tuple[int, str]] = []
    code_dirs: list[Path] = []
    try:
        for marker in (1, 2):
            repo = tmp_path / f"repo-{marker}"
            repo.mkdir()
            snapshot_id = _snapshot_repo(repo, marker=str(marker))
            leaf = SubprocessCwdLeaf(marker=marker)
            provenance = SubmitProvenance(
                git=GitIdentity(
                    commit="0" * 40,
                    branch=None,
                    remote=None,
                    repo_root=str(repo),
                    dirty=False,
                    diff_stats=None,
                ),
                # Stale lock hash: the job runs in its snapshot's venv, so the
                # worker's own environment must not be checked.
                environment=EnvironmentIdentity(
                    python="3.12.0",
                    uv="0",
                    project_root=str(repo),
                    uv_lock_hash="blake2s:stale",
                    pyproject_hash="blake2s:0",
                    furu="0",
                ),
                snapshot_id=snapshot_id,
                submitted=SubmitContext.capture().model_copy(update={"cwd": str(repo)}),
            )
            result = slot.run(
                Job(
                    artifacts=[ArtifactSpec.from_furu(leaf)],
                    provenance=provenance,
                    process=ProcessSettings.from_metadata(leaf._metadata),
                ),
                cancelled=threading.Event(),
            )
            assert isinstance(result, JobCompletedResult)
            pids_and_cwds.append(_pid_and_value(leaf))
            code_dirs.append(
                (
                    get_config().run_directories.snapshots / snapshot_id / "code"
                ).resolve()
            )
    finally:
        slot.close()

    # Each child ran inside its own job's extracted snapshot...
    assert [Path(cwd) for _, cwd in pids_and_cwds] == code_dirs
    assert code_dirs[0] != code_dirs[1]
    # ...so the warm first child was retired when the code changed.
    assert pids_and_cwds[0][0] != pids_and_cwds[1][0]


def test_spec_module_is_imported_only_in_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_name = "import_side_effect_objects"
    monkeypatch.setenv("FURU_TEST_IMPORT_MARKER_DIR", str(tmp_path))
    leaf = importlib.import_module(module_name).ImportSideEffectLeaf()
    # Forget the module so a parent-side import would leave a fresh marker;
    # the marker written while constructing the leaf is not the parent's.
    monkeypatch.delitem(sys.modules, module_name)
    (tmp_path / str(os.getpid())).unlink(missing_ok=True)

    child_pid = furu.create(leaf, on=(LocalThreadWorkerBackend(),))

    assert child_pid != os.getpid()
    assert {int(path.name) for path in tmp_path.iterdir()} == {child_pid}
