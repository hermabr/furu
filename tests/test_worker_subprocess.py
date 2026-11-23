from __future__ import annotations

import importlib
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest
from gitrepo import git, init_repo
from runs import ends_of, events_of, log_of, run_of
from subprocess_objects import (
    OtherSubprocessEnvLeaf,
    SubprocessBatchLeaf,
    SubprocessBlockedParent,
    SubprocessChattyLeaf,
    SubprocessCrashLeaf,
    SubprocessCwdLeaf,
    SubprocessDependencyLeaf,
    SubprocessEnvLeaf,
    SubprocessForkingLeaf,
    SubprocessSnapshotsDirectoryLeaf,
)

import furu
from furu import Metadata, Spec, plan
from furu.config import _project_anchor, get_config
from furu.core import attempt_parent
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.snapshot import CodeLocation
from furu.storage import attempt_lock
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.execute import ChildSlot
from furu.worker.messages import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    ProcessSettings,
)


@pytest.fixture
def child_slot() -> Iterator[ChildSlot]:
    slot = ChildSlot(worker="w0", backend="test")
    try:
        yield slot
    finally:
        slot.close()


def _job(objs: list[Spec], *, code: CodeLocation | None = None) -> Job:
    return Job(
        specs=[obj.object_id for obj in objs],
        lines=list(map(plan.dump, plan.lines(objs))),
        plan=get_config().directories.anchored().plans / "e1.plan",
        class_dir=objs[0]._class_dir,
        execution_log=get_config().directories.anchored().executions
        / "e1"
        / "execution.log",
        process=ProcessSettings.from_metadata(objs[0]._metadata),
        config=get_config().anchored(),
        code=code or CodeLocation.here(),
    )


def _run(slot: ChildSlot, obj: Spec) -> JobResult:
    return slot.run(_job([obj]), cancelled=threading.Event())


def _pid_and_value(obj: furu.Creates[str]) -> tuple[int, str]:
    pid, _, value = obj.load_existing().partition(":")
    return int(pid), value


def test_metadata_defaults_to_warm_child_with_inherited_environment() -> None:
    metadata = Metadata()

    assert metadata.environment == {}
    assert metadata.required_environment_variables == ()
    assert metadata.reuse == "same_environment"


def test_subprocess_environment_override_is_visible_in_child(
    child_slot: ChildSlot,
) -> None:
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
    )

    result = _run(child_slot, leaf)

    assert isinstance(result, JobCompletedResult)
    pid, value = _pid_and_value(leaf)
    assert value == "from-override"
    assert pid != os.getpid()

    # The child's own record begins with a line about it, and the run's start
    # event names the file.
    record, finished = plan.read(Path(events_of(leaf)[1]["provenance"]))
    assert record["worker"].items() >= {"backend": "test", "pid": pid}.items()
    assert finished["run"] == leaf._artifact_hash


def test_subprocess_none_unsets_variable_in_child(
    monkeypatch: pytest.MonkeyPatch, child_slot: ChildSlot
) -> None:
    monkeypatch.setenv("FURU_TEST_VARIABLE", "from-parent")
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value=None,
    )

    result = _run(child_slot, leaf)

    assert isinstance(result, JobCompletedResult)
    assert _pid_and_value(leaf)[1] == "None"


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


def test_subprocess_required_environment_variables_satisfied_by_parent_or_override(
    monkeypatch: pytest.MonkeyPatch, child_slot: ChildSlot
) -> None:
    monkeypatch.setenv("FURU_TEST_REQUIRED", "from-parent")
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
        required_environment_variables=("FURU_TEST_REQUIRED", "FURU_TEST_VARIABLE"),
    )

    assert isinstance(_run(child_slot, leaf), JobCompletedResult)


def test_subprocess_child_is_reused_across_jobs_with_same_environment(
    child_slot: ChildSlot,
) -> None:
    first = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=1,
    )
    second = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=2,
    )

    assert isinstance(_run(child_slot, first), JobCompletedResult)
    assert isinstance(_run(child_slot, second), JobCompletedResult)

    assert _pid_and_value(first)[0] == _pid_and_value(second)[0]


def test_subprocess_environment_change_spawns_new_child(child_slot: ChildSlot) -> None:
    first = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="one",
    )
    second = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="two",
    )

    assert isinstance(_run(child_slot, first), JobCompletedResult)
    assert isinstance(_run(child_slot, second), JobCompletedResult)

    assert _pid_and_value(first)[0] != _pid_and_value(second)[0]


def test_subprocess_reuse_never_gets_fresh_interpreter_and_leaves_nothing_warm(
    child_slot: ChildSlot,
) -> None:
    first = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="pristine",
        reuse="never",
        marker=1,
    )
    second = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="pristine",
        reuse="never",
        marker=2,
    )

    assert isinstance(_run(child_slot, first), JobCompletedResult)
    assert child_slot._child is None
    assert isinstance(_run(child_slot, second), JobCompletedResult)
    assert child_slot._child is None

    assert _pid_and_value(first)[0] != _pid_and_value(second)[0]


def test_subprocess_same_environment_same_spec_honors_class_boundary(
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

    for obj in (looser, strict, strict_again, looser_after):
        assert isinstance(_run(child_slot, obj), JobCompletedResult)

    looser_pid = _pid_and_value(looser)[0]
    strict_pid = _pid_and_value(strict)[0]
    assert strict_pid != looser_pid
    # Same class and mapping: the strict child stays warm.
    assert _pid_and_value(strict_again)[0] == strict_pid
    # A looser job may reuse a strict job's leftover child; it opted into sharing.
    assert _pid_and_value(looser_after)[0] == strict_pid


def test_subprocess_crash_becomes_job_failed_result_and_slot_survives(
    child_slot: ChildSlot,
) -> None:
    crashing = SubprocessCrashLeaf()

    result = _run(child_slot, crashing)

    assert isinstance(result, JobFailedResult)
    # The log's tail comes first, so the reason is the error's last line.
    assert result.error == (
        "crash-leaf about to die\nsubprocess died: killed by signal 9 (SIGKILL)"
    )
    # A start with no end: the attempt was killed.
    assert [event["event"] for event in events_of(crashing)] == ["opened", "start"]
    assert child_slot._child is None
    # The worker saw its child die, so the folder is free for the next attempt.
    (left,) = attempt_lock.entries(attempt_parent([crashing]))
    assert left.startswith("free__")

    follow_up = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="after-crash",
    )
    assert isinstance(_run(child_slot, follow_up), JobCompletedResult)
    assert _pid_and_value(follow_up)[1] == "after-crash"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_kill_takes_down_what_the_job_forked(
    child_slot: ChildSlot, tmp_path: Path
) -> None:
    # The helper inherits the child's end of the protocol pipe; if it survives
    # the kill, the worker waits for it to exit before it sees the child died.
    pid_file = tmp_path / "helper.pid"
    results: list[JobResult] = []
    job = threading.Thread(
        target=lambda: results.append(
            _run(child_slot, SubprocessForkingLeaf(pid_file=str(pid_file)))
        )
    )
    job.start()
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    helper = int(pid_file.read_text())
    try:
        child_slot.kill()
        job.join(timeout=5)
        assert results, "the worker waited on the job's helper"
        (result,) = results
        assert isinstance(result, JobFailedResult)
        assert result.error.endswith("subprocess died: killed by signal 9 (SIGKILL)")
        deadline = time.monotonic() + 5
        while _alive(helper) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _alive(helper)
    finally:
        with suppress(ProcessLookupError):
            os.kill(helper, signal.SIGKILL)
        job.join()


def test_subprocess_blocked_dependency_is_relayed(child_slot: ChildSlot) -> None:
    parent = SubprocessBlockedParent()
    dependency = SubprocessDependencyLeaf()

    result = _run(child_slot, parent)

    assert isinstance(result, JobBlockedResult)
    assert result.dependencies == [dependency.object_id]
    # The worker saved the line of the spec it found in a plan file of its
    # own, in the launch's folder.
    found = {"fields": {"marker": 0}, "runs": True, "spec": dependency.object_id}
    assert result.lines == [plan.dump(found)]
    assert result.plan.parent == get_config().directories.anchored().plans / "e1"
    assert ends_of(parent) == [
        {"outcome": "blocked", "waiting_on": [dependency.object_id]}
    ]


def test_subprocess_cache_hit_completes_without_recreating(
    child_slot: ChildSlot,
) -> None:
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="cached",
    )
    leaf.create()
    assert _pid_and_value(leaf)[0] == os.getpid()

    result = _run(child_slot, leaf)

    assert isinstance(result, JobCompletedResult)
    # The child saw the cached result and did not run create() again.
    assert _pid_and_value(leaf)[0] == os.getpid()


def _snapshot_repo(repo: Path, marker: str) -> str:
    """Commit a repo whose checked-in venv python shims this interpreter, so a
    child spawned from the extracted snapshot actually runs."""
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python.chmod(0o755)
    (repo / "marker.txt").write_text(marker)
    return git(init_repo(repo), "rev-parse", "HEAD")


def test_child_receives_anchored_directories(child_slot: ChildSlot) -> None:
    # A child's cwd may be a snapshot, so it must not anchor relative paths itself.
    config = get_config()
    relative = config.directories.model_copy(
        update={"snapshots": Path("furu-data/anchor-probe")}
    )
    leaf = SubprocessSnapshotsDirectoryLeaf()
    with override_config(config.model_copy(update={"directories": relative})):
        assert isinstance(_run(child_slot, leaf), JobCompletedResult)
    assert _pid_and_value(leaf)[1] == str(
        _project_anchor() / "furu-data" / "anchor-probe"
    )


def test_one_worker_runs_jobs_from_two_repos_into_each_repo(tmp_path: Path) -> None:
    slot = ChildSlot(worker="w0", backend="test")
    pids_and_cwds: list[tuple[int, str]] = []
    code_dirs: list[Path] = []
    repo_roots: list[Path] = []
    try:
        for marker in (1, 2):
            repo = tmp_path / f"repo-{marker}"
            repo.mkdir()
            commit = _snapshot_repo(repo, marker=str(marker))
            config = get_config().anchored()
            config = config.model_copy(
                update={
                    "directories": config.directories.model_copy(
                        update={"objects": repo / "furu-data" / "objects"}
                    )
                }
            )
            with override_config(config):
                leaf = SubprocessCwdLeaf(marker=marker)
                provenance = SubmitProvenance(
                    git=GitIdentity(
                        commit=commit, branch=None, remote=None, repo_root=str(repo)
                    ),
                    environment=EnvironmentIdentity.capture().model_copy(
                        update={"project_root": str(repo)}
                    ),
                    submitted=SubmitContext.capture().model_copy(
                        update={"cwd": str(repo)}
                    ),
                )
                result = slot.run(
                    _job([leaf], code=CodeLocation.from_snapshot(provenance)),
                    cancelled=threading.Event(),
                )
                assert isinstance(result, JobCompletedResult)
                # The result lands in the repo that submitted it.
                assert run_of(leaf).is_relative_to(repo)
                pid, value = _pid_and_value(leaf)
            cwd, repo_root = value.split("\n")
            pids_and_cwds.append((pid, cwd))
            repo_roots.append(Path(repo_root))
            code_dirs.append(
                (get_config().directories.anchored().snapshots / commit).resolve()
            )
    finally:
        slot.close()

    # Each child ran inside its own job's extracted snapshot...
    assert [Path(cwd) for _, cwd in pids_and_cwds] == code_dirs
    assert code_dirs[0] != code_dirs[1]
    # ...yet sees the checkout it was submitted from, not the snapshot.
    assert repo_roots == [tmp_path / "repo-1", tmp_path / "repo-2"]
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


def test_subprocess_execution_through_local_worker_backend() -> None:
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="end-to-end",
    )

    result = furu.create(leaf, on=(LocalThreadWorkerBackend(),))

    pid, _, value = result.partition(":")
    assert value == "end-to-end"
    assert int(pid) != os.getpid()


def test_batched_subprocess_execution_through_local_worker_backend() -> None:
    objs = [SubprocessBatchLeaf(value=value) for value in (1, 2)]

    results = furu.create(objs, on=(LocalThreadWorkerBackend(),))
    parsed = [result.partition(":") for result in results]

    assert [value for _, _, value in parsed] == ["1", "2"]
    assert len({pid for pid, _, _ in parsed}) == 1
    assert int(parsed[0][0]) != os.getpid()
    assert run_of(objs[0]) == run_of(objs[1])


def test_job_output_lands_in_its_own_log_on_a_reused_child(
    child_slot: ChildSlot,
) -> None:
    first, second = SubprocessChattyLeaf(marker=1), SubprocessChattyLeaf(marker=2)
    SubprocessDependencyLeaf().create()

    assert isinstance(_run(child_slot, first), JobCompletedResult)
    assert isinstance(_run(child_slot, second), JobCompletedResult)
    assert _pid_and_value(first)[0] == _pid_and_value(second)[0]

    for obj in (first, second):
        # Only what the job wrote, through every channel, and nothing of furu's.
        printed, logged, raw = log_of(obj).splitlines()
        assert printed == f"print from job {obj.marker}"
        assert f"I chatty stdlib info from job {obj.marker}  [" in logged
        assert raw == f"raw fd 2 from job {obj.marker}"
