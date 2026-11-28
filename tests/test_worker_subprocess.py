from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import fake_srun
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
    SubprocessHeldLeaf,
    SubprocessSnapshotsDirectoryLeaf,
)

import furu
from furu import Metadata, Spec, plan, slurm
from furu.config import _project_anchor, get_config
from furu.core import attempt_parent
from furu.provenance import (
    EnvironmentIdentity,
    GitIdentity,
    SubmitContext,
    SubmitProvenance,
)
from furu.snapshot import CodeLocation
from furu.storage._layout import list_names
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.execute import Children
from furu.worker.messages import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResult,
    ProcessSettings,
)


@pytest.fixture
def children() -> Iterator[Children]:
    children = Children(worker="w0", backend="test")
    try:
        yield children
    finally:
        children.close()


def _job(
    objs: list[Spec],
    *,
    code: CodeLocation | None = None,
    lease: int = 0,
    slots: int = 1,
) -> Job:
    return Job(
        lease=lease,
        slots=slots,
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


def _run(children: Children, obj: Spec, *, lease: int = 0, slots: int = 1) -> JobResult:
    job = _job([obj], lease=lease, slots=slots)
    return children.run(job, cancelled=threading.Event())


def _pid_and_value(obj: furu.Creates[str]) -> tuple[int, str]:
    pid, _, value = obj.load_existing().partition(":")
    return int(pid), value


def test_metadata_defaults_to_warm_child_with_inherited_environment() -> None:
    metadata = Metadata()

    assert metadata.environment == {}
    assert metadata.required_environment_variables == ()
    assert metadata.reuse == "same_environment"


def test_subprocess_environment_override_is_visible_in_child(
    children: Children,
) -> None:
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
    )

    result = _run(children, leaf)

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
    monkeypatch: pytest.MonkeyPatch, children: Children
) -> None:
    monkeypatch.setenv("FURU_TEST_VARIABLE", "from-parent")
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value=None,
    )

    result = _run(children, leaf)

    assert isinstance(result, JobCompletedResult)
    assert _pid_and_value(leaf)[1] == "None"


def test_subprocess_missing_required_environment_variables_fails_before_spawn(
    monkeypatch: pytest.MonkeyPatch, children: Children
) -> None:
    monkeypatch.delenv("FURU_TEST_REQUIRED", raising=False)
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="irrelevant",
        required_environment_variables=("FURU_TEST_REQUIRED",),
    )

    with pytest.raises(RuntimeError, match="FURU_TEST_REQUIRED"):
        _run(children, leaf)
    assert not children._idle


def test_subprocess_required_environment_variables_satisfied_by_parent_or_override(
    monkeypatch: pytest.MonkeyPatch, children: Children
) -> None:
    monkeypatch.setenv("FURU_TEST_REQUIRED", "from-parent")
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
        required_environment_variables=("FURU_TEST_REQUIRED", "FURU_TEST_VARIABLE"),
    )

    assert isinstance(_run(children, leaf), JobCompletedResult)


def test_subprocess_child_is_reused_across_jobs_with_same_environment(
    children: Children,
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

    assert isinstance(_run(children, first), JobCompletedResult)
    assert isinstance(_run(children, second), JobCompletedResult)

    assert _pid_and_value(first)[0] == _pid_and_value(second)[0]


def test_subprocess_environment_change_spawns_new_child(children: Children) -> None:
    first = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="one",
    )
    second = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="two",
    )

    assert isinstance(_run(children, first), JobCompletedResult)
    assert isinstance(_run(children, second), JobCompletedResult)

    assert _pid_and_value(first)[0] != _pid_and_value(second)[0]


def test_subprocess_reuse_never_gets_fresh_interpreter_and_leaves_nothing_warm(
    children: Children,
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

    assert isinstance(_run(children, first), JobCompletedResult)
    assert not children._idle
    assert isinstance(_run(children, second), JobCompletedResult)
    assert not children._idle

    assert _pid_and_value(first)[0] != _pid_and_value(second)[0]


def test_subprocess_same_environment_same_spec_honors_class_boundary(
    children: Children,
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
        assert isinstance(_run(children, obj), JobCompletedResult)

    looser_pid = _pid_and_value(looser)[0]
    strict_pid = _pid_and_value(strict)[0]
    assert strict_pid != looser_pid
    # Same class and mapping: the strict child stays warm.
    assert _pid_and_value(strict_again)[0] == strict_pid
    # A looser job may reuse a strict job's leftover child; it opted into sharing.
    assert _pid_and_value(looser_after)[0] == strict_pid


def test_subprocess_crash_becomes_job_failed_result_and_the_next_job_runs(
    children: Children,
) -> None:
    crashing = SubprocessCrashLeaf()

    result = _run(children, crashing)

    assert isinstance(result, JobFailedResult)
    # The log's tail comes first, so the reason is the error's last line.
    assert result.error == (
        "crash-leaf about to die\nsubprocess died: killed by signal 9 (SIGKILL)"
    )
    # A start with no end: the attempt was killed.
    assert [event["event"] for event in events_of(crashing)] == ["opened", "start"]
    assert not children._idle
    # The worker saw its child die, so the folder is free for the next attempt.
    (left,) = list_names(attempt_parent([crashing]))
    assert left.startswith("free__")

    follow_up = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="after-crash",
    )
    assert isinstance(_run(children, follow_up), JobCompletedResult)
    assert _pid_and_value(follow_up)[1] == "after-crash"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_kill_takes_down_what_the_job_forked(
    children: Children, tmp_path: Path
) -> None:
    # The helper inherits the child's end of the protocol pipe; if it survives
    # the kill, the worker waits for it to exit before it sees the child died.
    pid_file = tmp_path / "helper.pid"
    results: list[JobResult] = []
    job = threading.Thread(
        target=lambda: results.append(
            _run(children, SubprocessForkingLeaf(pid_file=str(pid_file)))
        )
    )
    job.start()
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    helper = int(pid_file.read_text())
    try:
        children.kill(0)
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


def test_subprocess_blocked_dependency_is_relayed(children: Children) -> None:
    parent = SubprocessBlockedParent()
    dependency = SubprocessDependencyLeaf()

    result = _run(children, parent)

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
    children: Children,
) -> None:
    leaf = SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE",
        variable_value="cached",
    )
    leaf.create()
    assert _pid_and_value(leaf)[0] == os.getpid()

    result = _run(children, leaf)

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


def test_child_receives_anchored_directories(children: Children) -> None:
    # A child's cwd may be a snapshot, so it must not anchor relative paths itself.
    config = get_config()
    relative = config.directories.model_copy(
        update={"snapshots": Path("furu-data/anchor-probe")}
    )
    leaf = SubprocessSnapshotsDirectoryLeaf()
    with override_config(config.model_copy(update={"directories": relative})):
        assert isinstance(_run(children, leaf), JobCompletedResult)
    assert _pid_and_value(leaf)[1] == str(
        _project_anchor() / "furu-data" / "anchor-probe"
    )


def test_one_worker_runs_jobs_from_two_repos_into_each_repo(tmp_path: Path) -> None:
    slot = Children(worker="w0", backend="test")
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
    children: Children,
) -> None:
    first, second = SubprocessChattyLeaf(marker=1), SubprocessChattyLeaf(marker=2)
    SubprocessDependencyLeaf().create()

    assert isinstance(_run(children, first), JobCompletedResult)
    assert isinstance(_run(children, second), JobCompletedResult)
    assert _pid_and_value(first)[0] == _pid_and_value(second)[0]

    for obj in (first, second):
        # Only what the job wrote, through every channel, and nothing of furu's.
        printed, logged, raw = log_of(obj).splitlines()
        assert printed == f"print from job {obj.marker}"
        assert f"I chatty stdlib info from job {obj.marker}  [" in logged
        assert raw == f"raw fd 2 from job {obj.marker}"


def _wait_for(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 30
    while not done():
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.fixture
def allocation() -> Iterator[Callable[[int], Children]]:
    """An allocation cut into slots. Outside Slurm, so its children are plain
    processes."""
    made: list[Children] = []

    def cut(slots: int) -> Children:
        made.append(Children(worker="w0", backend="test", slots=slots))
        return made[-1]

    yield cut
    for children in made:
        children.close()


def _env_leaf(value: str, marker: int = 0) -> SubprocessEnvLeaf:
    return SubprocessEnvLeaf(
        variable_name="FURU_TEST_VARIABLE", variable_value=value, marker=marker
    )


def _pid(leaf: SubprocessEnvLeaf) -> int:
    return _pid_and_value(leaf)[0]


def test_jobs_run_at_once_each_in_a_child_of_its_own(
    allocation: Callable[[int], Children], tmp_path: Path
) -> None:
    children = allocation(2)
    first, second = (
        SubprocessHeldLeaf(marker=marker, hold=str(tmp_path)) for marker in (1, 2)
    )
    with ThreadPoolExecutor() as threads:
        try:
            results = [
                threads.submit(_run, children, leaf, lease=leaf.marker)
                for leaf in (first, second)
            ]
            # Both are running before either may finish.
            _wait_for(lambda: len(list(tmp_path.glob("started-*"))) == 2)
        finally:
            (tmp_path / "go").touch()
        for result in results:
            assert isinstance(result.result(timeout=30), JobCompletedResult)
    one, two = int(first.load_existing()), int(second.load_existing())
    assert one != two

    # A job that takes every slot runs once the children that held them are
    # gone.
    whole = _env_leaf("whole")
    assert isinstance(_run(children, whole, slots=2), JobCompletedResult)
    assert not _alive(one) and not _alive(two)

    # And it makes way in turn for a job that takes a part.
    assert isinstance(_run(children, _env_leaf("part")), JobCompletedResult)
    assert not _alive(_pid(whole))


def test_idle_children_stay_warm_until_a_job_needs_their_slots(
    allocation: Callable[[int], Children],
) -> None:
    children = allocation(4)
    a, b, a_again = _env_leaf("a"), _env_leaf("b"), _env_leaf("a", marker=1)
    for leaf in (a, b, a_again):
        assert isinstance(_run(children, leaf), JobCompletedResult)
    pid_a, pid_b = _pid(a), _pid(b)
    # Two slots hold a warm child each, and a job goes back to its own.
    assert pid_a != pid_b
    assert _pid(a_again) == pid_a

    # A child is warm for jobs of its own size only; two slots are still free.
    wide = _env_leaf("a", marker=2)
    assert isinstance(_run(children, wide, slots=2), JobCompletedResult)
    pid_wide = _pid(wide)
    assert pid_wide not in (pid_a, pid_b)
    assert _alive(pid_a) and _alive(pid_b) and _alive(pid_wide)

    # No slot is free now: a new kind of job takes the place of the child
    # that has been idle the longest.
    c = _env_leaf("c")
    assert isinstance(_run(children, c), JobCompletedResult)
    assert not _alive(pid_b)
    assert _alive(pid_a) and _alive(pid_wide)

    # And one that needs the whole allocation waits for every child to go.
    whole = _env_leaf("a", marker=3)
    assert isinstance(_run(children, whole, slots=4), JobCompletedResult)
    assert not any(map(_alive, (pid_a, pid_wide, _pid(c))))
    assert [child.slots for child in children._idle] == [4]


def test_a_cancelled_job_is_killed_and_the_others_run_on(
    allocation: Callable[[int], Children], tmp_path: Path
) -> None:
    children = allocation(2)
    doomed, kept = (
        SubprocessHeldLeaf(marker=marker, hold=str(tmp_path)) for marker in (1, 2)
    )
    with ThreadPoolExecutor() as threads:
        try:
            results = {
                leaf.marker: threads.submit(_run, children, leaf, lease=leaf.marker)
                for leaf in (doomed, kept)
            }
            _wait_for(lambda: len(list(tmp_path.glob("started-*"))) == 2)
            children.kill(1)
            killed = results[1].result(timeout=30)
            assert not results[2].done()
        finally:
            (tmp_path / "go").touch()
        assert isinstance(results[2].result(timeout=30), JobCompletedResult)

    assert isinstance(killed, JobFailedResult)
    assert killed.error.endswith("subprocess died: killed by signal 9 (SIGKILL)")
    assert not doomed.done and kept.done
    # The slot of the killed job is free again.
    assert isinstance(_run(children, _env_leaf("next")), JobCompletedResult)


def test_a_job_the_allocation_has_no_room_for_fails(
    allocation: Callable[[int], Children], tmp_path: Path
) -> None:
    children = allocation(2)
    running = SubprocessHeldLeaf(marker=1, hold=str(tmp_path))
    with ThreadPoolExecutor() as threads:
        try:
            result = threads.submit(_run, children, running, slots=2)
            _wait_for(lambda: (tmp_path / "started-1").exists())
            # The manager counts slots, so it never sends this.
            with pytest.raises(RuntimeError, match="no room for a job on 1 of 2"):
                _run(children, _env_leaf("late"), lease=1)
        finally:
            (tmp_path / "go").touch()
        assert isinstance(result.result(timeout=30), JobCompletedResult)


# --- the steps of a Slurm job that is cut into slots ---------------------------


@dataclass
class _SlurmJob:
    """What the tests of a Slurm job's steps look at."""

    record_file: Path
    # The steps that were ended, and what the folder of ``watched`` held then.
    ended: list[tuple[str, list[str]]] = field(default_factory=list)
    watched: Spec | None = None
    # What ending a step raises, if it is not to work.
    refuses: Exception | None = None

    def steps(self) -> list[list[str]]:
        """The options of each srun call."""
        if not self.record_file.exists():
            return []
        return [
            json.loads(line)["argv"]
            for line in self.record_file.read_text().splitlines()
        ]

    def end_step(self, name: str) -> None:
        held = list_names(attempt_parent([self.watched])) if self.watched else []
        self.ended.append((name, held))
        if self.refuses is not None:
            raise self.refuses


@pytest.fixture
def slurm_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SlurmJob:
    """This process as a Slurm job of 8 CPUs on 4 cores, 4 GPUs and 16 GiB,
    whose srun notes its options and runs the program as it is."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_srun.install(bin_dir)
    job = _SlurmJob(record_file=tmp_path / "srun.jsonl")
    for name, value in {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FURU_FAKE_SLURM_RECORD_FILE": str(job.record_file),
        "SLURM_JOB_ID": "100",
        "SLURM_CLUSTER_NAME": "fake",
        "SLURM_CPUS_ON_NODE": "8",
        "SLURM_GPUS_ON_NODE": "4",
        "SLURM_MEM_PER_NODE": "16384",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(slurm, "_threads_per_core", lambda: 2)
    monkeypatch.setattr(slurm, "end_step", job.end_step)
    return job


def _share(options: list[str]) -> list[str]:
    """What a step asks Slurm for, without its name."""
    assert options[-1].startswith("--job-name=furu-")
    return options[:-1]


def test_a_child_of_a_slurm_job_cut_into_slots_is_a_step_with_its_share(
    slurm_job: _SlurmJob, allocation: Callable[[int], Children]
) -> None:
    children = allocation(4)
    part, half, whole = _env_leaf("part"), _env_leaf("half"), _env_leaf("whole")

    for leaf, slots in ((part, 1), (half, 2), (whole, 4)):
        assert isinstance(_run(children, leaf, slots=slots), JobCompletedResult)

    step = ["--exact", "--ntasks=1"]
    assert [_share(options) for options in slurm_job.steps()] == [
        [*step, "--cpus-per-task=2", "--gpus=1", "--mem=4096M", "--export=ALL"],
        [*step, "--cpus-per-task=4", "--gpus=2", "--mem=8192M", "--export=ALL"],
        [*step, "--cpus-per-task=8", "--gpus=4", "--mem=16384M", "--export=ALL"],
    ]
    # Each step has a name of its own to be found by.
    assert len({options[-1] for options in slurm_job.steps()}) == 3
    # The job ran in the step, with the environment of this process.
    assert _pid_and_value(part) == (_pid(part), "part")
    assert _pid(part) != os.getpid()
    # A step whose child exits is over: Slurm is not asked.
    children.close()
    assert slurm_job.ended == []


def test_a_child_is_reused_by_the_jobs_that_fit_its_step(
    slurm_job: _SlurmJob, allocation: Callable[[int], Children]
) -> None:
    children = allocation(2)
    first, second = _env_leaf("a"), _env_leaf("a", marker=1)

    for leaf in (first, second):
        assert isinstance(_run(children, leaf), JobCompletedResult)

    assert _pid(first) == _pid(second)
    assert len(slurm_job.steps()) == 1


def test_a_slurm_job_that_is_not_cut_into_slots_runs_its_children_as_it_is(
    slurm_job: _SlurmJob, children: Children
) -> None:
    assert isinstance(_run(children, _env_leaf("whole")), JobCompletedResult)

    assert slurm_job.steps() == []


def test_a_killed_step_is_ended_before_its_folder_is_given_up(
    slurm_job: _SlurmJob, allocation: Callable[[int], Children], tmp_path: Path
) -> None:
    children = allocation(2)
    doomed = SubprocessHeldLeaf(marker=1, hold=str(tmp_path))
    slurm_job.watched = doomed
    with ThreadPoolExecutor() as threads:
        result = threads.submit(_run, children, doomed)
        _wait_for(lambda: (tmp_path / "started-1").exists())
        children.kill(0)
        killed = result.result(timeout=30)

    assert isinstance(killed, JobFailedResult)
    assert killed.error.endswith("subprocess died: killed by signal 9 (SIGKILL)")
    # srun can be gone before its step is, so Slurm is asked for the step by
    # name while the folder is still the job's.
    ((name, held),) = slurm_job.ended
    assert f"--job-name={name}" == slurm_job.steps()[0][-1]
    assert [folder.rpartition("__")[0] for folder in held] == ["slurm__fake__100__0"]
    (left,) = list_names(attempt_parent([doomed]))
    assert left.startswith("free__")
    assert isinstance(_run(children, _env_leaf("next")), JobCompletedResult)


def test_a_step_slurm_does_not_call_ended_keeps_its_folder(
    slurm_job: _SlurmJob, allocation: Callable[[int], Children]
) -> None:
    children = allocation(2)
    crashing = SubprocessCrashLeaf()
    slurm_job.refuses = TimeoutError("Slurm still lists the killed steps ['100.0']")

    with pytest.raises(TimeoutError, match="still lists"):
        _run(children, crashing)

    # Something of the step may still write to the folder, so it stays the
    # Slurm job's: free once Slurm says that the job is gone.
    (left,) = list_names(attempt_parent([crashing]))
    assert left.startswith("slurm__fake__100__0__")
    # The worker itself goes on.
    assert not children._running and not children._idle
    slurm_job.refuses = None
    assert isinstance(_run(children, _env_leaf("next")), JobCompletedResult)
