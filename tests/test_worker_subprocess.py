from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from subprocess_objects import (
    OtherSubprocessEnvLeaf,
    SubprocessBlockedParent,
    SubprocessCrashLeaf,
    SubprocessDependencyLeaf,
    SubprocessEnvLeaf,
)

import furu
from furu import Metadata, Spec, Subprocess
from furu.metadata import ArtifactSpec
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.execute import ChildSlot
from furu.worker.protocol import (
    Job,
    JobBlockedResult,
    JobCompletedResult,
    JobFailedResult,
    JobResultRequest,
)


@pytest.fixture(autouse=True)
def _child_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    tests_directory = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        tests_directory if not existing else f"{tests_directory}{os.pathsep}{existing}",
    )


@pytest.fixture
def child_slot() -> Iterator[ChildSlot]:
    slot = ChildSlot()
    try:
        yield slot
    finally:
        slot.close()


def _run(slot: ChildSlot, obj: Spec[Any]) -> JobResultRequest:
    execution = obj._metadata.execution
    assert isinstance(execution, Subprocess)
    return slot.run(
        obj,
        job=Job(lease_id=str(uuid4()), artifact=ArtifactSpec.from_furu(obj)),
        execution=execution,
    )


def _pid_and_value(obj: Spec[str]) -> tuple[int, str]:
    pid, _, value = obj.load_existing().partition(":")
    return int(pid), value


def test_metadata_execution_defaults_to_inline() -> None:
    assert Metadata().execution == "inline"


def test_subprocess_environment_override_is_visible_in_child(
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    leaf = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
    )

    result = _run(child_slot, leaf)

    assert isinstance(result, JobCompletedResult)
    pid, value = _pid_and_value(leaf)
    assert value == "from-override"
    assert pid != os.getpid()


def test_subprocess_none_unsets_variable_in_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, child_slot: ChildSlot
) -> None:
    monkeypatch.setenv("FURU_TEST_VARIABLE", "from-parent")
    leaf = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value=None,
    )

    result = _run(child_slot, leaf)

    assert isinstance(result, JobCompletedResult)
    assert _pid_and_value(leaf)[1] == "None"


def test_subprocess_missing_required_environment_fails_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, child_slot: ChildSlot
) -> None:
    monkeypatch.delenv("FURU_TEST_REQUIRED", raising=False)
    leaf = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="irrelevant",
        required_environment=("FURU_TEST_REQUIRED",),
    )

    with pytest.raises(RuntimeError, match="FURU_TEST_REQUIRED"):
        _run(child_slot, leaf)
    assert child_slot._child is None


def test_subprocess_required_environment_satisfied_by_parent_or_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, child_slot: ChildSlot
) -> None:
    monkeypatch.setenv("FURU_TEST_REQUIRED", "from-parent")
    leaf = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="from-override",
        required_environment=("FURU_TEST_REQUIRED", "FURU_TEST_VARIABLE"),
    )

    assert isinstance(_run(child_slot, leaf), JobCompletedResult)


def test_subprocess_child_is_reused_across_jobs_with_same_environment(
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    first = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=1,
    )
    second = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=2,
    )

    assert isinstance(_run(child_slot, first), JobCompletedResult)
    assert isinstance(_run(child_slot, second), JobCompletedResult)

    assert _pid_and_value(first)[0] == _pid_and_value(second)[0]


def test_subprocess_environment_change_spawns_new_child(
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    first = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="one",
    )
    second = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="two",
    )

    assert isinstance(_run(child_slot, first), JobCompletedResult)
    assert isinstance(_run(child_slot, second), JobCompletedResult)

    assert _pid_and_value(first)[0] != _pid_and_value(second)[0]


def test_subprocess_reuse_never_gets_fresh_interpreter_and_leaves_nothing_warm(
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    first = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="pristine",
        reuse="never",
        marker=1,
    )
    second = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
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
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    looser = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        marker=1,
    )
    strict = OtherSubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        reuse="same_environment_same_spec",
        marker=1,
    )
    strict_again = OtherSubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="shared",
        reuse="same_environment_same_spec",
        marker=2,
    )
    looser_after = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
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
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    crashing = SubprocessCrashLeaf(storage_root=str(tmp_path))

    result = _run(child_slot, crashing)

    assert isinstance(result, JobFailedResult)
    assert "subprocess died: signal 9 (SIGKILL)" in result.error
    assert "crash-leaf about to die" in result.error
    assert child_slot._child is None

    follow_up = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="after-crash",
    )
    assert isinstance(_run(child_slot, follow_up), JobCompletedResult)
    assert _pid_and_value(follow_up)[1] == "after-crash"


def test_subprocess_blocked_dependency_is_relayed(
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    parent = SubprocessBlockedParent(storage_root=str(tmp_path))
    dependency = SubprocessDependencyLeaf(storage_root=str(tmp_path))

    result = _run(child_slot, parent)

    assert isinstance(result, JobBlockedResult)
    assert [artifact.object_id for artifact in result.dependencies] == [
        dependency.object_id
    ]


def test_subprocess_cache_hit_does_not_spawn_child(
    tmp_path: Path, child_slot: ChildSlot
) -> None:
    leaf = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="cached",
    )
    leaf.create()

    result = _run(child_slot, leaf)

    assert isinstance(result, JobCompletedResult)
    assert child_slot._child is None


def test_subprocess_execution_through_local_worker_backend(tmp_path: Path) -> None:
    leaf = SubprocessEnvLeaf(
        storage_root=str(tmp_path),
        variable_name="FURU_TEST_VARIABLE",
        variable_value="end-to-end",
    )

    result = furu.create(leaf, on=(LocalThreadWorkerBackend(),))

    pid, _, value = result.partition(":")
    assert value == "end-to-end"
    assert int(pid) != os.getpid()
