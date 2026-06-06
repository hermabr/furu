from __future__ import annotations

import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from furu.execution.coordinator import ExecutionCoordinator
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import SlurmResources
from slurm_objects import SlurmTaskKind, SlurmWorkloadTask


A_WORKER_COUNT = 2
B_WORKER_COUNT = 3
TASK_DURATION_SECONDS = 0.2
FINAL_SHARED_COUNT = 12

pytestmark = [
    pytest.mark.slurm_integration,
    pytest.mark.skipif(
        os.environ.get("FURU_SLURM_INTEGRATION") != "1",
        reason="set FURU_SLURM_INTEGRATION=1 inside the Slurm test container",
    ),
]


@pytest.mark.parametrize(
    ("scenario_id", "worker_idle_timeout", "expect_worker_restarts"),
    [
        ("long-idle", 60.0, False),
        ("short-idle", 0.05, True),
    ],
    ids=["long-idle", "short-idle"],
)
def test_slurm_backend_runs_worker_job_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    scenario_id: str,
    worker_idle_timeout: float,
    expect_worker_restarts: bool,
) -> None:
    sbatch_path = shutil.which("sbatch")
    assert sbatch_path is not None

    sbatch_version = subprocess.run(
        ["sbatch", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "slurm" in sbatch_version.stdout.lower()

    tests_dir = Path(__file__).parent.resolve()
    monkeypatch.chdir(tests_dir)

    all_tasks, final_tasks = _build_workload(
        scenario_id=scenario_id,
        duration_seconds=TASK_DURATION_SECONDS,
    )
    ExecutionCoordinator.run(
        final_tasks,
        worker_backends=(
            SlurmWorkerBackend(
                max_workers=A_WORKER_COUNT,
                resources=SlurmResources(
                    partition="debug",
                    cpus_per_worker=1,
                ),
                worker_connect_host="127.0.0.1",
                coordinator_listen_host="0.0.0.0",
                job_name="furu-worker-a",
                poll_interval=0.1,
                worker_idle_timeout=worker_idle_timeout,
            ),
            SlurmWorkerBackend(
                max_workers=B_WORKER_COUNT,
                resources=SlurmResources(
                    partition="debug",
                    cpus_per_worker=2,
                ),
                worker_connect_host="127.0.0.1",
                coordinator_listen_host="0.0.0.0",
                job_name="furu-worker-b",
                poll_interval=0.1,
                worker_idle_timeout=worker_idle_timeout,
            ),
        ),
    )

    results = {task.task_id: task.load_existing() for task in all_tasks}

    assert len(all_tasks) == 50 + FINAL_SHARED_COUNT
    assert Counter(task.kind for task in all_tasks) == {
        "a_only": 16,
        "b_only": 20,
        "shared": 14 + FINAL_SHARED_COUNT,
    }
    assert all(task.status() == "completed" for task in all_tasks)

    expected_parent_ids = {task.task_id: _parent_task_ids(task) for task in all_tasks}
    assert {
        task_id: _str_list(result["parent_task_ids"])
        for task_id, result in results.items()
    } == expected_parent_ids

    a_worker_jobs: set[str] = set()
    b_worker_jobs: set[str] = set()
    for task in all_tasks:
        result = results[task.task_id]
        assert result["cwd"] == str(tests_dir)
        assert result["duration_seconds"] == TASK_DURATION_SECONDS
        assert result["scenario_id"] == scenario_id
        assert _str(result["task_id"]) == task.task_id
        assert _str(result["kind"]) == task.kind
        assert _str(result["slurm_job_id"]).isdigit()

        job_name = _str(result["slurm_job_name"])
        cpus_per_task = _int(result["slurm_cpus_per_task"])
        match task.kind:
            case "a_only":
                assert job_name == "furu-worker-a"
                assert cpus_per_task == 1
            case "b_only":
                assert job_name == "furu-worker-b"
                assert cpus_per_task == 2
            case "shared":
                assert job_name in {"furu-worker-a", "furu-worker-b"}
                assert cpus_per_task in {1, 2}

        if job_name == "furu-worker-a":
            a_worker_jobs.add(_str(result["slurm_job_id"]))
        elif job_name == "furu-worker-b":
            b_worker_jobs.add(_str(result["slurm_job_id"]))

    root_b_jobs = _worker_jobs_for(
        results, prefixes=("b0", "s0"), job_name="furu-worker-b"
    )
    stage1_b_jobs = _worker_jobs_for(
        results, prefixes=("b1",), job_name="furu-worker-b"
    )
    pre_b_gate_a_jobs = _worker_jobs_for(
        results,
        prefixes=("a0", "s0", "ag", "a1", "s1"),
        job_name="furu-worker-a",
    )
    stage2_a_jobs = _worker_jobs_for(
        results, prefixes=("a2",), job_name="furu-worker-a"
    )
    stage2_jobs = _worker_jobs_for(results, prefixes=("a2", "b2", "s2"))
    narrow_jobs = _worker_jobs_for(results, prefixes=("ns",))
    final_shared_jobs = _worker_jobs_for(results, prefixes=("sf",))

    assert len(narrow_jobs) == 2

    if expect_worker_restarts:
        assert len(a_worker_jobs) > A_WORKER_COUNT
        assert len(b_worker_jobs) > B_WORKER_COUNT
        assert root_b_jobs.isdisjoint(stage1_b_jobs)
        assert pre_b_gate_a_jobs.isdisjoint(stage2_a_jobs)
        assert narrow_jobs <= stage2_jobs
        assert len(stage2_jobs - narrow_jobs) >= 3
        assert (stage2_jobs - narrow_jobs).isdisjoint(final_shared_jobs)
        assert narrow_jobs <= final_shared_jobs
        assert final_shared_jobs - narrow_jobs
    else:
        assert len(a_worker_jobs) == A_WORKER_COUNT
        assert len(b_worker_jobs) == B_WORKER_COUNT
        assert stage1_b_jobs <= root_b_jobs
        assert stage2_a_jobs <= pre_b_gate_a_jobs

    _assert_worker_jobs_are_no_longer_active(a_worker_jobs | b_worker_jobs)


def _build_workload(
    *,
    scenario_id: str,
    duration_seconds: float,
) -> tuple[list[SlurmWorkloadTask], list[SlurmWorkloadTask]]:
    roots_a = [
        _task(scenario_id, duration_seconds, "a0", index, "a_only")
        for index in range(4)
    ]
    roots_b = [
        _task(scenario_id, duration_seconds, "b0", index, "b_only")
        for index in range(4)
    ]
    roots_shared = [
        _task(scenario_id, duration_seconds, "s0", index, "shared")
        for index in range(4)
    ]
    roots = [*roots_a, *roots_b, *roots_shared]

    a_gate = [
        _task(scenario_id, duration_seconds, "ag", index, "a_only", *roots)
        for index in range(4)
    ]

    stage1_a = [
        _task(scenario_id, duration_seconds, "a1", index, "a_only", a_gate[index])
        for index in range(4)
    ]
    stage1_b = [
        _task(scenario_id, duration_seconds, "b1", index, "b_only", a_gate[index])
        for index in range(4)
    ]
    stage1_shared = [
        _task(scenario_id, duration_seconds, "s1", index, "shared", a_gate[index])
        for index in range(4)
    ]
    stage1 = [*stage1_a, *stage1_b, *stage1_shared]

    b_gate = [
        _task(scenario_id, duration_seconds, "bg", index, "b_only", *stage1)
        for index in range(4)
    ]

    stage2_a = [
        _task(scenario_id, duration_seconds, "a2", index, "a_only", b_gate[index])
        for index in range(4)
    ]
    stage2_b = [
        _task(
            scenario_id,
            duration_seconds,
            "b2",
            index,
            "b_only",
            b_gate[index % 4],
        )
        for index in range(8)
    ]
    stage2_shared = [
        _task(scenario_id, duration_seconds, "s2", index, "shared", b_gate[index])
        for index in range(4)
    ]
    stage2 = [*stage2_a, *stage2_b, *stage2_shared]

    narrow_shared = [
        _task(scenario_id, duration_seconds, "ns", index, "shared", *stage2)
        for index in range(2)
    ]

    final_shared = [
        _task(
            scenario_id,
            duration_seconds,
            "sf",
            index,
            "shared",
            *narrow_shared,
            stage2_a[index % len(stage2_a)],
            stage2_b[index % len(stage2_b)],
            stage2_b[(index + 4) % len(stage2_b)],
            stage2_shared[index % len(stage2_shared)],
        )
        for index in range(FINAL_SHARED_COUNT)
    ]

    all_tasks = [
        *roots_a,
        *roots_b,
        *roots_shared,
        *a_gate,
        *stage1_a,
        *stage1_b,
        *stage1_shared,
        *b_gate,
        *stage2_a,
        *stage2_b,
        *stage2_shared,
        *narrow_shared,
        *final_shared,
    ]
    return all_tasks, final_shared


def _task(
    scenario_id: str,
    duration_seconds: float,
    prefix: str,
    index: int,
    kind: SlurmTaskKind,
    *parents: SlurmWorkloadTask,
) -> SlurmWorkloadTask:
    return SlurmWorkloadTask(
        task_id=f"{prefix}-{index:02d}",
        kind=kind,
        scenario_id=scenario_id,
        duration_seconds=duration_seconds,
        parents=parents,
    )


def _str(value: object) -> str:
    assert isinstance(value, str)
    return value


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _str_list(value: object) -> list[str]:
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return cast(list[str], value)


def _parent_task_ids(task: SlurmWorkloadTask) -> list[str]:
    return [cast(SlurmWorkloadTask, parent).task_id for parent in task.parents]


def _worker_jobs_for(
    results: dict[str, dict[str, object]],
    *,
    prefixes: tuple[str, ...],
    job_name: str | None = None,
) -> set[str]:
    return {
        _str(result["slurm_job_id"])
        for task_id, result in results.items()
        if task_id.startswith(prefixes)
        and (job_name is None or result["slurm_job_name"] == job_name)
    }


def _assert_worker_jobs_are_no_longer_active(job_ids: set[str]) -> None:
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--jobs",
            ",".join(sorted(job_ids)),
            "--format=%A",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
