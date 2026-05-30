from __future__ import annotations

import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from furu.execution.manager import Manager
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import SlurmResources
from slurm_objects import SlurmTaskKind, SlurmWorkloadTask


pytestmark = [
    pytest.mark.slurm_integration,
    pytest.mark.skipif(
        os.environ.get("FURU_SLURM_INTEGRATION") != "1",
        reason="set FURU_SLURM_INTEGRATION=1 inside the Slurm test container",
    ),
]


def test_slurm_backend_runs_worker_job_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
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

    all_tasks, final_tasks = _build_workload()
    manager = Manager(final_tasks)

    manager.run(
        worker_backends=(
            SlurmWorkerBackend(
                max_workers=2,
                resources=SlurmResources(
                    partition="debug",
                    cpus_per_worker=1,
                ),
                worker_connect_host="127.0.0.1",
                manager_listen_host="0.0.0.0",
                job_name="furu-worker-a",
                poll_interval=0.1,
                worker_idle_timeout=0.05,
            ),
            SlurmWorkerBackend(
                max_workers=3,
                resources=SlurmResources(
                    partition="debug",
                    cpus_per_worker=2,
                ),
                worker_connect_host="127.0.0.1",
                manager_listen_host="0.0.0.0",
                job_name="furu-worker-b",
                poll_interval=0.1,
                worker_idle_timeout=0.05,
            ),
        ),
    )

    results = {task.task_id: task.try_load() for task in all_tasks}

    assert len(all_tasks) == 52
    assert Counter(task.kind for task in all_tasks) == {
        "a_only": 16,
        "b_only": 20,
        "shared": 16,
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

    assert len(a_worker_jobs) > 2
    assert len(b_worker_jobs) > 3
    _assert_worker_jobs_are_no_longer_active(a_worker_jobs | b_worker_jobs)


def _build_workload() -> tuple[list[SlurmWorkloadTask], list[SlurmWorkloadTask]]:
    roots_a = [_task("a0", index, "a_only") for index in range(4)]
    roots_b = [_task("b0", index, "b_only") for index in range(4)]
    roots_shared = [_task("s0", index, "shared") for index in range(4)]

    a_gate = [
        _task(
            "ag", index, "a_only", roots_a[index], roots_b[index], roots_shared[index]
        )
        for index in range(4)
    ]

    stage1_a = [_task("a1", index, "a_only", a_gate[index]) for index in range(4)]
    stage1_b = [_task("b1", index, "b_only", a_gate[index]) for index in range(4)]
    stage1_shared = [_task("s1", index, "shared", a_gate[index]) for index in range(4)]

    b_gate = [
        _task(
            "bg",
            index,
            "b_only",
            stage1_a[index],
            stage1_b[index],
            stage1_shared[index],
        )
        for index in range(4)
    ]

    stage2_a = [_task("a2", index, "a_only", b_gate[index]) for index in range(4)]
    stage2_b = [
        _task(
            "b2",
            index,
            "b_only",
            b_gate[index % 4],
        )
        for index in range(8)
    ]
    stage2_shared = [_task("s2", index, "shared", b_gate[index]) for index in range(4)]

    final_shared = [
        _task(
            "sf",
            index,
            "shared",
            stage2_a[index],
            stage2_b[index],
            stage2_b[index + 4],
            stage2_shared[index],
        )
        for index in range(4)
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
        *final_shared,
    ]
    return all_tasks, final_shared


def _task(
    prefix: str,
    index: int,
    kind: SlurmTaskKind,
    *parents: SlurmWorkloadTask,
) -> SlurmWorkloadTask:
    return SlurmWorkloadTask(
        task_id=f"{prefix}-{index:02d}",
        kind=kind,
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
