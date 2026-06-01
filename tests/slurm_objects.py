from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from furu import Furu
from furu.resources import ResourceRequirements

type SlurmTaskKind = Literal["a_only", "b_only", "shared"]


class SlurmWorkloadTask(Furu[dict[str, object]]):
    task_id: str
    kind: SlurmTaskKind
    scenario_id: str
    duration_seconds: float
    parents: tuple[Furu[Any], ...] = ()

    @property
    def resource_requirements(self) -> ResourceRequirements | None:
        match self.kind:
            case "a_only":
                return ResourceRequirements(cpus=(1, 1))
            case "b_only":
                return ResourceRequirements(cpus=(2, 2))
            case "shared":
                return ResourceRequirements(cpus=(1, 2))

    def create(self) -> dict[str, object]:
        parent_results = [
            cast(dict[str, object], parent.create()) for parent in self.parents
        ]
        time.sleep(self.duration_seconds)

        return {
            "cwd": str(Path.cwd().resolve()),
            "duration_seconds": self.duration_seconds,
            "kind": self.kind,
            "parent_task_ids": [
                str(parent_result["task_id"]) for parent_result in parent_results
            ],
            "scenario_id": self.scenario_id,
            "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_job_name": os.environ["SLURM_JOB_NAME"],
            "task_id": self.task_id,
        }
