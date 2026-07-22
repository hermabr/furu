from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal, cast

import furu
from furu import Metadata, Requires, Spec, between

type SlurmTaskKind = Literal["a_only", "b_only", "shared"]


class SlurmWorkloadTask(Spec[dict[str, object]]):
    task_id: str
    kind: SlurmTaskKind
    scenario_id: str
    duration_seconds: float
    parents: tuple[Spec[Any], ...] = ()

    def metadata(self) -> Metadata:
        match self.kind:
            case "a_only":
                requires = Requires(cpus=1)
            case "b_only":
                requires = Requires(cpus=2)
            case "shared":
                requires = Requires(cpus=between(1, 2))
        return Metadata(requires=requires)

    def create(self) -> dict[str, object]:
        parent_results = [
            cast(dict[str, object], furu.create(parent)) for parent in self.parents
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
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_job_name": os.environ["SLURM_JOB_NAME"],
            "task_id": self.task_id,
        }
