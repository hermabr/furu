from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from furu.resources import Worker
from furu.utils import _hash_dict_deterministically
from furu.worker.backends.slurm.resources import SlurmResources
from furu.worker.messages import SlurmPool

if TYPE_CHECKING:
    from furu.core import Spec

type SlurmExport = Literal["NIL", "ALL"] | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class SlurmWorkerBackend:
    """Slurm workers started by the furu manager; same recipe, same pool."""

    max_workers: int
    resources: SlurmResources
    job_name: str = "furu-worker"
    pre_worker_commands: tuple[str, ...] = ()
    export: SlurmExport = None
    labels: tuple[str, ...] = ()
    accepts: Callable[[Spec], bool] | None = None

    @property
    def worker(self) -> Worker:
        return Worker(
            cpus=self.resources.cpus_per_worker,
            gpus=self.resources.gpus,
            memory_gib=self.resources.memory_gib,
            labels=self.labels,
        )

    @property
    def pool_key(self) -> str:
        recipe = self.pool().model_dump(exclude={"max_workers"})
        return "slurm:" + _hash_dict_deterministically(
            {**recipe, "labels": list(self.labels)}
        )

    def pool(self) -> SlurmPool:
        match self.export:
            case None | ():
                export_args = []
            case "NIL" | "ALL":
                export_args = [f"--export={self.export}"]
            case (*names,):
                export_args = [f"--export={','.join(names)}"]
            case _:
                assert_never(self.export)
        return SlurmPool(
            max_workers=self.max_workers,
            sbatch_args=[
                f"--job-name={self.job_name}",
                *self.resources.to_sbatch_args(),
                *export_args,
            ],
            pre_worker_commands=list(self.pre_worker_commands),
        )
