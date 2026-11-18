from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from furu.backends.slurm.resources import SlurmResources
from furu.protocol.messages import SlurmPool
from furu.resources import Worker
from furu.utils import _hash_dict_deterministically

if TYPE_CHECKING:
    from furu.core import Spec

type SlurmExport = Literal["NIL", "ALL"] | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class SlurmWorkerBackend:
    """Slurm workers started by the furu manager; same recipe, same pool."""

    max_workers: int
    resources: SlurmResources
    job_name: str = "furu-worker"
    idle_timeout_seconds: float = 60.0
    pre_worker_commands: tuple[str, ...] = ()
    export: SlurmExport = None
    use_job_arrays: bool = True
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
        recipe = self.pool().model_dump(
            exclude={"max_workers", "job_name", "idle_timeout_seconds"}
        )
        return "slurm:" + _hash_dict_deterministically(
            {**recipe, "labels": list(self.labels)}
        )

    def pool(self) -> SlurmPool:
        export = (
            self.export if isinstance(self.export, str) else ",".join(self.export or ())
        )
        export_args = [f"--export={export}"] if export else []
        return SlurmPool(
            max_workers=self.max_workers,
            sbatch_args=[*self.resources.to_sbatch_args(), *export_args],
            pre_worker_commands=list(self.pre_worker_commands),
            use_job_arrays=self.use_job_arrays,
            job_name=self.job_name,
            idle_timeout_seconds=self.idle_timeout_seconds,
        )
