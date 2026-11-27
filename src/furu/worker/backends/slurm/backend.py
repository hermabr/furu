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
    """Slurm allocations started by the furu manager; same recipe, same pool.

    An allocation is ``slots`` even shares of ``resources``, and each job takes
    as few of them as its spec's ``runs_on`` accepts. One whole node with
    ``slots=8`` runs one job on all 8 shares, eight on one each, or any mix
    that adds up to 8, from one stage to the next without queueing again.
    """

    max_allocations: int
    resources: SlurmResources
    slots: int = 1
    job_name: str = "furu-worker"
    idle_timeout_seconds: float = 60.0
    pre_worker_commands: tuple[str, ...] = ()
    export: SlurmExport = None
    use_job_arrays: bool = True
    labels: tuple[str, ...] = ()
    accepts: Callable[[Spec], bool] | None = None

    def __post_init__(self) -> None:
        resources = self.resources
        if self.slots < 1 or self.slots > resources.cpus:
            raise ValueError(
                f"slots={self.slots} needs 1 <= slots <= cpus={resources.cpus}: "
                "each slot gets at least one CPU"
            )
        if self.slots > 1 and resources.nodes != 1:
            raise ValueError(
                f"slots={self.slots} splits one node, not {resources.nodes}"
            )
        if resources.gpus % self.slots:
            raise ValueError(
                f"gpus={resources.gpus} does not split evenly into slots={self.slots}"
            )

    @property
    def worker(self) -> Worker:
        return Worker(
            cpus=self.resources.cpus,
            gpus=self.resources.gpus,
            memory_gib=self.resources.memory_gib,
            labels=self.labels,
        )

    @property
    def pool_key(self) -> str:
        recipe = self.pool().model_dump(
            exclude={"max_allocations", "job_name", "idle_timeout_seconds"}
        )
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
            max_allocations=self.max_allocations,
            slots=self.slots,
            sbatch_args=[*self.resources.to_sbatch_args(), *export_args],
            pre_worker_commands=list(self.pre_worker_commands),
            use_job_arrays=self.use_job_arrays,
            job_name=self.job_name,
            idle_timeout_seconds=self.idle_timeout_seconds,
        )
