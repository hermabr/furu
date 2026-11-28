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

    ``resources`` is what one allocation asks Slurm for. It is cut into
    ``slots`` even parts, and a job takes as many of them as its spec needs,
    so one allocation runs several jobs at once: with 8 GPUs and ``slots=8``,
    eight jobs on one GPU each, or one on four beside four on one, with
    nothing queued again in between. A job that takes ``k`` slots is offered
    ``k * total // slots`` of each resource.

    Each job of an allocation that is cut runs in a step of it (``srun
    --exact``), and Slurm keeps the job and all it starts to its share: it
    sees only its GPUs, runs only on its CPUs and is killed when it uses more
    than its memory. Where two CPUs share a core a step takes whole cores,
    so ask for at least ``slots`` cores. With ``slots=1`` a job has the
    allocation as it is.
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
        if self.slots < 1:
            raise ValueError(f"slots must be at least 1, got {self.slots}")
        if self.slots > 1 and self.resources.nodes != 1:
            raise ValueError(
                f"slots={self.slots} cuts one node into parts, and resources "
                f"asks for {self.resources.nodes} nodes"
            )
        if self.resources.gpus % self.slots:
            raise ValueError(
                f"{self.resources.gpus} GPUs do not divide evenly into "
                f"{self.slots} slots"
            )
        if self.resources.cpus < self.slots:
            raise ValueError(
                f"{self.resources.cpus} CPUs leave none for each of {self.slots} slots"
            )

    @property
    def workers(self) -> dict[int, Worker]:
        """What a job gets for each number of slots it may take: a divisor of
        ``slots``, so that jobs of any mix of sizes fill an allocation."""
        resources = self.resources
        return {
            taken: Worker(
                cpus=taken * resources.cpus // self.slots,
                gpus=taken * resources.gpus // self.slots,
                memory_gib=taken * resources.memory_gib // self.slots,
                labels=self.labels,
            )
            for taken in range(1, self.slots + 1)
            if self.slots % taken == 0
        }

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
