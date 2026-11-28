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

    Each allocation is split into ``slots`` even slices. A job takes the
    fewest slots whose slice its spec ``runs_on``, so one 8-GPU node with
    ``slots=8`` runs one 8-GPU job, four 2-GPU jobs or eight 1-GPU jobs at a
    time, as the queue asks. With more than one slot, each job's process runs
    in its own Slurm step, which holds only its slice. Slurm rounds a step's
    CPUs up to whole cores, so give each slot whole cores (an even number of
    CPUs with 2 threads a core), or steps wait for one another.
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
        resources, slots = self.resources, self.slots
        if slots < 1:
            raise ValueError(f"slots must be at least 1, got {slots}")
        if slots > 1 and (
            problems := [
                problem
                for problem, bad in (
                    (f"nodes=1, not {resources.nodes}", resources.nodes != 1),
                    (
                        f"gpus divisible by slots={slots}, not {resources.gpus}",
                        resources.gpus % slots != 0,
                    ),
                    (
                        f"at least slots={slots} cpus, not {resources.cpus}",
                        resources.cpus < slots,
                    ),
                    (
                        f"at least slots={slots} GiB of memory, for each step to name",
                        resources.memory is None or resources.memory_gib < slots,
                    ),
                )
                if bad
            ]
        ):
            raise ValueError(
                f"a Slurm backend with slots={slots} needs " + "; ".join(problems)
            )

    @property
    def worker(self) -> Worker:
        return self.slice(self.slots)

    def slice(self, k: int) -> Worker:
        """What ``k`` of the allocation's slots offer."""
        resources = self.resources.sliced(k, self.slots)
        return Worker(
            cpus=resources.cpus,
            gpus=resources.gpus,
            memory_gib=resources.memory_gib,
            labels=self.labels,
        )

    def slices(self) -> list[tuple[int, Worker]]:
        """Every slot count a job can take, fewest first, and what it offers."""
        return [(k, self.slice(k)) for k in _divisors(self.slots)]

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
            steps={}
            if self.slots == 1
            else {
                k: self.resources.sliced(k, self.slots).to_srun_args()
                for k in _divisors(self.slots)
            },
            sbatch_args=[*self.resources.to_sbatch_args(), *export_args],
            pre_worker_commands=list(self.pre_worker_commands),
            use_job_arrays=self.use_job_arrays,
            job_name=self.job_name,
            idle_timeout_seconds=self.idle_timeout_seconds,
        )


def _divisors(n: int) -> list[int]:
    return [k for k in range(1, n + 1) if n % k == 0]
