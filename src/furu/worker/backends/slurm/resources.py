from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class MemoryPerNode:
    gib: int

    def to_sbatch_arg(self) -> str:
        return f"--mem={self.gib}G"


@dataclass(frozen=True, slots=True)
class MemoryPerCpu:
    gib: int

    def to_sbatch_arg(self) -> str:
        return f"--mem-per-cpu={self.gib}G"


@dataclass(frozen=True, slots=True)
class MemoryPerGpu:
    gib: int

    def to_sbatch_arg(self) -> str:
        return f"--mem-per-gpu={self.gib}G"


@dataclass(frozen=True, slots=True)
class SlurmResources:
    """What one allocation asks of Slurm: ``cpus`` and ``gpus`` are its totals."""

    cpus: int
    nodes: int = 1
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    time_limit: str | None = None
    memory: MemoryPerNode | MemoryPerCpu | MemoryPerGpu | None = None
    gpus: int = 0
    constraint: str | None = None
    extra_sbatch_args: tuple[str, ...] = ()

    @property
    def memory_gib(self) -> int:
        match self.memory:
            case None:
                return 0
            case MemoryPerNode(gib=gib):
                return gib
            case MemoryPerCpu(gib=gib):
                return gib * self.cpus
            case MemoryPerGpu(gib=gib):
                return gib * self.gpus

    def sliced(self, k: int, slots: int) -> SlurmResources:
        """``k`` of ``slots`` even slices of this allocation, rounded down.

        Rounding down never hands out more than there is: ``slots`` slices of
        one slot each fit, and ``k = slots`` is the whole allocation.
        """
        return replace(
            self,
            cpus=k * self.cpus // slots,
            gpus=k * self.gpus // slots,
            memory=None
            if self.memory is None
            else MemoryPerNode(k * self.memory_gib // slots),
        )

    def to_srun_args(self) -> list[str]:
        """What a step of one task asks for, to get exactly these resources.

        Every amount is said, also when it is the whole allocation: a step that
        names no memory or GPUs takes all of the job's, and the next step waits.
        """
        args = ["--exact", "--nodes=1", "--ntasks=1", f"--cpus-per-task={self.cpus}"]
        if self.memory is not None:
            args.append(f"--mem={self.memory_gib}G")
        if self.gpus > 0:
            args.append(f"--gpus={self.gpus}")
        return args

    def to_sbatch_args(self) -> list[str]:
        args: list[str] = []
        if self.account is not None:
            args.append(f"--account={self.account}")
        if self.partition is not None:
            args.append(f"--partition={self.partition}")
        if self.qos is not None:
            args.append(f"--qos={self.qos}")
        if self.time_limit is not None:
            args.append(f"--time={self.time_limit}")
        args.append(f"--nodes={self.nodes}")
        args.append(f"--cpus-per-task={self.cpus}")
        if self.memory is not None:
            args.append(self.memory.to_sbatch_arg())
        if self.gpus > 0:
            args.append(f"--gpus={self.gpus}")
        if self.constraint is not None:
            args.append(f"--constraint={self.constraint}")
        args.extend(self.extra_sbatch_args)
        return args
