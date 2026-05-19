from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryPerNode:
    value: str

    def to_sbatch_arg(self) -> str:
        return f"--mem={self.value}"


@dataclass(frozen=True, slots=True)
class MemoryPerCpu:
    value: str

    def to_sbatch_arg(self) -> str:
        return f"--mem-per-cpu={self.value}"


@dataclass(frozen=True, slots=True)
class MemoryPerGpu:
    value: str

    def to_sbatch_arg(self) -> str:
        return f"--mem-per-gpu={self.value}"


@dataclass(frozen=True, slots=True)
class Gpus:
    count: int
    kind: str | None = None

    def to_sbatch_arg(self) -> str:
        if self.kind is None:
            return f"--gpus={self.count}"
        return f"--gpus={self.kind}:{self.count}"


@dataclass(frozen=True, slots=True)
class SlurmResources:
    cpus_per_worker: int
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    time_limit: str | None = None
    memory: MemoryPerNode | MemoryPerCpu | MemoryPerGpu | None = None
    gpus: Gpus | int = 0
    constraint: str | None = None
    extra_sbatch_args: tuple[str, ...] = ()

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
        args.append(f"--cpus-per-task={self.cpus_per_worker}")
        if self.memory is not None:
            args.append(self.memory.to_sbatch_arg())
        if isinstance(self.gpus, Gpus):
            args.append(self.gpus.to_sbatch_arg())
        elif self.gpus > 0:
            args.append(f"--gpus={self.gpus}")
        if self.constraint is not None:
            args.append(f"--constraint={self.constraint}")
        args.extend(self.extra_sbatch_args)
        return args
