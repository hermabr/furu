from dataclasses import dataclass

type ResourceConstraint = tuple[int | None, int | None] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: ResourceConstraint = None
    memory: ResourceConstraint = None
    gpus: ResourceConstraint = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    memory: int
    cpus: int = 1
    gpus: int = 0
