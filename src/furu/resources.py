from dataclasses import dataclass
from typing import TypeAlias

ResourceConstraint: TypeAlias = tuple[int | None, int | None] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: ResourceConstraint = None
    gpus: ResourceConstraint = None
    memory_gb: ResourceConstraint = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    cpus: int = 1
    gpus: int = 0
    memory_gb: int = 0


def resource_request_satisfies(
    request: ResourceRequest, requirements: ResourceRequirements | None
) -> bool:
    if requirements is None:
        return True

    def _matches(value: int, constraint: ResourceConstraint) -> bool:
        if constraint is None:
            return True
        lo, hi = constraint
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    return (
        _matches(request.cpus, requirements.cpus)
        and _matches(request.gpus, requirements.gpus)
        and _matches(request.memory_gb, requirements.memory_gb)
    )
