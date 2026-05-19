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


def resource_request_satisfies(
    request: ResourceRequest, requirements: ResourceRequirements | None
) -> bool:
    if requirements is None:
        return True
    return (
        _matches_constraint(request.cpus, requirements.cpus)
        and _matches_constraint(request.gpus, requirements.gpus)
        and _matches_constraint(request.memory, requirements.memory)
    )


def _matches_constraint(value: int, constraint: ResourceConstraint) -> bool:
    if constraint is None:
        return True
    lo, hi = constraint
    return (lo is None or value >= lo) and (hi is None or value <= hi)
