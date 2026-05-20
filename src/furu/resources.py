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

    def _matches(value: int, constraint: ResourceConstraint) -> bool:
        if constraint is None:
            return True
        lo, hi = constraint
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    return (
        _matches(request.cpus, requirements.cpus)
        and _matches(request.gpus, requirements.gpus)
        and _matches(request.memory, requirements.memory)
    )
