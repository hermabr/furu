from dataclasses import dataclass
from typing import TypeAlias

from furu.spec_metadata import Between, Requires

ResourceConstraint: TypeAlias = tuple[int | None, int | None] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequirements:
    cpus: ResourceConstraint = None
    gpus: ResourceConstraint = None
    memory_gib: ResourceConstraint = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0


def resource_requirements_from_requires(requires: Requires) -> ResourceRequirements:
    def _constraint(value: int | Between) -> tuple[int, int | None]:
        if isinstance(value, Between):
            return (value.low, value.high)
        return (value, value)

    return ResourceRequirements(
        cpus=_constraint(requires.cpus),
        gpus=_constraint(requires.gpus),
        memory_gib=None
        if requires.ram is None
        else (-(-requires.ram // 1024**3), None),
    )


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
        and _matches(request.memory_gib, requirements.memory_gib)
    )
