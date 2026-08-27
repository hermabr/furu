from dataclasses import dataclass, field

from pydantic import TypeAdapter

from furu.spec_metadata import Between, GiB, Requires


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceFloor:
    """Minimums a job must demand before a reserved worker leases it."""

    cpus: int | None = None
    gpus: int | None = None
    memory_gib: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    """A worker's capacity and the jobs it is reserved for.

    ``reserve_for`` keeps a worker for jobs whose declared ``requires`` meet
    its lower bounds. Jobs that omit a reserved dimension do not qualify.
    """

    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0
    reserve_for: ResourceFloor = field(default_factory=ResourceFloor)


resource_request_adapter: TypeAdapter[ResourceRequest] = TypeAdapter(ResourceRequest)

type _Constraint = int | GiB | Between[int] | Between[GiB] | None


def _minimum(constraint: _Constraint) -> int | None:
    match constraint:
        case None:
            return None
        case Between(low=low):
            return _minimum(low)
        case GiB(count=count):
            return count
        case int():
            return constraint


def _demands_at_least(constraint: _Constraint, floor: int | None) -> bool:
    if floor is None:
        return True
    demanded = _minimum(constraint)
    return demanded is not None and demanded >= floor


def _matches(value: int, constraint: int | Between[int] | None) -> bool:
    if constraint is None:
        return True
    if isinstance(constraint, Between):
        return constraint.low <= value and (
            constraint.high is None or value <= constraint.high
        )
    return value == constraint


def _memory_matches(value: int, constraint: GiB | Between[GiB] | None) -> bool:
    if constraint is None:
        return True
    if isinstance(constraint, Between):
        return constraint.low.count <= value and (
            constraint.high is None or value <= constraint.high.count
        )
    return value >= constraint.count


def resource_request_satisfies(request: ResourceRequest, requires: Requires) -> bool:
    floor = request.reserve_for
    return (
        _matches(request.cpus, requires.cpus)
        and _matches(request.gpus, requires.gpus)
        and _memory_matches(request.memory_gib, requires.memory)
        and _demands_at_least(requires.cpus, floor.cpus)
        and _demands_at_least(requires.gpus, floor.gpus)
        and _demands_at_least(requires.memory, floor.memory_gib)
    )
