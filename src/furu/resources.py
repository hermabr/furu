from dataclasses import dataclass, field

from pydantic import TypeAdapter

from furu.spec_metadata import Between, GiB, Requires


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    """What a worker offers when it leases a job.

    ``reserve_for`` keeps a large worker from being spent on jobs a smaller
    pool could run: the worker only leases jobs whose ``requires`` demand at
    least that much in every dimension ``reserve_for`` constrains. A job that
    leaves a reserved dimension unconstrained never qualifies, so
    ``reserve_for=Requires(ram=GiB(200))`` leases only jobs that themselves
    ask for 200 GiB or more.
    """

    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0
    reserve_for: Requires = field(default_factory=Requires)


resource_request_adapter: TypeAdapter[ResourceRequest] = TypeAdapter(ResourceRequest)

type _Constraint = int | GiB | Between[int] | Between[GiB] | None


def _minimum(constraint: _Constraint) -> int | None:
    """The least a constraint asks for, or None when it asks for nothing."""
    match constraint:
        case None:
            return None
        case Between(low=low):
            return _minimum(low)
        case GiB(count=count):
            return count
        case int():
            return constraint


def _demands_at_least(constraint: _Constraint, floor: _Constraint) -> bool:
    threshold = _minimum(floor)
    if threshold is None:
        return True
    demanded = _minimum(constraint)
    return demanded is not None and demanded >= threshold


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
        and _memory_matches(request.memory_gib, requires.ram)
        and _demands_at_least(requires.cpus, floor.cpus)
        and _demands_at_least(requires.gpus, floor.gpus)
        and _demands_at_least(requires.ram, floor.ram)
    )
