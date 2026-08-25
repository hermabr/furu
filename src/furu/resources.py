from dataclasses import dataclass, field, fields

from pydantic import TypeAdapter

from furu.spec_metadata import Between, GiB, Requires


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    """What a worker brings to a lease: its capacity and, optionally, a floor.

    ``reserve_for`` keeps a big worker from wasting itself on jobs a smaller
    pool could run: the worker only takes jobs whose declared ``requires``
    demand at least that much in every dimension ``reserve_for`` constrains.
    A job that leaves a reserved dimension unconstrained never qualifies, so
    ``reserve_for=Requires(ram=GiB(200))`` leases only jobs that themselves
    ask for 200 GiB or more.
    """

    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0
    reserve_for: Requires = field(default_factory=Requires)


resource_request_adapter: TypeAdapter[ResourceRequest] = TypeAdapter(ResourceRequest)


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


def _minimum(constraint: int | GiB | Between[int] | Between[GiB] | None) -> int | None:
    """The least a constraint demands, or None when it does not constrain."""
    match constraint:
        case None:
            return None
        case Between(low=low):
            return _minimum(low)
        case GiB(count=count):
            return count
        case int():
            return constraint


def _demands_at_least(requires: Requires | None, floor: Requires) -> bool:
    for f in fields(Requires):
        threshold = _minimum(getattr(floor, f.name))
        if threshold is None:
            continue
        declared = None if requires is None else _minimum(getattr(requires, f.name))
        if declared is None or declared < threshold:
            return False
    return True


def resource_request_satisfies(
    request: ResourceRequest, requires: Requires | None
) -> bool:
    if not _demands_at_least(requires, request.reserve_for):
        return False
    if requires is None:
        return True
    return (
        _matches(request.cpus, requires.cpus)
        and _matches(request.gpus, requires.gpus)
        and _memory_matches(request.memory_gib, requires.ram)
    )
