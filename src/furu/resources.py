from dataclasses import dataclass, fields

from pydantic import TypeAdapter

from furu.spec_metadata import Between, GiB, Requires


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    """A worker's capacity and the jobs it is reserved for.

    ``reserve_for`` is a lower bound on what a job must declare in
    ``Metadata.requires``. A job that does not declare a reserved dimension
    does not qualify for the worker.
    """

    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0
    reserve_for: Requires | None = None


resource_request_adapter: TypeAdapter[ResourceRequest] = TypeAdapter(ResourceRequest)


def _minimum(constraint: int | GiB | Between[int] | Between[GiB] | None) -> int | None:
    match constraint:
        case None:
            return None
        case Between(low=low):
            return _minimum(low)
        case GiB(count=count):
            return count
        case int():
            return constraint


def _demands_at_least(requires: Requires | None, reserve_for: Requires) -> bool:
    for resource in fields(Requires):
        reserved_minimum = _minimum(getattr(reserve_for, resource.name))
        if reserved_minimum is None:
            continue
        required_minimum = (
            None if requires is None else _minimum(getattr(requires, resource.name))
        )
        if required_minimum is None or required_minimum < reserved_minimum:
            return False
    return True


def resource_request_satisfies(
    request: ResourceRequest, requires: Requires | None
) -> bool:
    if request.reserve_for is not None and not _demands_at_least(
        requires, request.reserve_for
    ):
        return False
    if requires is None:
        return True

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

    return (
        _matches(request.cpus, requires.cpus)
        and _matches(request.gpus, requires.gpus)
        and _memory_matches(request.memory_gib, requires.ram)
    )
