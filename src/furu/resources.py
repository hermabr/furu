from dataclasses import dataclass, field
from typing import cast

from pydantic import TypeAdapter

from furu.spec_metadata import Between, GiB, Requires


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    """What a worker has, and which jobs it is reserved for.

    ``reserve_for`` is matched against a job's declared lower bounds with the
    same rules a job's ``requires`` is matched against a worker's resources,
    so ``reserve_for=Requires(ram=GiB(200))`` leases only jobs that themselves
    ask for at least 200 GiB.  A job that declares nothing for a reserved
    dimension never qualifies.
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


def _lower_bound[T](constraint: T | Between[T] | None) -> T | None:
    if isinstance(constraint, Between):
        return cast(T, constraint.low)
    return constraint


def resource_request_satisfies(
    request: ResourceRequest, requires: Requires | None
) -> bool:
    if requires is None:
        return True
    reserve = request.reserve_for
    job_cpus = _lower_bound(requires.cpus)
    job_gpus = _lower_bound(requires.gpus)
    job_ram = _lower_bound(requires.ram)
    return (
        _matches(request.cpus, requires.cpus)
        and _matches(request.gpus, requires.gpus)
        and _memory_matches(request.memory_gib, requires.ram)
        and (
            reserve.cpus is None
            or (job_cpus is not None and _matches(job_cpus, reserve.cpus))
        )
        and (
            reserve.gpus is None
            or (job_gpus is not None and _matches(job_gpus, reserve.gpus))
        )
        and (
            reserve.ram is None
            or (job_ram is not None and _memory_matches(job_ram.count, reserve.ram))
        )
    )
