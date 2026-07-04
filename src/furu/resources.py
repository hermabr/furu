from dataclasses import dataclass

from furu.spec_metadata import Between, GiB, Requires


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRequest:
    cpus: int = 1
    gpus: int = 0
    memory_gib: int = 0


def resource_request_satisfies(
    request: ResourceRequest, requires: Requires | None
) -> bool:
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
