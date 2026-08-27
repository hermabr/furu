import pytest

from furu import GiB, Requires, at_least, between
from furu.resources import ResourceRequest, resource_request_satisfies

RESERVED = ResourceRequest(
    cpus=8, gpus=2, memory_gib=600, reserve_for=Requires(ram=GiB(200), gpus=1)
)


@pytest.mark.parametrize(
    ("requires", "eligible"),
    [
        (Requires(ram=GiB(200), gpus=2), True),
        (Requires(ram=at_least(GiB(300)), gpus=at_least(1)), True),
        (Requires(ram=between(GiB(200), GiB(600)), gpus=between(1, 2)), True),
        # Every reserved dimension must be declared, and at or above the floor.
        (Requires(), False),
        (Requires(ram=GiB(200)), False),
        (Requires(gpus=2), False),
        (Requires(ram=GiB(199), gpus=2), False),
        (Requires(ram=between(GiB(100), GiB(600)), gpus=2), False),
        (Requires(ram=GiB(200), gpus=0), False),
        # The floor never loosens the worker's own capacity limits.
        (Requires(ram=GiB(601), gpus=2), False),
        (Requires(ram=GiB(200), gpus=3), False),
    ],
)
def test_reserve_for_matches_job_lower_bounds(
    requires: Requires, eligible: bool
) -> None:
    assert resource_request_satisfies(RESERVED, requires) is eligible


def test_reserve_for_floor_uses_lower_bound_of_range() -> None:
    request = ResourceRequest(
        memory_gib=600, reserve_for=Requires(ram=at_least(GiB(200)))
    )
    assert resource_request_satisfies(request, Requires(ram=GiB(500)))
    assert not resource_request_satisfies(request, Requires(ram=GiB(100)))


def test_unreserved_worker_accepts_undeclared_jobs() -> None:
    assert resource_request_satisfies(ResourceRequest(), Requires())
