from furu import GiB, Requires, at_least, between
from furu.resources import ResourceRequest, resource_request_satisfies

RESERVED = ResourceRequest(
    cpus=8,
    gpus=2,
    memory_gib=600,
    reserve_for=Requires(ram=GiB(200), gpus=1),
)


def test_reserve_for_requires_every_reserved_dimension_to_be_declared() -> None:
    assert resource_request_satisfies(RESERVED, Requires(ram=GiB(200), gpus=2))
    assert resource_request_satisfies(
        RESERVED, Requires(ram=at_least(GiB(300)), gpus=at_least(1))
    )
    assert resource_request_satisfies(
        RESERVED, Requires(ram=between(GiB(200), GiB(600)), gpus=between(1, 2))
    )
    assert not resource_request_satisfies(RESERVED, Requires())
    assert not resource_request_satisfies(RESERVED, Requires(ram=GiB(200)))
    assert not resource_request_satisfies(RESERVED, Requires(gpus=2))
    assert not resource_request_satisfies(RESERVED, Requires(ram=GiB(199), gpus=2))
    assert not resource_request_satisfies(
        RESERVED, Requires(ram=between(GiB(100), GiB(600)), gpus=2)
    )
    assert not resource_request_satisfies(RESERVED, Requires(ram=GiB(200), gpus=0))


def test_reserve_for_still_enforces_worker_capacity() -> None:
    assert not resource_request_satisfies(RESERVED, Requires(ram=GiB(601), gpus=2))
    assert not resource_request_satisfies(RESERVED, Requires(ram=GiB(200), gpus=3))


def test_reserve_for_uses_the_lower_bound_of_ranges() -> None:
    request = ResourceRequest(
        memory_gib=600, reserve_for=Requires(ram=between(GiB(200), GiB(400)))
    )

    assert resource_request_satisfies(request, Requires(ram=GiB(500)))
    assert not resource_request_satisfies(request, Requires(ram=GiB(100)))


def test_unreserved_worker_accepts_unconstrained_jobs() -> None:
    assert resource_request_satisfies(ResourceRequest(), Requires())
