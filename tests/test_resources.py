from furu import Spec, Worker
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.backends.protocol import slots_for
from furu.worker.backends.slurm.backend import SlurmWorkerBackend
from furu.worker.backends.slurm.resources import MemoryPerNode, SlurmResources


class Generate(Spec):
    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus in (1, 8) and "hopper" in worker.labels

    def create(self) -> str:
        return "x"


class Tokenize(Spec):
    def create(self) -> str:
        return "x"


class Train(Spec):
    gpus: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= self.gpus

    def create(self) -> str:
        return "x"


def _pool(worker: Worker) -> LocalThreadWorkerBackend:
    return LocalThreadWorkerBackend(worker=worker)


def _node(slots: int) -> SlurmWorkerBackend:
    """A whole node: 8 GPUs, 64 CPUs and 512 GiB."""
    return SlurmWorkerBackend(
        max_allocations=1,
        resources=SlurmResources(cpus=64, gpus=8, memory=MemoryPerNode(512)),
        labels=("hopper",),
        slots=slots,
    )


def test_runs_on_sees_the_pool_worker() -> None:
    assert slots_for(_pool(Worker(gpus=1, labels=("hopper",))), Generate()) == 1
    assert slots_for(_pool(Worker(gpus=8, labels=("hopper", "ib"))), Generate()) == 1
    assert slots_for(_pool(Worker(gpus=4, labels=("hopper",))), Generate()) is None
    assert slots_for(_pool(Worker(gpus=8, labels=("ampere",))), Generate()) is None


def test_specs_run_anywhere_by_default() -> None:
    assert slots_for(_pool(Worker()), Tokenize()) == 1
    assert slots_for(_pool(Worker(gpus=8, labels=("hopper",))), Tokenize()) == 1


def test_accepts_reserves_a_pool_for_chosen_specs() -> None:
    cpu_pool = LocalThreadWorkerBackend(
        worker=Worker(cpus=64), accepts=lambda spec: isinstance(spec, Tokenize)
    )

    assert slots_for(cpu_pool, Tokenize()) == 1
    assert slots_for(cpu_pool, Generate()) is None


def test_accepts_and_runs_on_must_both_hold() -> None:
    gpu_pool = LocalThreadWorkerBackend(
        worker=Worker(gpus=8, labels=("hopper",)),
        accepts=lambda spec: not isinstance(spec, Tokenize),
    )

    assert slots_for(gpu_pool, Generate()) == 1
    assert slots_for(gpu_pool, Tokenize()) is None


def test_a_share_is_its_slots_part_of_the_whole_rounded_down() -> None:
    whole = Worker(cpus=10, gpus=8, memory_gib=100, labels=("hopper",))

    assert whole.share(1, 8) == Worker(
        cpus=1, gpus=1, memory_gib=12, labels=("hopper",)
    )
    assert whole.share(3, 8) == Worker(
        cpus=3, gpus=3, memory_gib=37, labels=("hopper",)
    )
    assert whole.share(8, 8) == whole
    # The shares of slots that run at once never add up to more than the whole.
    for k in range(1, 8):
        rest = whole.share(8 - k, 8)
        assert whole.share(k, 8).cpus + rest.cpus <= whole.cpus
        assert whole.share(k, 8).memory_gib + rest.memory_gib <= whole.memory_gib


def test_a_spec_takes_the_fewest_slots_it_runs_on() -> None:
    eight = _node(slots=8)

    assert [slots_for(eight, Train(gpus=gpus)) for gpus in (0, 1, 2, 3, 8, 9)] == [
        1,
        1,
        2,
        3,
        8,
        None,
    ]
    # One slot is the whole node.
    assert slots_for(_node(slots=1), Train(gpus=1)) == 1
    # Labels hold on every share.
    assert slots_for(eight, Generate()) == 1
