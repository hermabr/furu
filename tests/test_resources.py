from furu import Spec, Worker
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.backends.protocol import WorkerBackend, slice_for
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


def _pool(worker: Worker) -> LocalThreadWorkerBackend:
    return LocalThreadWorkerBackend(worker=worker)


def can_run(backend: WorkerBackend, spec: Spec) -> bool:
    return slice_for(backend, spec) is not None


def test_runs_on_sees_the_pool_worker() -> None:
    assert can_run(_pool(Worker(gpus=1, labels=("hopper",))), Generate())
    assert can_run(_pool(Worker(gpus=8, labels=("hopper", "ib"))), Generate())
    assert not can_run(_pool(Worker(gpus=4, labels=("hopper",))), Generate())
    assert not can_run(_pool(Worker(gpus=8, labels=("ampere",))), Generate())


def test_specs_run_anywhere_by_default() -> None:
    assert can_run(_pool(Worker()), Tokenize())
    assert can_run(_pool(Worker(gpus=8, labels=("hopper",))), Tokenize())


def test_accepts_reserves_a_pool_for_chosen_specs() -> None:
    cpu_pool = LocalThreadWorkerBackend(
        worker=Worker(cpus=64), accepts=lambda spec: isinstance(spec, Tokenize)
    )

    assert can_run(cpu_pool, Tokenize())
    assert not can_run(cpu_pool, Generate())


def test_accepts_and_runs_on_must_both_hold() -> None:
    gpu_pool = LocalThreadWorkerBackend(
        worker=Worker(gpus=8, labels=("hopper",)),
        accepts=lambda spec: not isinstance(spec, Tokenize),
    )

    assert can_run(gpu_pool, Generate())
    assert not can_run(gpu_pool, Tokenize())


class Train(Spec):
    gpus: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= self.gpus

    def create(self) -> str:
        return "x"


NODE = SlurmWorkerBackend(
    max_allocations=1,
    resources=SlurmResources(cpus=64, gpus=8, memory=MemoryPerNode(512)),
    slots=8,
    labels=("hopper",),
)


def test_a_spec_takes_the_fewest_slots_it_runs_on() -> None:
    assert [slice_for(NODE, Train(gpus=gpus)) for gpus in (0, 1, 2, 3, 8)] == [
        (1, NODE.slice(1)),
        (1, NODE.slice(1)),
        (2, NODE.slice(2)),
        (4, NODE.slice(4)),
        (8, NODE.slice(8)),
    ]
    assert slice_for(NODE, Train(gpus=9)) is None
    # Runs on the 1-GPU slice, which also has the label.
    assert slice_for(NODE, Generate()) == (1, NODE.slice(1))
    # A pool without slots has one slice: the whole worker.
    assert slice_for(_pool(Worker(gpus=8)), Train(gpus=2)) == (1, Worker(gpus=8))
