from typing import Any

import pytest

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


def _slurm(slots: int, **overrides: Any) -> SlurmWorkerBackend:
    fields: dict[str, Any] = {
        "max_allocations": 1,
        "resources": SlurmResources(cpus=64, gpus=8, memory=MemoryPerNode(512)),
        "slots": slots,
    }
    return SlurmWorkerBackend(**{**fields, **overrides})


def test_a_job_is_offered_its_part_of_the_allocation_for_each_number_of_slots() -> None:
    assert _slurm(8, labels=("hopper",)).workers == {
        1: Worker(cpus=8, gpus=1, memory_gib=64, labels=("hopper",)),
        2: Worker(cpus=16, gpus=2, memory_gib=128, labels=("hopper",)),
        4: Worker(cpus=32, gpus=4, memory_gib=256, labels=("hopper",)),
        8: Worker(cpus=64, gpus=8, memory_gib=512, labels=("hopper",)),
    }
    # Only sizes that fill an allocation with nothing left over.
    assert list(_slurm(6, resources=SlurmResources(cpus=6)).workers) == [1, 2, 3, 6]
    assert _slurm(1).workers == {1: Worker(cpus=64, gpus=8, memory_gib=512)}


def test_parts_round_down_and_the_whole_allocation_is_exact() -> None:
    workers = _slurm(
        4, resources=SlurmResources(cpus=10, memory=MemoryPerNode(7))
    ).workers

    assert {slots: (w.cpus, w.memory_gib) for slots, w in workers.items()} == {
        1: (2, 1),
        2: (5, 3),
        4: (10, 7),
    }
    # However the slots are shared out, no more is offered than there is.
    assert all(4 // slots * w.cpus <= 10 for slots, w in workers.items())


class NeedsGpus(Spec):
    gpus: int

    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus >= self.gpus

    def create(self) -> str:
        return "x"


def test_a_spec_takes_the_fewest_slots_that_run_it() -> None:
    backend = _slurm(8)

    assert slice_for(backend, Tokenize()) == (1, backend.workers[1])
    assert slice_for(backend, NeedsGpus(gpus=2)) == (2, backend.workers[2])
    # Three GPUs are not a part of eight slots: the next size up is four.
    assert slice_for(backend, NeedsGpus(gpus=3)) == (4, backend.workers[4])
    assert slice_for(backend, NeedsGpus(gpus=8)) == (8, backend.workers[8])
    assert slice_for(backend, NeedsGpus(gpus=9)) is None
    assert slice_for(_slurm(8, accepts=lambda spec: False), Tokenize()) is None


def test_slots_must_cut_one_node_into_even_parts() -> None:
    with pytest.raises(ValueError, match="slots must be at least 1"):
        _slurm(0)
    with pytest.raises(ValueError, match="8 GPUs do not divide evenly into 3 slots"):
        _slurm(3)
    with pytest.raises(ValueError, match="one node into parts.*2 nodes"):
        _slurm(2, resources=SlurmResources(cpus=8, nodes=2))
    with pytest.raises(ValueError, match="2 CPUs leave none for each of 4 slots"):
        _slurm(4, resources=SlurmResources(cpus=2))
    # Without slots to cut, an allocation may span nodes.
    assert _slurm(1, resources=SlurmResources(cpus=8, nodes=2)).slots == 1
