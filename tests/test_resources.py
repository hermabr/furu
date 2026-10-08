from furu import Spec, Worker
from furu.worker.backends.local import LocalThreadWorkerBackend
from furu.worker.backends.protocol import can_run


class Generate(Spec[str]):
    def runs_on(self, worker: Worker) -> bool:
        return worker.gpus in (1, 8) and "hopper" in worker.labels

    def create(self) -> str:
        return "x"


class Tokenize(Spec[str]):
    def create(self) -> str:
        return "x"


def _pool(worker: Worker) -> LocalThreadWorkerBackend:
    return LocalThreadWorkerBackend(worker=worker)


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
