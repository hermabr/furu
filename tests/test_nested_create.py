"""A create() that finds missing specs is aborted and rerun once they're built."""

import shutil

import pytest
from runs import ends_of, fixed_run

import furu
from furu import Spec, plan
from furu.dependencies import _DependencyNotReady
from furu.worker.backends.local import LocalThreadWorkerBackend

CALLS: list[str] = []


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


class Leaf(Spec[int]):
    n: int
    fail: bool = False

    def create(self) -> int:
        CALLS.append(f"leaf {self.n}")
        if self.fail:
            raise ValueError(f"leaf {self.n} broke")
        return self.n


class SumOfLeaves(Spec[int]):
    count: int

    def create(self) -> int:
        CALLS.append("parent")
        return sum(Leaf(n=n).create() for n in range(self.count))


class LoadsFailingLeaf(Spec[int]):
    def create(self) -> int:
        return Leaf(n=99, fail=True).create()


class CycleA(Spec[int]):
    def create(self) -> int:
        return CycleB().create()


class CycleB(Spec[int]):
    def create(self) -> int:
        return CycleA().create()


class LosesItsLeaf(Spec[int]):
    def create(self) -> int:
        CALLS.append("loses")
        leaf = Leaf(n=7)
        shutil.rmtree(fixed_run(leaf), ignore_errors=True)
        return leaf.create()


class CannotSeeItsLeaf(Spec[int]):
    """Its create() keeps finding Leaf(n=8) missing, like code reading elsewhere."""

    def create(self) -> int:
        CALLS.append("cannot see")
        if len(CALLS) > 20:
            raise AssertionError("reran without bound")
        raise _DependencyNotReady([Leaf(n=8)], plan.listing())


class MissesEachLeafOnce(Spec[int]):
    """Finds each leaf missing once, as if another run published it meanwhile."""

    count: int

    def create(self) -> int:
        misses = sum(call.startswith("misses") for call in CALLS)
        CALLS.append(f"misses {misses}")
        if misses < self.count:
            raise _DependencyNotReady([Leaf(n=misses)], plan.listing())
        return sum(Leaf(n=n).create() for n in range(self.count))


class AddsOne(Spec[int]):
    leaf: Leaf

    def create(self) -> int:
        return self.leaf.create() + 1


class LazyParent(Spec[int]):
    n: int

    def create(self) -> int:
        return AddsOne(leaf=Leaf(n=self.n)).create()


class BatchedParent(furu.Spec[int]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["BatchedParent"], directory: furu.SpecDirectory) -> list[int]:
        CALLS.append(f"batch {[obj.key for obj in objs]}")
        return furu.create([Leaf(n=obj.key) for obj in objs])


def test_parent_reruns_from_scratch_after_its_dependency_is_built() -> None:
    parent = SumOfLeaves(count=1)

    assert parent.create() == 0
    assert CALLS == ["parent", "leaf 0", "parent"]


def test_parent_reruns_once_per_dependency_discovered_in_a_loop() -> None:
    assert SumOfLeaves(count=3).create() == 3
    assert CALLS == [
        "parent",
        "leaf 0",
        "parent",
        "leaf 1",
        "parent",
        "leaf 2",
        "parent",
    ]


def test_existing_dependencies_load_without_rerunning_the_parent() -> None:
    furu.create([Leaf(n=0), Leaf(n=1)])
    CALLS.clear()

    assert SumOfLeaves(count=2).create() == 1
    assert CALLS == ["parent"]


def test_dependency_failure_propagates_and_names_the_parent() -> None:
    parent = LoadsFailingLeaf()

    with pytest.raises(ValueError, match="leaf 99 broke") as exc_info:
        parent.create()

    assert exc_info.value.__notes__ == [
        f"while building dependencies of {parent._log_label}"
    ]
    # It only blocked: the failure is its leaf's.
    assert ends_of(parent) == [
        {"outcome": "blocked", "waiting_on": [Leaf(n=99, fail=True).object_id]}
    ]


def test_dependency_cycle_raises_instead_of_looping() -> None:
    a, b = CycleA(), CycleB()

    with pytest.raises(RuntimeError) as exc_info:
        a.create()

    assert str(exc_info.value) == (
        f"dependency cycle: {b._log_label} and {a._log_label} wait for each other"
    )


def test_dependency_still_missing_after_it_was_built_raises() -> None:
    parent = LosesItsLeaf()

    with pytest.raises(RuntimeError) as exc_info:
        parent.create()

    assert str(exc_info.value) == (
        f"{parent._log_label} still finds {Leaf(n=7)._log_label} missing after a "
        "rerun, though this run sees it built"
    )
    # Built here, then one rerun in case the create raced a publish.
    assert CALLS == ["loses", "leaf 7", "loses", "loses"]


def test_dependency_another_run_built_but_create_cannot_see_raises() -> None:
    Leaf(n=8).create()  # another run built it
    parent = CannotSeeItsLeaf()

    with pytest.raises(RuntimeError, match="still finds .* missing after a rerun"):
        parent.create()
    # One rerun in case another run published it just after create() looked.
    assert CALLS == ["leaf 8", "cannot see", "cannot see"]


def test_dependencies_published_while_create_looks_rerun_it_each_time() -> None:
    parent = MissesEachLeafOnce(count=3)
    furu.create([Leaf(n=n) for n in range(3)])

    assert parent.create() == 3
    assert CALLS[-4:] == ["misses 0", "misses 1", "misses 2", "misses 3"]


def test_create_blocked_without_naming_a_dependency_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furu.execution.load_or_create as load_or_create_module

    def create(group: object, **kwargs: object) -> None:
        raise _DependencyNotReady([], plan.listing())

    monkeypatch.setattr(load_or_create_module, "_create_in_process", create)
    parent = SumOfLeaves(count=1)

    with pytest.raises(RuntimeError) as exc_info:
        parent.create()
    assert str(exc_info.value) == (
        f"{parent._log_label} reported itself blocked without naming a missing "
        "dependency"
    )


def test_create_misses_a_dependency_again_after_building_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furu.execution.load_or_create as load_or_create_module

    seen, missing = Leaf(n=0), Leaf(n=1)
    seen.create()
    real_create = load_or_create_module._create_in_process
    attempts: list[str] = []

    def create(group: list[Spec]) -> None:
        if group[0] != parent:
            return real_create(group)
        attempts.append("parent")
        # A race on seen, a real block on missing, then a race on seen again.
        if len(attempts) in (1, 3):
            raise _DependencyNotReady([seen], plan.listing())
        if len(attempts) == 2:
            raise _DependencyNotReady([missing], plan.listing())
        return real_create(group)

    monkeypatch.setattr(load_or_create_module, "_create_in_process", create)
    parent = SumOfLeaves(count=2)

    assert parent.create() == 1
    assert len(attempts) == 4


@pytest.mark.parametrize("on_workers", [False, True])
def test_spec_discovered_after_its_dependency_was_built_runs(on_workers: bool) -> None:
    on = [LocalThreadWorkerBackend()] if on_workers else None

    assert furu.create([Leaf(n=3), LazyParent(n=3)], on=on) == [3, 4]


def test_batched_group_reruns_as_a_whole() -> None:
    objs = [BatchedParent(key=1), BatchedParent(key=2)]

    assert furu.create(objs) == [1, 2]
    assert CALLS == ["batch [1, 2]", "leaf 1", "leaf 2", "batch [1, 2]"]
    waiting_on = [Leaf(n=obj.key).object_id for obj in objs]
    for obj in objs:
        assert ends_of(obj) == [
            {"outcome": "blocked", "waiting_on": waiting_on},
            {"outcome": "ok"},
        ]
