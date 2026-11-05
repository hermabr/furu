"""A create() that finds missing specs is aborted and rerun once they're built."""

import shutil

import pytest

import furu
from furu import Spec
from furu.storage._layout import run_log_path_in

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
        leaf = Leaf(n=7)
        shutil.rmtree(leaf._base_dir, ignore_errors=True)
        return leaf.create()


class AddsOne(Spec[int]):
    leaf: Leaf

    def create(self) -> int:
        return self.leaf.create() + 1


class LazilyAddsOne(Spec[int]):
    n: int

    def create(self) -> int:
        return AddsOne(leaf=Leaf(n=self.n)).create()


class BatchedParent(furu.Spec[int]):
    key: int

    def batch_key(self, worker: furu.Worker) -> tuple[None, int]:
        return (None, 1024)

    @furu.batched(batch_key)
    def create(objs: list["BatchedParent"]) -> list[int]:
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
    parent_log = run_log_path_in(parent._base_dir).read_text(encoding="utf-8")
    assert parent_log.rstrip("\n").endswith(" · 1 missing dependency")
    assert "failed" not in parent_log
    # It only blocked, so its leftover attempt/ reads as incomplete, not failed.
    assert parent.status == "incomplete"


def test_dependency_cycle_raises_instead_of_looping() -> None:
    a, b = CycleA(), CycleB()

    with pytest.raises(RuntimeError) as exc_info:
        a.create()

    assert str(exc_info.value) == (
        f"dependency cycle: {a._log_label} → {b._log_label} → {a._log_label}"
    )


def test_dependency_still_missing_after_it_was_built_raises() -> None:
    parent = LosesItsLeaf()

    with pytest.raises(RuntimeError) as exc_info:
        parent.create()

    assert str(exc_info.value) == (
        f"{parent._log_label} still finds {Leaf(n=7)._log_label} missing after "
        "building it"
    )


def test_dependency_built_elsewhere_but_still_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furu.execution.load_or_create as load_or_create_module
    from furu.dag import Dag
    from furu.dependencies import _DependencyNotReady

    parent, leaf = SumOfLeaves(count=1), Leaf(n=0)
    leaf.create()  # done on disk, so never joins the graph
    dag = Dag()
    dag.add([parent])
    attempts: list[object] = []

    def blocked(group: object, **kwargs: object) -> None:
        attempts.append(group)
        raise _DependencyNotReady([leaf])

    monkeypatch.setattr(load_or_create_module, "_create_in_process", blocked)
    with pytest.raises(RuntimeError) as exc_info:
        load_or_create_module._run_in_process(dag)

    assert len(attempts) == 2  # one rerun, in case it was built just then
    assert str(exc_info.value) == (
        f"{parent._log_label} still finds {leaf._log_label} missing after it was built"
    )


def test_spec_found_after_its_dependency_was_built_here_still_runs() -> None:
    # Leaf builds first, then the lazy parent finds AddsOne, which declares it.
    assert furu.create([Leaf(n=3), LazilyAddsOne(n=3)]) == [3, 4]


def test_dependencies_built_elsewhere_one_after_another_do_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furu.execution.load_or_create as load_or_create_module
    from furu.dag import Dag
    from furu.dependencies import _DependencyNotReady

    parent, first, second = SumOfLeaves(count=2), Leaf(n=0), Leaf(n=1)
    first.create()
    second.create()
    dag = Dag()
    dag.add([parent])
    attempts: list[object] = []

    def create(group: object, **kwargs: object) -> None:
        attempts.append(group)
        # Each one is published just as the parent looks for it.
        if len(attempts) <= 2:
            raise _DependencyNotReady([(first, second)[len(attempts) - 1]])

    monkeypatch.setattr(load_or_create_module, "_create_in_process", create)
    load_or_create_module._run_in_process(dag)

    assert len(attempts) == 3
    assert set(dag.completed) == {parent.object_id}


@pytest.mark.parametrize(
    "blocks", [[(0,), (1,), (0,)], [(0,), (0, 1)]], ids=["alternating", "overlapping"]
)
def test_dependency_built_elsewhere_and_named_again_raises(
    monkeypatch: pytest.MonkeyPatch, blocks: list[tuple[int, ...]]
) -> None:
    import furu.execution.load_or_create as load_or_create_module
    from furu.dag import Dag
    from furu.dependencies import _DependencyNotReady

    parent, leaves = SumOfLeaves(count=2), [Leaf(n=0), Leaf(n=1)]
    furu.create(leaves)
    dag = Dag()
    dag.add([parent])
    attempts: list[object] = []

    def create(group: object, **kwargs: object) -> None:
        attempts.append(group)
        raise _DependencyNotReady([leaves[n] for n in blocks[len(attempts) - 1]])

    monkeypatch.setattr(load_or_create_module, "_create_in_process", create)
    with pytest.raises(RuntimeError) as exc_info:
        load_or_create_module._run_in_process(dag)

    assert len(attempts) == len(blocks)
    assert str(exc_info.value) == (
        f"{parent._log_label} still finds {leaves[0]._log_label} missing after it "
        "was built"
    )


def test_batched_group_reruns_as_a_whole() -> None:
    objs = [BatchedParent(key=1), BatchedParent(key=2)]

    assert furu.create(objs) == [1, 2]
    assert CALLS == ["batch [1, 2]", "leaf 1", "leaf 2", "batch [1, 2]"]
    for obj in objs:
        run_log = run_log_path_in(obj._base_dir).read_text(encoding="utf-8")
        assert " · 2 missing dependencies\n" in run_log
