from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from runs import ends_of, log_of

import furu
from furu import Spec
from furu.config import get_config
from furu.dag import Dag
from furu.execution.launch import missing
from furu.execution.run_on_workers import run_on_workers
from furu.locking import lock
from furu.storage._layout import attempts_dir_in, lock_path_in
from furu.testing import override_config
from furu.worker.backends.local import LocalThreadWorkerBackend


class Leaf(Spec[str]):
    name: str

    def create(self) -> str:
        return f"leaf:{self.name}"


class Mid(Spec[str]):
    label: str
    child: Leaf

    def create(self) -> str:
        return f"mid:{self.label}:{self.child.create()}"


class Top(Spec[str]):
    name: str
    left: Mid
    right: Mid

    def create(self) -> str:
        return f"top:{self.name}"


@dataclass(frozen=True)
class LeafBundle:
    a: Leaf
    b: Leaf


class NestedParent(Spec[str]):
    bundle: LeafBundle

    def create(self) -> str:
        return self.bundle.a.create()


class ComputedParent(Spec[str]):
    name: str

    @furu.dependency()
    def computed_child(self) -> Leaf:
        return Leaf(name=f"computed-{self.name}")

    def create(self) -> str:
        return self.computed_child.create()


def _ids(objs: Sequence[Spec]) -> set[str]:
    return {obj.object_id for obj in objs}


# --- the graph, by id ----------------------------------------------------------


def test_a_spec_is_ready_once_the_specs_it_holds_are_finished() -> None:
    dag = Dag()
    dag.add({"top": ["left", "right", "stored"], "left": ["leaf"], "right": ["leaf"]})
    # "leaf" and "stored" are not among the keys: they are finished.
    assert list(dag.ready) == ["left", "right"]

    del dag.ready["left"], dag.ready["right"]
    dag.complete("left")
    assert not dag.ready and dag.waiting_for(["right"]) == ["top"]
    dag.complete("right")
    assert list(dag.ready) == ["top"]


def test_a_value_that_holds_specs_has_nothing_to_run_and_ends_with_them() -> None:
    dag = Dag()
    dag.add(
        {"a": ["shared"], "b": ["shared"], "shared": ["x", "y"], "x": [], "y": []},
        values={"shared"},
    )
    assert list(dag.ready) == ["x", "y"]

    dag.complete("x")
    assert dag.waits("a") and dag.waiting_for(["y"]) == ["a", "b"]
    dag.complete("y")
    assert list(dag.ready) == ["a", "b"] and "shared" not in dag.nodes


def test_a_blocked_spec_waits_for_what_it_found_missing() -> None:
    parent, found = "pkg.Parent:c:1", "pkg.Found:c:2"
    dag = Dag()
    dag.add({parent: []})
    del dag.ready[parent]
    dag.add({found: []})

    dag.block(parent, [found, "pkg.Stored:c:3"])
    assert list(dag.ready) == [found]
    del dag.ready[found]
    with pytest.raises(RuntimeError) as cycle:
        dag.block(found, [parent])
    assert (
        str(cycle.value)
        == "dependency cycle: Found:c:2 asks for Parent:c:1, which already waits for it"
    )
    dag.complete(found)
    assert list(dag.ready) == [parent]


# --- what a launch finds missing -----------------------------------------------


def test_missing_walks_down_through_fields_to_what_is_not_stored() -> None:
    leaf_a, leaf_b = Leaf(name="a"), Leaf(name="b")
    left, right = Mid(label="L", child=leaf_a), Mid(label="R", child=leaf_a)
    top = Top(name="t", left=left, right=right)
    dag = Dag()

    assert _ids(missing([top, leaf_b], dag)) == _ids([top, left, right, leaf_a, leaf_b])
    assert set(dag.ready) == _ids([leaf_a, leaf_b])
    # The shared leaf is one node that both wait for.
    assert set(dag.waiting_for([leaf_a.object_id])) == _ids([left, right, top])


def test_missing_stops_at_a_stored_spec() -> None:
    leaf = Leaf(name="cached")
    mid = Mid(label="m", child=leaf)
    leaf.create()
    dag = Dag()

    assert missing([mid], dag) == [mid]
    assert list(dag.ready) == [mid.object_id]

    mid.create()
    assert missing([Top(name="t", left=mid, right=mid), mid], Dag())[1:] == []


def test_missing_reaches_specs_in_dataclasses_and_ones_a_dependency_method_returns() -> (
    None
):
    nested = NestedParent(bundle=LeafBundle(a=Leaf(name="a"), b=Leaf(name="b")))
    computed = ComputedParent(name="p")

    assert _ids(missing([nested, computed], Dag())) == _ids(
        [nested, nested.bundle.a, nested.bundle.b, computed, computed.computed_child]
    )


@pytest.mark.parametrize("held", [True, False])
def test_a_spec_whose_attempt_runs_or_stopped_is_missing(held: bool) -> None:
    leaf = Leaf(name=f"attempt-{held}")
    attempts_dir_in(leaf._class_dir).mkdir(parents=True)
    path = lock_path_in(leaf._class_dir, leaf._artifact_hash)
    if held:
        with lock(path):
            assert missing([leaf], Dag()) == [leaf]
    else:
        path.touch()
        assert missing([leaf], Dag()) == [leaf]


def _record_call(calls_dir: str, cls: type, value: object) -> None:
    # Workers run create() in a child process, so calls are recorded on disk.
    with (Path(calls_dir) / cls.__name__).open("a", encoding="utf-8") as f:
        f.write(f"{value}\n")


def _calls(calls_dir: Path, cls: type) -> list[str]:
    path = calls_dir / cls.__name__
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


class TrackingLeaf(Spec[int]):
    n: int
    calls_dir: str

    def create(self) -> int:
        _record_call(self.calls_dir, type(self), self.n)
        return self.n * 2


class TrackingMid(Spec[int]):
    label: str
    child: TrackingLeaf
    calls_dir: str

    def create(self) -> int:
        _record_call(self.calls_dir, type(self), self.label)
        return self.child.create() + 1


class LazyChildLoader(Spec[int]):
    base: int
    calls_dir: str

    def create(self) -> int:
        _record_call(self.calls_dir, type(self), self.base)
        return self.base + TrackingLeaf(n=self.base, calls_dir=self.calls_dir).create()


class AlwaysFails(Spec[int]):
    name: str

    def create(self) -> int:
        raise RuntimeError(f"intentional failure: {self.name}")


class DependsOnFailing(Spec[int]):
    label: str
    child: AlwaysFails

    def create(self) -> int:
        return self.child.create() + 1


def test_worker_run_runs_single_zero_dependency_node(tmp_path: Path):
    leaf = TrackingLeaf(n=3, calls_dir=str(tmp_path))

    run_on_workers([leaf], [LocalThreadWorkerBackend()])

    assert _calls(tmp_path, TrackingLeaf) == ["3"]
    assert leaf.done
    assert leaf.create() == 6


def test_worker_run_runs_static_dependencies_in_order(tmp_path: Path):
    leaf = TrackingLeaf(n=4, calls_dir=str(tmp_path))
    mid = TrackingMid(label="m", child=leaf, calls_dir=str(tmp_path))

    run_on_workers([mid], [LocalThreadWorkerBackend()])

    assert _calls(tmp_path, TrackingLeaf) == ["4"]
    assert _calls(tmp_path, TrackingMid) == ["m"]
    assert mid.create() == 9


def test_worker_run_handles_shared_dependency_only_once(
    tmp_path: Path,
):
    shared = TrackingLeaf(n=5, calls_dir=str(tmp_path))
    left = TrackingMid(label="L", child=shared, calls_dir=str(tmp_path))
    right = TrackingMid(label="R", child=shared, calls_dir=str(tmp_path))

    run_on_workers([left, right], [LocalThreadWorkerBackend()])

    assert _calls(tmp_path, TrackingLeaf) == ["5"]
    assert sorted(_calls(tmp_path, TrackingMid)) == ["L", "R"]


def test_worker_run_with_multiple_workers_runs_independent_nodes(
    tmp_path: Path,
):
    leaves = [TrackingLeaf(n=i, calls_dir=str(tmp_path)) for i in range(8)]

    run_on_workers(leaves, [LocalThreadWorkerBackend(max_workers=4)])

    assert sorted(_calls(tmp_path, TrackingLeaf), key=int) == [str(i) for i in range(8)]
    for leaf in leaves:
        assert leaf.done


def test_worker_run_discovers_lazy_dependencies_and_reruns_parent(
    tmp_path: Path,
):
    parent = LazyChildLoader(base=7, calls_dir=str(tmp_path))

    run_on_workers([parent], [LocalThreadWorkerBackend()])

    assert _calls(tmp_path, TrackingLeaf) == ["7"]
    # Parent's create() is called once to discover the lazy dep (raising
    # _DependencyNotReady), then once more after the dep completes.
    assert _calls(tmp_path, LazyChildLoader) == ["7", "7"]
    assert parent.create() == 21
    assert [end["outcome"] for end in ends_of(parent)] == ["blocked", "ok"]
    assert "create failed" not in log_of(parent)


def test_worker_run_skips_already_completed_objects(tmp_path: Path):
    leaf = TrackingLeaf(n=8, calls_dir=str(tmp_path))
    leaf.create()
    mid = TrackingMid(label="cached-child", child=leaf, calls_dir=str(tmp_path))

    run_on_workers([mid], [LocalThreadWorkerBackend()])

    # Only the in-process create above; the worker never recomputed the leaf.
    assert _calls(tmp_path, TrackingLeaf) == ["8"]
    assert _calls(tmp_path, TrackingMid) == ["cached-child"]


def test_worker_run_reports_worker_failures():
    failing = AlwaysFails(name="boom")
    parent = DependsOnFailing(label="p", child=failing)

    config = get_config()
    no_retries = config.model_copy(
        update={
            "worker": config.worker.model_copy(update={"max_retries_per_object": 0})
        }
    )
    with (
        override_config(no_retries),
        pytest.raises(RuntimeError, match="1 spec failed"),
    ):
        run_on_workers(
            [parent],
            [LocalThreadWorkerBackend()],
        )

    assert ends_of(failing) == [
        {"outcome": "failed", "error": "RuntimeError: intentional failure: boom"}
    ]
    assert not parent._attempt_dir.exists()
