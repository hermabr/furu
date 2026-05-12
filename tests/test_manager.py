from furu import Furu
from furu.dag import FuruDagNode, make_execution_dag
from furu.execution.manager import Manager
from furu.metadata import ArtifactSpec
from furu.worker import BlockedUpdate, FinishUpdate, Job


class ManagerLeaf(Furu[int]):
    key: int

    def create(self) -> int:
        return self.key


class ManagerParent(Furu[int]):
    child: ManagerLeaf

    def create(self) -> int:
        return self.child.load_or_create() + 1


class ManagerLazyParent(Furu[int]):
    key: int

    def create(self) -> int:
        return ManagerLeaf(key=self.key).load_or_create() + 1


def test_make_execution_dag_populates_ready_and_blocked_maps() -> None:
    leaf = ManagerLeaf(key=1)
    parent = ManagerParent(child=leaf)
    nodes_by_id: dict[str, FuruDagNode[Furu]] = {}
    ready: dict[str, FuruDagNode[Furu]] = {}
    blocked: dict[str, FuruDagNode[Furu]] = {}

    zero_dep = make_execution_dag(
        [parent],
        nodes_by_id,
        ready=ready,
        blocked=blocked,
    )

    assert [node.obj.object_id for node in zero_dep] == [leaf.object_id]
    assert set(ready) == {leaf.object_id}
    assert set(blocked) == {parent.object_id}


def test_manager_moves_ready_job_to_running_then_completed() -> None:
    leaf = ManagerLeaf(key=2)
    manager = Manager.from_objects([leaf])

    job = manager.get_job()
    assert isinstance(job, Job)
    assert job.artifact.object_id == leaf.object_id
    assert manager.ready == {}
    assert set(manager.running) == {job.lease_id}

    manager.finish_job(job.lease_id, FinishUpdate(status="completed"))

    assert set(manager.completed) == {leaf.object_id}
    assert manager.done_event.is_set()
    assert manager.completion_error() is None


def test_manager_blocked_job_adds_dependencies_and_unblocks_after_finish() -> None:
    parent = ManagerLazyParent(key=3)
    leaf = ManagerLeaf(key=3)
    manager = Manager.from_objects([parent])

    parent_job = manager.get_job()
    assert isinstance(parent_job, Job)

    manager.block_job(
        parent_job.lease_id,
        BlockedUpdate(dependencies=[ArtifactSpec.from_furu(leaf)]),
    )

    assert set(manager.ready) == {leaf.object_id}
    assert set(manager.blocked) == {parent.object_id}

    leaf_job = manager.get_job()
    assert isinstance(leaf_job, Job)
    assert leaf_job.artifact.object_id == leaf.object_id

    manager.finish_job(leaf_job.lease_id, FinishUpdate(status="completed"))

    assert set(manager.ready) == {parent.object_id}
    assert manager.blocked == {}
