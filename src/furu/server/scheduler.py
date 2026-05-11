from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from furu.core import Furu
from furu.graph import GraphEdge, GraphFragment, NodeKey, node_key_for
from furu.metadata import ArtifactSpec
from furu.migration import result_dir_for_loading
from furu.server.models import (
    CreateSubmissionRequest,
    FailureRecord,
    LeaseResponse,
    NodeState,
    NodeStatus,
    SubmissionGraphView,
    SubmissionState,
    SubmissionStatus,
)


class ProtocolError(RuntimeError):
    pass


@dataclass(slots=True)
class SubmissionRecord:
    id: str
    roots: tuple[NodeKey, ...]
    input_order: tuple[NodeKey, ...]
    single_input: bool
    nodes: set[NodeKey] = field(default_factory=set)
    cancelled: bool = False


@dataclass(slots=True)
class NodeRecord:
    key: NodeKey
    artifact: ArtifactSpec
    state: NodeState
    active_lease_id: str | None = None
    failure: FailureRecord | None = None


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    id: str
    submission_id: str
    node_key: NodeKey
    artifact: ArtifactSpec


@dataclass(frozen=True, slots=True)
class JobGroupRecord:
    id: str
    lease_ids: tuple[str, ...]


class SchedulerState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.submissions: dict[str, SubmissionRecord] = {}
        self.nodes: dict[NodeKey, NodeRecord] = {}
        self.edges: set[GraphEdge] = set()
        self.leases: dict[str, LeaseRecord] = {}
        self.job_groups: dict[str, JobGroupRecord] = {}

    def create_submission(self, request: CreateSubmissionRequest) -> str:
        with self.lock:
            submission_id = uuid.uuid4().hex
            record = SubmissionRecord(
                id=submission_id,
                roots=request.roots,
                input_order=request.input_order,
                single_input=request.single_input,
            )
            self.submissions[submission_id] = record
            self._merge_graph_fragment_locked(submission_id, request.graph)
            for key in (*request.roots, *request.input_order):
                if key not in self.nodes:
                    raise ProtocolError(f"submission references unknown node {key!r}")
                record.nodes.add(key)
            self._recompute_graph_state_locked(submission_id)
            return submission_id

    def get_submission(self, submission_id: str) -> SubmissionStatus:
        with self.lock:
            record = self._submission_locked(submission_id)
            state = self._submission_state_locked(record)
            return SubmissionStatus(
                id=record.id,
                state=state,
                roots=record.roots,
                input_order=record.input_order,
                single_input=record.single_input,
                failure_summary=self._failure_summary_locked(record),
            )

    def get_graph(self, submission_id: str) -> SubmissionGraphView:
        with self.lock:
            record = self._submission_locked(submission_id)
            nodes = tuple(
                self._node_status_locked(self.nodes[key])
                for key in sorted(record.nodes, key=_key_sort_value)
            )
            edges = tuple(
                sorted(
                    (
                        edge
                        for edge in self.edges
                        if edge.dependency in record.nodes
                        and edge.dependent in record.nodes
                    ),
                    key=_edge_sort_value,
                )
            )
            return SubmissionGraphView(
                nodes=nodes,
                edges=edges,
                roots=record.roots,
                input_order=record.input_order,
                single_input=record.single_input,
            )

    def cancel_submission(self, submission_id: str) -> None:
        with self.lock:
            record = self._submission_locked(submission_id)
            record.cancelled = True
            for key in record.nodes:
                node = self.nodes[key]
                if node.state not in {
                    NodeState.DONE,
                    NodeState.FAILED,
                    NodeState.CANCELLED,
                }:
                    node.state = NodeState.CANCELLED
                    node.active_lease_id = None

    def reserve_ready_nodes(
        self,
        *,
        submission_id: str,
        max_count: int,
    ) -> list[LeaseRecord]:
        if max_count <= 0:
            return []

        with self.lock:
            record = self._submission_locked(submission_id)
            if record.cancelled:
                return []

            self._recompute_graph_state_locked(submission_id)
            leases: list[LeaseRecord] = []

            for key in sorted(record.nodes, key=_key_sort_value):
                if len(leases) == max_count:
                    break

                node = self.nodes[key]
                if node.state != NodeState.READY or node.active_lease_id is not None:
                    continue

                if self._result_exists_locked(node):
                    node.state = NodeState.DONE
                    node.failure = None
                    self._recompute_graph_state_locked(submission_id)
                    continue

                lease = LeaseRecord(
                    id=uuid.uuid4().hex,
                    submission_id=submission_id,
                    node_key=node.key,
                    artifact=node.artifact,
                )
                self.leases[lease.id] = lease
                node.state = NodeState.RUNNING
                node.active_lease_id = lease.id
                leases.append(lease)

            return leases

    def get_lease(self, lease_id: str) -> LeaseRecord:
        with self.lock:
            try:
                return self.leases[lease_id]
            except KeyError as exc:
                raise ProtocolError(f"unknown lease {lease_id}") from exc

    def complete_lease(self, lease_id: str, *, node_key: NodeKey) -> None:
        with self.lock:
            lease = self._lease_locked(lease_id)
            node = self._node_for_lease_locked(lease)
            self._validate_lease_locked(lease, node, node_key=node_key)

            if not self._result_exists_locked(node):
                raise ProtocolError("worker reported completion but result is missing")

            node.state = NodeState.DONE
            node.active_lease_id = None
            node.failure = None
            self._close_lease_locked(lease_id)
            self._recompute_graph_state_locked(lease.submission_id)

    def report_dependency(
        self,
        lease_id: str,
        *,
        blocked: NodeKey,
        dependencies: Iterable[NodeKey],
        graph_fragment: GraphFragment,
    ) -> None:
        with self.lock:
            lease = self._lease_locked(lease_id)
            node = self._node_for_lease_locked(lease)
            self._validate_lease_locked(lease, node, node_key=blocked)

            self._merge_graph_fragment_locked(lease.submission_id, graph_fragment)
            for dependency in dependencies:
                if dependency not in self.nodes:
                    raise ProtocolError(
                        f"dependency report references unknown node {dependency!r}"
                    )
                self.edges.add(
                    GraphEdge(
                        dependency=dependency,
                        dependent=blocked,
                    )
                )
                self.submissions[lease.submission_id].nodes.add(dependency)

            node.state = NodeState.WAITING
            node.active_lease_id = None
            self._close_lease_locked(lease_id)
            self._recompute_graph_state_locked(lease.submission_id)

    def fail_lease(self, lease_id: str, *, failure: FailureRecord) -> None:
        with self.lock:
            lease = self._lease_locked(lease_id)
            node = self._node_for_lease_locked(lease)
            self._validate_lease_locked(lease, node, node_key=failure.node_key)

            node.state = NodeState.FAILED
            node.failure = failure
            node.active_lease_id = None
            self._close_lease_locked(lease_id)
            self._propagate_failure_to_dependents_locked(node)

    def create_job_group(self, lease_ids: tuple[str, ...]) -> JobGroupRecord:
        with self.lock:
            for lease_id in lease_ids:
                self._lease_locked(lease_id)
            record = JobGroupRecord(
                id=uuid.uuid4().hex,
                lease_ids=lease_ids,
            )
            self.job_groups[record.id] = record
            return record

    def get_lease_for_array_index(
        self,
        *,
        job_group_id: str,
        array_index: int,
    ) -> LeaseRecord:
        with self.lock:
            try:
                group = self.job_groups[job_group_id]
            except KeyError as exc:
                raise ProtocolError(f"unknown job group {job_group_id}") from exc

            if array_index < 0 or array_index >= len(group.lease_ids):
                raise ProtocolError(
                    f"array index {array_index} is outside job group {job_group_id}"
                )
            return self._lease_locked(group.lease_ids[array_index])

    def _merge_graph_fragment_locked(
        self,
        submission_id: str,
        fragment: GraphFragment,
    ) -> None:
        record = self._submission_locked(submission_id)
        for graph_node in fragment.nodes:
            existing = self.nodes.get(graph_node.key)
            if existing is None:
                self.nodes[graph_node.key] = NodeRecord(
                    key=graph_node.key,
                    artifact=graph_node.artifact,
                    state=NodeState.WAITING,
                )
            elif existing.artifact != graph_node.artifact:
                raise ProtocolError(f"conflicting artifact for node {graph_node.key!r}")
            record.nodes.add(graph_node.key)

        for edge in fragment.edges:
            if edge.dependency not in self.nodes or edge.dependent not in self.nodes:
                raise ProtocolError(f"edge references unknown node {edge!r}")
            self.edges.add(edge)
            record.nodes.add(edge.dependency)
            record.nodes.add(edge.dependent)

        for key in fragment.done:
            node = self.nodes.get(key)
            if node is None:
                raise ProtocolError(f"done list references unknown node {key!r}")
            node.state = NodeState.DONE
            node.active_lease_id = None
            node.failure = None

    def _recompute_graph_state_locked(self, submission_id: str) -> None:
        record = self._submission_locked(submission_id)
        changed = True
        while changed:
            changed = False
            for key in sorted(record.nodes, key=_key_sort_value):
                node = self.nodes[key]
                previous = node.state

                if node.state == NodeState.CANCELLED:
                    continue
                if node.state == NodeState.RUNNING:
                    continue
                if self._result_exists_locked(node):
                    node.state = NodeState.DONE
                    node.failure = None
                elif node.state == NodeState.DONE:
                    continue
                elif node.state == NodeState.FAILED:
                    continue
                else:
                    dependencies = self._dependencies_of_locked(key)
                    dependency_states = [
                        self.nodes[dependency].state for dependency in dependencies
                    ]
                    failed_dependencies = [
                        self.nodes[dependency]
                        for dependency in dependencies
                        if self.nodes[dependency].state == NodeState.FAILED
                    ]
                    if failed_dependencies:
                        failed_dependency = failed_dependencies[0]
                        node.state = NodeState.FAILED
                        node.failure = FailureRecord(
                            node_key=node.key,
                            error_type="UpstreamDependencyFailed",
                            error_message=(
                                f"dependency failed: {failed_dependency.key.object_id}"
                            ),
                            traceback="",
                        )
                    elif dependencies and any(
                        state != NodeState.DONE for state in dependency_states
                    ):
                        node.state = NodeState.WAITING
                    else:
                        node.state = NodeState.READY

                if node.state != previous:
                    changed = True

    def _propagate_failure_to_dependents_locked(self, node: NodeRecord) -> None:
        for dependent_key in self._dependents_of_locked(node.key):
            dependent = self.nodes[dependent_key]
            if dependent.state in {
                NodeState.DONE,
                NodeState.FAILED,
                NodeState.CANCELLED,
            }:
                continue
            dependent.state = NodeState.FAILED
            dependent.active_lease_id = None
            dependent.failure = FailureRecord(
                node_key=dependent.key,
                error_type="UpstreamDependencyFailed",
                error_message=f"dependency failed: {node.key.object_id}",
                traceback="",
            )
            self._propagate_failure_to_dependents_locked(dependent)

    def _submission_state_locked(self, record: SubmissionRecord) -> SubmissionState:
        if record.cancelled:
            return SubmissionState.CANCELLED
        root_states = [self.nodes[root].state for root in record.roots]
        if all(state == NodeState.DONE for state in root_states):
            return SubmissionState.DONE
        if any(state == NodeState.CANCELLED for state in root_states):
            return SubmissionState.CANCELLED
        if any(state == NodeState.FAILED for state in root_states):
            return SubmissionState.FAILED
        return SubmissionState.RUNNING

    def _failure_summary_locked(self, record: SubmissionRecord) -> str | None:
        failed_nodes = [
            self.nodes[key]
            for key in sorted(record.nodes, key=_key_sort_value)
            if self.nodes[key].state == NodeState.FAILED
        ]
        if not failed_nodes:
            return None
        failed = failed_nodes[0]
        if failed.failure is None:
            return f"{failed.key.object_id} failed"
        return (
            f"{failed.key.object_id} failed with "
            f"{failed.failure.error_type}: {failed.failure.error_message}"
        )

    def _result_exists_locked(self, node: NodeRecord) -> bool:
        obj = Furu.from_artifact(node.artifact)
        if node_key_for(obj) != node.key:
            raise ProtocolError(
                "node key does not match reconstructed artifact: "
                f"node={node.key!r}, reconstructed={node_key_for(obj)!r}"
            )
        return result_dir_for_loading(obj) is not None

    def _dependencies_of_locked(self, key: NodeKey) -> tuple[NodeKey, ...]:
        return tuple(
            sorted(
                (edge.dependency for edge in self.edges if edge.dependent == key),
                key=_key_sort_value,
            )
        )

    def _dependents_of_locked(self, key: NodeKey) -> tuple[NodeKey, ...]:
        return tuple(
            sorted(
                (edge.dependent for edge in self.edges if edge.dependency == key),
                key=_key_sort_value,
            )
        )

    def _lease_locked(self, lease_id: str) -> LeaseRecord:
        try:
            return self.leases[lease_id]
        except KeyError as exc:
            raise ProtocolError(f"unknown lease {lease_id}") from exc

    def _submission_locked(self, submission_id: str) -> SubmissionRecord:
        try:
            return self.submissions[submission_id]
        except KeyError as exc:
            raise ProtocolError(f"unknown submission {submission_id}") from exc

    def _node_for_lease_locked(self, lease: LeaseRecord) -> NodeRecord:
        node = self.nodes[lease.node_key]
        if node.active_lease_id != lease.id:
            raise ProtocolError(f"lease {lease.id} is not active for its node")
        return node

    def _validate_lease_locked(
        self,
        lease: LeaseRecord,
        node: NodeRecord,
        *,
        node_key: NodeKey,
    ) -> None:
        if node.key != node_key:
            raise ProtocolError(
                f"lease {lease.id} is for {node.key!r}, got {node_key!r}"
            )

    def _close_lease_locked(self, lease_id: str) -> None:
        self.leases.pop(lease_id, None)

    def _node_status_locked(self, node: NodeRecord) -> NodeStatus:
        return NodeStatus(
            key=node.key,
            artifact=node.artifact,
            state=node.state,
            active_lease_id=node.active_lease_id,
            failure=node.failure,
        )


def lease_response_from_record(lease: LeaseRecord) -> LeaseResponse:
    return LeaseResponse(
        lease_id=lease.id,
        submission_id=lease.submission_id,
        node_key=lease.node_key,
        artifact=lease.artifact,
    )


def _key_sort_value(key: NodeKey) -> tuple[str, str]:
    return (key.data_path, key.object_id)


def _edge_sort_value(edge: GraphEdge) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_key_sort_value(edge.dependency), _key_sort_value(edge.dependent))
