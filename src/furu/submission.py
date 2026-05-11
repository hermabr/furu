from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from furu.core import Furu
from furu.graph import NodeKey, node_key_for
from furu.metadata import ArtifactSpec
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle
from furu.server.client import SchedulerClient
from furu.server.models import SubmissionGraphView, SubmissionState, SubmissionStatus

T = TypeVar("T")


class SubmissionFailed(RuntimeError):
    pass


class SubmissionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Submission(Generic[T]):
    id: str
    _client: SchedulerClient
    _input_order: tuple[NodeKey, ...]
    _single_input: bool
    _on_poll: Callable[[], None] | None = None
    _cancel_callback: Callable[[], None] | None = None

    def status(self) -> SubmissionStatus:
        return self._client.get_submission(self.id)

    def graph(self) -> SubmissionGraphView:
        return self._client.get_graph(self.id)

    def result(self, *, timeout_s: float | None = None) -> T:
        status = self._client.wait_until_done(
            self.id,
            timeout_s=timeout_s,
            on_poll=self._on_poll,
        )

        if status.state == SubmissionState.FAILED:
            raise SubmissionFailed(
                status.failure_summary or f"submission {self.id} failed"
            )
        if status.state == SubmissionState.CANCELLED:
            raise SubmissionCancelled(self.id)

        graph = self.graph()
        nodes_by_key = {node.key: node for node in graph.nodes}
        values = [
            _load_result_for_node(key, nodes_by_key[key].artifact)
            for key in self._input_order
        ]

        if self._single_input:
            return cast(T, values[0])
        return cast(T, values)

    def cancel(self) -> None:
        self._client.cancel_submission(self.id)
        if self._cancel_callback is not None:
            self._cancel_callback()


def _load_result_for_node(key: NodeKey, artifact: ArtifactSpec) -> Any:
    obj = Furu.from_artifact(artifact)
    if node_key_for(obj) != key:
        raise RuntimeError(
            "node key does not match reconstructed artifact: "
            f"node={key!r}, reconstructed={node_key_for(obj)!r}"
        )
    result_dir = result_dir_for_loading(obj)
    if result_dir is None:
        raise RuntimeError(f"result for {key.object_id} is missing")
    return load_result_bundle(result_dir)
