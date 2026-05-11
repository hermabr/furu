from __future__ import annotations

import traceback as traceback_module
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

from furu.core import Furu
from furu.execution import (
    cleanup_incomplete_attempt,
    cleanup_or_mark_failed_attempt,
    execute_artifact_create_logic,
)
from furu.graph import GraphFragment, NodeKey, discover_missing_closure, node_key_for
from furu.metadata import ArtifactSpec
from furu.migration import result_dir_for_loading
from furu.worker_context import _DependencyNotReady, worker_execution_context


class WorkerExecutionResultKind(str, Enum):
    DONE = "done"
    DEPENDENCY_NOT_READY = "dependency_not_ready"
    FAILED = "failed"


class WorkerExecutionResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    kind: WorkerExecutionResultKind
    node_key: NodeKey

    call_kind: Literal["load_or_create", "try_load"] | None = None
    dependencies: tuple[NodeKey, ...] = ()
    graph_fragment: GraphFragment | None = None

    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None

    @classmethod
    def done(cls, *, node_key: NodeKey) -> Self:
        return cls(
            kind=WorkerExecutionResultKind.DONE,
            node_key=node_key,
        )

    @classmethod
    def dependency_not_ready(
        cls,
        *,
        blocked: NodeKey,
        call_kind: Literal["load_or_create", "try_load"],
        dependencies: tuple[NodeKey, ...],
        graph_fragment: GraphFragment,
    ) -> Self:
        return cls(
            kind=WorkerExecutionResultKind.DEPENDENCY_NOT_READY,
            node_key=blocked,
            call_kind=call_kind,
            dependencies=dependencies,
            graph_fragment=graph_fragment,
        )

    @classmethod
    def failed(
        cls,
        *,
        node_key: NodeKey,
        error_type: str,
        error_message: str,
        traceback: str,
    ) -> Self:
        return cls(
            kind=WorkerExecutionResultKind.FAILED,
            node_key=node_key,
            error_type=error_type,
            error_message=error_message,
            traceback=traceback,
        )


def execute_one_artifact(
    *,
    lease_id: str,
    node_key: NodeKey,
    artifact: ArtifactSpec,
) -> WorkerExecutionResult:
    obj = Furu.from_artifact(artifact)
    if node_key_for(obj) != node_key:
        return WorkerExecutionResult.failed(
            node_key=node_key,
            error_type="ValueError",
            error_message=(
                "leased node key does not match reconstructed artifact: "
                f"leased={node_key!r}, reconstructed={node_key_for(obj)!r}"
            ),
            traceback="",
        )

    if result_dir_for_loading(obj) is not None:
        return WorkerExecutionResult.done(node_key=node_key)

    try:
        with worker_execution_context(
            current_node=node_key,
            lease_id=lease_id,
        ):
            execute_artifact_create_logic(obj)

        return WorkerExecutionResult.done(node_key=node_key)

    except _DependencyNotReady as exc:
        cleanup_incomplete_attempt(obj)
        fragment = discover_missing_closure(exc.dependencies)

        return WorkerExecutionResult.dependency_not_ready(
            blocked=node_key,
            call_kind=exc.call_kind,
            dependencies=tuple(node_key_for(dep) for dep in exc.dependencies),
            graph_fragment=fragment,
        )

    except Exception as exc:
        cleanup_or_mark_failed_attempt(obj)

        return WorkerExecutionResult.failed(
            node_key=node_key,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback_module.format_exc(),
        )
