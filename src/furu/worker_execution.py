from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from furu.artifact import artifact_spec_for
from furu.dag import NodeKey, node_key_for
from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.core import Furu

type DependencyCallKind = Literal["load_or_create", "try_load"]


@dataclass(frozen=True)
class WorkerExecutionContext:
    current_node: NodeKey
    lease_id: str


_worker_execution_context: ContextVar[WorkerExecutionContext | None] = ContextVar(
    "_worker_execution_context",
    default=None,
)


def _in_worker_execution_context() -> bool:
    return _worker_execution_context.get() is not None


@contextmanager
def worker_execution_context(
    *,
    current_node: NodeKey,
    lease_id: str,
) -> Iterator[None]:
    token = _worker_execution_context.set(
        WorkerExecutionContext(
            current_node=current_node,
            lease_id=lease_id,
        )
    )

    try:
        yield
    finally:
        _worker_execution_context.reset(token)


class _DependencyNotReady(BaseException):
    dependencies: tuple[Furu[Any], ...]
    call_kind: DependencyCallKind
    artifacts: tuple[ArtifactSpec, ...]
    keys: tuple[NodeKey, ...]

    def __init__(
        self,
        dependencies: Sequence[Furu[Any]],
        *,
        call_kind: DependencyCallKind,
    ) -> None:
        self.dependencies = tuple(dependencies)
        self.call_kind = call_kind
        self.artifacts = tuple(artifact_spec_for(dep) for dep in self.dependencies)
        self.keys = tuple(node_key_for(dep) for dep in self.dependencies)

        super().__init__(
            f"{call_kind} discovered "
            f"{len(self.dependencies)} missing dependency/dependencies"
        )
