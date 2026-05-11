from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

from furu.metadata import ArtifactSpec

if TYPE_CHECKING:
    from furu.core import Furu

type DependencyCallKind = Literal["load_or_create", "try_load"]


_worker_execution_lease_id: ContextVar[str | None] = ContextVar(
    "_worker_execution_lease_id",
    default=None,
)


def _in_worker_execution_context() -> bool:
    return _worker_execution_lease_id.get() is not None


@contextmanager
def worker_execution_context(
    *,
    lease_id: str,
) -> Iterator[None]:
    token = _worker_execution_lease_id.set(lease_id)

    try:
        yield
    finally:
        _worker_execution_lease_id.reset(token)


class _DependencyNotReady(BaseException):
    dependencies: tuple[Furu[Any], ...]
    call_kind: DependencyCallKind
    artifacts: tuple[ArtifactSpec, ...]

    def __init__(
        self,
        dependencies: Sequence[Furu[Any]],
        *,
        call_kind: DependencyCallKind,
    ) -> None:
        self.dependencies = tuple(dependencies)
        self.call_kind = call_kind
        self.artifacts = tuple(ArtifactSpec.from_furu(dep) for dep in self.dependencies)

        super().__init__(
            f"{call_kind} discovered "
            f"{len(self.dependencies)} missing dependency/dependencies"
        )
