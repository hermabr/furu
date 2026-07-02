from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from furu.core import Spec

DependencyCallKind: TypeAlias = Literal["create", "load_existing"]


_worker_execution_lease_id: ContextVar[str | None] = ContextVar(
    "_worker_execution_lease_id",
    default=None,
)


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
    dependencies: tuple[Spec, ...]
    call_kind: DependencyCallKind

    def __init__[T](
        self,
        dependencies: Sequence[Spec[T]],
        *,
        call_kind: DependencyCallKind,
    ) -> None:
        self.dependencies = tuple(dependencies)
        self.call_kind = call_kind

        super().__init__(
            f"{call_kind} discovered "
            f"{len(self.dependencies)} missing dependency/dependencies"
        )
