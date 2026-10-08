from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from furu.core import Spec


_in_worker_execution: ContextVar[bool] = ContextVar(
    "_in_worker_execution",
    default=False,
)


@contextmanager
def worker_execution_context() -> Generator[None]:
    token = _in_worker_execution.set(True)

    try:
        yield
    finally:
        _in_worker_execution.reset(token)


class _DependencyNotReady(BaseException):
    dependencies: tuple[Spec, ...]

    def __init__[T](self, dependencies: Sequence[Spec[T]]) -> None:
        self.dependencies = tuple(dependencies)

        super().__init__(
            f"create discovered {len(self.dependencies)} missing "
            "dependency/dependencies"
        )
