from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, NamedTuple, overload

from furu.resources import Worker

type BatchFn = Callable[[Any, Worker], tuple[Hashable, int]]


class _BatchedHook(NamedTuple):
    func: Callable[[list[Any]], list[Any]]
    batch_fn: BatchFn


class batched:
    """Run create() on groups of specs sharing a batch key.

    ``batch_fn(spec, worker)`` returns ``(key, cap)``; ``worker`` is the
    leasing worker (or this machine for in-process create), so the cap can
    follow its resources, e.g. ``(spec.model, worker.gpus)``.
    """

    def __init__(self, batch_fn: BatchFn, /) -> None:
        if not callable(batch_fn):
            raise TypeError(
                "@furu.batched needs a batch key function: @furu.batched(batch_key)"
            )
        self.batch_fn = batch_fn

    def __call__[S, T](
        self, func: Callable[[list[S]], list[T]], /
    ) -> _BatchedCreate[S, T]:
        if getattr(func, "__name__", None) != "create":
            raise TypeError("@furu.batched can only decorate create()")
        return _BatchedCreate(func, self.batch_fn)


class _BatchedCreate[S, T](NamedTuple):
    func: Callable[[list[S]], list[T]]
    batch_fn: BatchFn

    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: None, _objtype: type, /) -> Callable[[S], T]: ...

        @overload
        def __get__(self, obj: S, _objtype: type, /) -> Callable[[], T]: ...
