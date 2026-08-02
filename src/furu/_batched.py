from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, NamedTuple, overload


class _BatchedHook(NamedTuple):
    func: Callable[[list[Any]], list[Any]]
    batch_fn: Callable[[Any], tuple[Hashable, int]]


class batched:
    def __init__(self, batch_fn: Callable[[Any], tuple[Hashable, int]], /) -> None:
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
    batch_fn: Callable[[Any], tuple[Hashable, int]]

    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: None, _objtype: type, /) -> Callable[[S], T]: ...

        @overload
        def __get__(self, obj: S, _objtype: type, /) -> Callable[[], T]: ...
