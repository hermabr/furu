from __future__ import annotations

import functools
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from furu.core import Spec


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


class _BatchedCreate[S, T]:
    def __init__(
        self,
        func: Callable[[list[S]], list[T]],
        batch_fn: Callable[[Any], tuple[Hashable, int]],
        /,
    ) -> None:
        self.func = func
        self.batch_fn = batch_fn

    @overload
    def __get__(
        self, obj: None, _objtype: type, /
    ) -> Callable[[list[S]], list[T]]: ...
    @overload
    def __get__(self, obj: S, _objtype: type, /) -> Callable[[], T]: ...
    def __get__(self, obj: Any, _objtype: type | None = None, /) -> Any:
        raise TypeError(
            "@furu.batched hooks are captured by furu at class creation; "
            "define them on a furu.Spec subclass"
        )


class _BatchedCreateVerb:
    def __get__(self, obj: Spec[Any] | None, _objtype: type | None = None) -> Any:
        from furu.execution.load_or_create import _load_or_create

        if obj is None:
            return _load_or_create
        return functools.partial(_load_or_create, obj)
