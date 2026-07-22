from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, overload

from furu.core import Spec


class _BatchedCreate[S: Spec, T]:
    def __init__(
        self,
        batch_fn: Callable[[S], tuple[Hashable, int]],
        func: Callable[[list[S]], list[T]],
    ) -> None:
        self.batch_fn = batch_fn
        self.func = func

    def __call__(self, objs: list[S]) -> list[T]:
        return self.func(objs)

    @overload
    def __get__(self, obj: None, owner: type[S]) -> Callable[[list[S]], list[T]]: ...
    @overload
    def __get__(self, obj: S, owner: type[S] | None = None) -> Callable[[], T]: ...
    def __get__(
        self, obj: S | None, owner: type[S] | None = None
    ) -> Callable[[list[S]], list[T]] | Callable[[], T]:
        from furu.execution.load_or_create import _load_or_create

        return (
            (lambda objs: _load_or_create(objs))
            if obj is None
            else (lambda: _load_or_create(obj))
        )


def batched[S: Spec, T](
    batch_fn: Callable[[S], tuple[Hashable, int]],
) -> Callable[[Callable[[list[S]], list[T]]], _BatchedCreate[S, T]]:
    def decorate(func: Callable[[list[S]], list[T]]) -> _BatchedCreate[S, T]:
        if getattr(func, "__name__", None) != "create":
            raise TypeError("@furu.batched can only decorate create()")
        return _BatchedCreate(batch_fn, func)

    return decorate


def _batch_group(obj: Any) -> tuple[Hashable, int] | None:
    if (batch_fn := type(obj)._furu_batch_fn) is None:
        return None
    key, max_size = batch_fn(obj)
    if type(max_size) is not int or max_size < 1:
        raise ValueError("@furu.batched group size must be a positive integer")
    hash(key)
    execution = obj._metadata.execution
    execution_key = (
        execution
        if execution == "inline"
        else (
            tuple(sorted(execution.environment.items())),
            execution.required_environment,
            execution.reuse,
        )
    )
    return ((type(obj), key, obj._metadata.requires, execution_key), max_size)
