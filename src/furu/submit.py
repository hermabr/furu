from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, overload

from furu.core import Furu
from furu.executor import Executor
from furu.graph import discover_missing_closure, node_key_for
from furu.submission import Submission

T = TypeVar("T")


@overload
def submit(
    obj: Furu[T],
    *,
    executor: Executor,
) -> Submission[T]: ...


@overload
def submit(
    objs: Sequence[Furu[T]],
    *,
    executor: Executor,
) -> Submission[list[T]]: ...


def submit(
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    executor: Executor,
) -> Submission[T] | Submission[list[T]]:
    if isinstance(obj_or_objs, Furu):
        objs = [obj_or_objs]
        single_input = True
    else:
        if not isinstance(obj_or_objs, Sequence):
            raise TypeError(
                "submit() expected a Furu object or a sequence of Furu objects"
            )
        objs = list(obj_or_objs)
        single_input = False
        if any(not isinstance(obj, Furu) for obj in objs):
            raise TypeError("submit() expected Furu objects")

    graph = discover_missing_closure(objs)
    input_order = tuple(node_key_for(obj) for obj in objs)
    return executor.submit(
        graph=graph,
        roots=input_order,
        input_order=input_order,
        single_input=single_input,
    )
