from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from furu.core import Furu
from furu.executors import Executor
from furu.graph import discover_missing_closure
from furu.submission import Submission


@overload
def submit[T](obj: Furu[T], *, executor: Executor) -> Submission[T]: ...


@overload
def submit[T](
    objs: Sequence[Furu[T]], *, executor: Executor
) -> Submission[list[T]]: ...


def submit[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    executor: Executor,
) -> Submission[T] | Submission[list[T]]:
    if isinstance(obj_or_objs, Furu):
        roots = [obj_or_objs]
        single_input = True
    else:
        roots = list(obj_or_objs)
        single_input = False
        if any(not isinstance(obj, Furu) for obj in roots):
            raise TypeError("submit() expected Furu objects")
    if not roots:
        raise ValueError("submit() requires at least one artifact")

    graph = discover_missing_closure(roots)
    return executor.submit(roots=roots, graph=graph, single_input=single_input)
