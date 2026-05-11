from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from furu.core import Furu
from furu.executors import Executor
from furu.graph import discover_missing_closure
from furu.submission import Submission


@overload
def submit[T](
    obj: Furu[T],
    *,
    executor: Executor,
) -> Submission[T]: ...


@overload
def submit[T](
    objs: Sequence[Furu[T]],
    *,
    executor: Executor,
) -> Submission[list[T]]: ...


def submit[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
    *,
    executor: Executor,
) -> Submission[T] | Submission[list[T]]:
    roots, single_input = normalize_submit_input(obj_or_objs)
    graph = discover_missing_closure(roots)
    return executor.submit(
        roots=roots,
        graph=graph,
        single_input=single_input,
    )


def normalize_submit_input[T](
    obj_or_objs: Furu[T] | Sequence[Furu[T]],
) -> tuple[tuple[Furu[T], ...], bool]:
    if isinstance(obj_or_objs, Furu):
        return (obj_or_objs,), True
    if not isinstance(obj_or_objs, Sequence):
        raise TypeError("submit() expected a Furu object or a sequence of Furu objects")
    roots = tuple(obj_or_objs)
    if any(not isinstance(root, Furu) for root in roots):
        raise TypeError("submit() expected Furu objects")
    if not roots:
        raise ValueError("submit() requires at least one Furu object")
    return roots, False
