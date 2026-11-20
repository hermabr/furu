from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import TYPE_CHECKING, Any, NamedTuple, overload

from furu.resources import Worker
from furu.serializer.artifact import to_json
from furu.spec_metadata import Metadata
from furu.storage.directory import SpecDirectory
from furu.utils import _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec

type BatchFn = Callable[[Any, Worker], tuple[object, int]]


class _BatchedHook(NamedTuple):
    func: Callable[[list[Any], SpecDirectory], list[Any]]
    batch_fn: BatchFn


def is_batched(cls: type) -> bool:
    return isinstance(getattr(cls, "_furu_create_hook", None), _BatchedHook)


def batch_of(
    obj: Spec, worker: Worker, firsts: dict[tuple[Worker, str], Spec]
) -> tuple[str, int]:
    """The batch ``obj`` joins on ``worker``, by name, and how many specs it may hold.

    The name is the class, the batch key written as a field is in a plan
    line, and the cap, so two launches give the same batch the same name. A
    key that a field could not hold is refused. A spec without
    ``@furu.batched`` runs alone: a batch of one.

    A batch is one create() call in one process that publishes one folder,
    so two specs with one batch key and different Metadata stop the launch.
    ``firsts`` holds the first spec seen in each batch on each worker.
    """
    hook = getattr(type(obj), "_furu_create_hook", None)
    if not isinstance(hook, _BatchedHook):
        return obj._stored_name, 1
    key, cap = hook.batch_fn(obj, worker)
    if type(cap) is not int or cap < 1:
        raise TypeError(
            f"{type(obj).__qualname__} batch key cap must be a positive int, "
            f"got {cap!r} on {worker}"
        )
    written = _stable_json_dump(to_json(key, object, (), set()))
    name = f"{obj._stored_name}:{written}:{cap}"
    first = firsts.setdefault((worker, name), obj)
    if differs := [
        f"{f.name} ({getattr(first._metadata, f.name)!r} and "
        f"{getattr(obj._metadata, f.name)!r})"
        for f in fields(Metadata)
        if getattr(first._metadata, f.name) != getattr(obj._metadata, f.name)
    ]:
        raise ValueError(
            f"{first.object_id} and {obj.object_id} have the same batch "
            f"key and differ in Metadata {', '.join(differs)}. A batch is "
            "one create() call that publishes one folder, so it has one "
            "Metadata: give the specs different batch keys or the same "
            "Metadata."
        )
    return name, cap


class batched:
    """Run create() on groups of specs sharing a batch key.

    ``batch_fn(spec, worker)`` returns ``(key, cap)``; ``worker`` is the
    leasing worker (or this machine for in-process create), so the cap can
    follow its resources, e.g. ``(spec.model, worker.gpus)``. A key may be
    any value a field may be, and two keys are the same batch when they are
    written the same, as two fields are.

    The decorated ``create(specs, directory)`` gets the batch's one
    ``data/`` and ``scratch/`` as its second argument. A batch is stored as
    one folder with a row per spec, so name the files you write per spec by
    its fields, never by its row: the same spec can land in another batch.
    """

    def __init__(self, batch_fn: BatchFn, /) -> None:
        if not callable(batch_fn):
            raise TypeError(
                "@furu.batched needs a batch key function: @furu.batched(batch_key)"
            )
        self.batch_fn = batch_fn

    def __call__[S, T](
        self, func: Callable[[list[S], SpecDirectory], list[T]], /
    ) -> _BatchedCreate[S, T]:
        if getattr(func, "__name__", None) != "create":
            raise TypeError("@furu.batched can only decorate create()")
        return _BatchedCreate(func, self.batch_fn)


class _BatchedCreate[S, T](NamedTuple):
    func: Callable[[list[S], SpecDirectory], list[T]]
    batch_fn: BatchFn

    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: None, _objtype: type, /) -> Callable[[S], T]: ...

        @overload
        def __get__(self, obj: S, _objtype: type, /) -> Callable[[], T]: ...
