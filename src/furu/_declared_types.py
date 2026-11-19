from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from functools import cache
from itertools import chain, repeat
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from furu.core import Spec


@cache
def cached_type_hints(cls: type) -> Mapping[str, Any]:
    return get_type_hints(cls, include_extras=True)


def strip_annotated(declared_type: object) -> object:
    if get_origin(declared_type) is Annotated:
        return get_args(declared_type)[0]
    return declared_type


def child_declared_type(declared_type: object, key: object) -> object:
    declared_type = strip_annotated(declared_type)
    origin = get_origin(declared_type)
    args = get_args(declared_type)
    if origin is list and args:
        return args[0]
    if origin in (set, frozenset) and args:
        return args[0]
    if origin is dict and len(args) == 2:
        return args[1]
    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        if isinstance(key, int) and key < len(args):
            return args[key]
    return Any


def item_declared_types(declared_type: object) -> Iterator[object]:
    """The declared type of each item of a list or tuple, in order.

    Worked out once per container, not per item: one tuple can hold thousands
    of specs.
    """
    declared_type = strip_annotated(declared_type)
    args = get_args(declared_type)
    if get_origin(declared_type) is tuple and args and args[1:] != (Ellipsis,):
        return chain(args, repeat(Any))
    return repeat(child_declared_type(declared_type, 0))


@cache
def declared_result_type(spec_type: type[Spec]) -> object:
    """The result type ``create()`` is annotated to return; for a batched one, an item's.

    Read when the first result is stored, after the module that defines the
    class has run to its end, so the annotation may name a class defined
    below it, or a ``Result`` class nested in the spec.
    """
    from furu._batched import _BatchedHook

    owner = next(
        (c for c in spec_type.__mro__ if "_furu_create_hook" in c.__dict__), None
    )
    if owner is None:
        return Any
    hook = owner.__dict__["_furu_create_hook"]
    func = hook.func if isinstance(hook, _BatchedHook) else hook
    # The class's own names: a nested Result, and the T of class Base[T].
    names = {param.__name__: param for param in owner.__type_params__}
    declared = get_type_hints(
        func,
        globalns=sys.modules[owner.__module__].__dict__,
        localns={**names, **vars(owner)},
        include_extras=True,
    ).get("return", Any)
    if isinstance(hook, _BatchedHook):
        declared = child_declared_type(declared, 0)
    if _has_type_var(declared):
        raise TypeError(
            f"{spec_type.__name__}.create() must be annotated with a concrete "
            f"result type, not {declared}"
        )
    return declared


def _has_type_var(declared: object) -> bool:
    return isinstance(declared, TypeVar) or any(map(_has_type_var, get_args(declared)))
