from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from typing import Any, ClassVar, cast, get_type_hints

from furu.core import Furu


def furu_method[**P, T](func: Callable[P, T]) -> Callable[P, Furu[T]]:
    func_name = getattr(func, "__name__", None)
    func_qualname = getattr(func, "__qualname__", None)
    func_module = getattr(func, "__module__", None)
    func_doc = getattr(func, "__doc__", None)
    if not (
        isinstance(func_name, str)
        and isinstance(func_qualname, str)
        and isinstance(func_module, str)
    ):
        raise TypeError("@furu_method expects a named function")

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    annotations: dict[str, object] = {}
    field_names: list[str] = []
    positional_only_names: list[str] = []
    keyword_names: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(
                f"@furu_method does not support variadic parameter "
                f"{func_qualname}.{parameter.name}"
            )

        field_names.append(parameter.name)
        annotations[parameter.name] = type_hints.get(parameter.name, Any)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only_names.append(parameter.name)
        else:
            keyword_names.append(parameter.name)

    result_type = type_hints.get("return", Any)

    def __init__(self: Furu[T], *args: object, **kwargs: object) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name in field_names:
            object.__setattr__(self, name, bound.arguments[name])
        if post_init := getattr(self, "__post_init__", None):
            post_init()

    def create(self: Furu[T]) -> T:
        args = [getattr(self, name) for name in positional_only_names]
        kwargs = {name: getattr(self, name) for name in keyword_names}
        return func(*args, **kwargs)

    create.__name__ = "create"
    create.__qualname__ = f"{func_qualname}.create"
    create.__doc__ = func_doc
    create.__module__ = func_module

    def exec_body(namespace: dict[str, object]) -> None:
        namespace["__annotations__"] = {
            **annotations,
            "__furu_wrapped_function__": ClassVar[Callable[P, T]],
        }
        namespace["__module__"] = func_module
        namespace["__qualname__"] = func_qualname
        namespace["__doc__"] = func_doc
        namespace["__init__"] = __init__
        namespace["__signature__"] = signature
        namespace["__wrapped__"] = func
        namespace["__furu_wrapped_function__"] = func
        namespace["create"] = create
        for parameter in signature.parameters.values():
            if parameter.default is not inspect.Parameter.empty:
                namespace[parameter.name] = parameter.default

    cls = types.new_class(
        func_name,
        (Furu[result_type],),
        exec_body=exec_body,
    )
    return cast(Callable[P, Furu[T]], cls)
