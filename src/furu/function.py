from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar, Protocol, cast, get_type_hints, overload

from furu.core import Furu


class FuruFunction[**P, T](Protocol):
    make_furu_obj: Callable[P, Furu[T]]
    furu_type: Any

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


@overload
def function[**P, T](func: Callable[P, T]) -> FuruFunction[P, T]: ...


@overload
def function[**P, T]() -> Callable[[Callable[P, T]], FuruFunction[P, T]]: ...


def function[**P, T](
    func: Callable[P, T] | None = None,
) -> FuruFunction[P, T] | Callable[[Callable[P, T]], FuruFunction[P, T]]:
    def decorator(inner: Callable[P, T]) -> FuruFunction[P, T]:
        cls = _function_type(inner)
        return _function_wrapper(inner, cls)

    if func is None:
        return decorator
    return decorator(func)


def _function_type[**P, T](func: Callable[P, T]) -> type[Furu[T]]:
    func_name = getattr(func, "__name__", None)
    func_qualname = getattr(func, "__qualname__", None)
    func_module = getattr(func, "__module__", None)
    func_doc = getattr(func, "__doc__", None)
    if not (
        isinstance(func_name, str)
        and isinstance(func_qualname, str)
        and isinstance(func_module, str)
    ):
        raise TypeError("@furu.function expects a named function")

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
                f"@furu.function does not support variadic parameter "
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

    cls = cast(
        type[Furu[T]],
        types.new_class(
            func_name,
            (Furu[result_type],),
            exec_body=exec_body,
        ),
    )
    setattr(cls, "make_furu_obj", cls)
    setattr(cls, "furu_type", cls)
    return cls


def _function_wrapper[**P, T](
    func: Callable[P, T], cls: type[Furu[T]]
) -> FuruFunction[P, T]:
    make_furu_obj = cast(Callable[P, Furu[T]], cls)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return make_furu_obj(*args, **kwargs).create()

    setattr(wrapper, "make_furu_obj", make_furu_obj)
    setattr(wrapper, "furu_type", cls)
    return cast(FuruFunction[P, T], wrapper)
