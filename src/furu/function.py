from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar, Protocol, cast, get_type_hints, overload

from furu.core import Spec


class SpecFunction[**P, T](Protocol):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Spec[T]: ...


@overload
def spec[**P, T](func: Callable[P, T]) -> SpecFunction[P, T]: ...

@overload
def spec[**P, T]() -> Callable[[Callable[P, T]], SpecFunction[P, T]]: ...


def spec[**P, T](
    func: Callable[P, T] | None = None,
) -> SpecFunction[P, T] | Callable[[Callable[P, T]], SpecFunction[P, T]]:
    def decorator(inner: Callable[P, T]) -> SpecFunction[P, T]:
        cls = _function_type(inner)
        return _function_wrapper(inner, cls)

    if func is None:
        return decorator
    return decorator(func)


def _function_type[**P, T](func: Callable[P, T]) -> type[Spec[T]]:
    func_name = getattr(func, "__name__", None)
    func_qualname = getattr(func, "__qualname__", None)
    func_module = getattr(func, "__module__", None)
    func_doc = getattr(func, "__doc__", None)
    if not (
        isinstance(func_name, str)
        and isinstance(func_qualname, str)
        and isinstance(func_module, str)
    ):
        raise TypeError("@furu.spec expects a named function")

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
                f"@furu.spec does not support variadic parameter "
                f"{func_qualname}.{parameter.name}"
            )

        field_names.append(parameter.name)
        annotations[parameter.name] = type_hints.get(parameter.name, Any)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only_names.append(parameter.name)
        else:
            keyword_names.append(parameter.name)

    result_type = type_hints.get("return", Any)

    def __init__(self: Spec[T], *args: object, **kwargs: object) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name in field_names:
            object.__setattr__(self, name, bound.arguments[name])
        if post_init := getattr(self, "__post_init__", None):
            post_init()

    def create(self: Spec[T]) -> T:
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
        type[Spec[T]],
        types.new_class(
            func_name,
            (Spec[result_type],),
            exec_body=exec_body,
        ),
    )
    return cls


def _function_wrapper[**P, T](
    func: Callable[P, T], cls: type[Spec[T]]
) -> SpecFunction[P, T]:
    spec_cls = cast(Callable[P, Spec[T]], cls)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Spec[T]:
        return spec_cls(*args, **kwargs)

    setattr(wrapper, "__furu_spec_type__", cls)
    return cast(SpecFunction[P, T], wrapper)
