from __future__ import annotations

from typing import Any, cast

from furu.results.protocol import FuruResult
from furu.results.registry import ResultCodec
from furu.results.rules import SaveSpec, SaveWith, _WrappedValue, at, when_type


def save_with[T](
    value: T,
    *,
    serializer: str | ResultCodec[Any] | None = None,
    lazy: bool | None = None,
) -> T:
    return cast(
        T, _WrappedValue(value=value, spec=SaveSpec(serializer=serializer, lazy=lazy))
    )


def lazy[T](
    value: T,
    *,
    serializer: str | ResultCodec[Any] | None = None,
) -> T:
    return save_with(value, serializer=serializer, lazy=True)


__all__ = ["FuruResult", "SaveWith", "at", "lazy", "save_with", "when_type"]
