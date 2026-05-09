from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, cast

import pydantic

from furu.result.codec import ResultCodec


@dataclass(frozen=True)
class _SaveAs[T]:
    value: T
    codec: type[ResultCodec]


def save_as[T](value: T, *, codec: type[ResultCodec]) -> T:
    return cast(T, _SaveAs(value=value, codec=codec))


def _unwrap_save_as[T](value: T) -> T:
    match value:
        case _SaveAs(value=inner):
            return cast(T, _unwrap_save_as(inner))
        case list():
            return cast(T, [_unwrap_save_as(item) for item in value])
        case tuple():
            return cast(T, tuple(_unwrap_save_as(item) for item in value))
        case set():
            return cast(T, {_unwrap_save_as(item) for item in value})
        case frozenset():
            return cast(T, frozenset(_unwrap_save_as(item) for item in value))
        case dict():
            return cast(
                T, {key: _unwrap_save_as(child) for key, child in value.items()}
            )
        case pydantic.BaseModel():
            return cast(
                T,
                value.model_copy(
                    update={
                        name: _unwrap_save_as(getattr(value, name))
                        for name in value.__class__.model_fields
                    }
                ),
            )
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.replace(
                value,
                **{
                    field.name: _unwrap_save_as(getattr(value, field.name))
                    for field in dataclasses.fields(cast(Any, value))
                    if field.init
                },
            )
        case _:
            return value
