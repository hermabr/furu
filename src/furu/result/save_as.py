from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from furu.result.codec import ResultCodec


@dataclass(frozen=True)
class _SaveAs[T]:
    value: T
    codec: type[ResultCodec]


def save_as[T](value: T, *, codec: type[ResultCodec]) -> T:
    return cast(T, _SaveAs(value=value, codec=codec))
