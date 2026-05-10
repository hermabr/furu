from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Self, cast

from furu.utils import fully_qualified_name


class ResultCodec(ABC):
    @classmethod
    def _codec_id(cls) -> str:
        return fully_qualified_name(cls)

    @classmethod
    def dependencies_available(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def matches(cls, value: object) -> bool:
        pass

    @classmethod
    @abstractmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *, artifact_dir: Path) -> object:
        pass


class PolarsParquetCodec(ResultCodec):
    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("polars") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import polars as pl

        return isinstance(value, pl.DataFrame)

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
    ) -> None:
        import polars as pl

        cast(pl.DataFrame, value).write_parquet(artifact_dir / "data.parquet")

    @classmethod
    def load(cls, *, artifact_dir: Path) -> object:
        import polars as pl

        return pl.read_parquet(artifact_dir / "data.parquet")


class NumpyNpyCodec(ResultCodec):
    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("numpy") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import numpy as np

        return isinstance(value, np.ndarray)

    @classmethod
    def dump(
        cls,
        value: object,
        *,
        artifact_dir: Path,
    ) -> None:
        import numpy as np

        np.save(artifact_dir / "data.npy", cast(np.ndarray, value), allow_pickle=False)

    @classmethod
    def load(cls, *, artifact_dir: Path) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


@dataclass(frozen=True)
class ResultRegistry:
    codecs: tuple[type[ResultCodec], ...] = ()

    def register(self, codec: type[ResultCodec]) -> Self:
        return type(self)(codecs=(codec, *self.codecs))

    def find_codec(self, value: object) -> type[ResultCodec] | None:
        for codec in self.codecs:
            if codec.matches(value):
                return codec
        return None


@cache
def resolve_result_codec(codec_id: str) -> type[ResultCodec]:
    module_name, _, attr_name = codec_id.rpartition(".")
    if not module_name:
        raise KeyError(codec_id)

    codec = getattr(importlib.import_module(module_name), attr_name)
    if not issubclass(codec, ResultCodec):
        raise TypeError(f"{codec_id} is not a ResultCodec")
    return codec


@cache
def _default_result_registry() -> ResultRegistry:
    return ResultRegistry(
        codecs=tuple(
            codec
            for codec in (PolarsParquetCodec, NumpyNpyCodec)
            if codec.dependencies_available()
        ),
    )
