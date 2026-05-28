from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

from furu.utils import fully_qualified_name


class ResultCodec[T](ABC):
    load_after_dump: ClassVar[bool] = False

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
        value: T,
        *,
        artifact_dir: Path,
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *, artifact_dir: Path) -> T:
        pass


class PolarsParquetCodec(ResultCodec["pl.DataFrame"]):
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
        value: pl.DataFrame,
        *,
        artifact_dir: Path,
    ) -> None:
        value.write_parquet(artifact_dir / "data.parquet")

    @classmethod
    def load(cls, *, artifact_dir: Path) -> pl.DataFrame:
        import polars as pl

        return pl.read_parquet(artifact_dir / "data.parquet")


class NumpyNpyCodec(ResultCodec["np.ndarray[Any, Any]"]):
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
        value: np.ndarray[Any, Any],
        *,
        artifact_dir: Path,
    ) -> None:
        import numpy as np

        np.save(artifact_dir / "data.npy", value, allow_pickle=False)

    @classmethod
    def load(cls, *, artifact_dir: Path) -> np.ndarray[Any, Any]:
        import numpy as np

        return cast(
            np.ndarray[Any, Any], np.load(artifact_dir / "data.npy", allow_pickle=False)
        )


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
def _default_result_registry() -> ResultRegistry:
    return ResultRegistry(
        codecs=tuple(
            codec
            for codec in (PolarsParquetCodec, NumpyNpyCodec)
            if codec.dependencies_available()
        ),
    )
