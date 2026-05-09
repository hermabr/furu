from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final, cast

from furu.utils import fully_qualified_name


class ResultCodec(ABC):
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "codec_id" in cls.__dict__:
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__} must not override codec_id; "
                "codec ids are derived from codec class identity"
            )

    @classmethod
    def codec_id(cls) -> str:
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


_DEFAULT_CODECS: Final[dict[str, type[ResultCodec]]] = {
    c.codec_id(): c
    for c in [PolarsParquetCodec, NumpyNpyCodec]
    if c.dependencies_available()
}
