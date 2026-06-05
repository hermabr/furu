from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, final

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

from furu.config import get_config
from furu.utils import fully_qualified_name, resolve_fully_qualified_name


@dataclass(frozen=True, slots=True, init=False)
class ResultCodecContext:
    """Runtime paths and validation helpers available to result codecs."""

    artifact_dir: Path
    _data_dir: Path = field(repr=False)

    def __init__(self, *, artifact_dir: Path, data_dir: Path) -> None:
        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "_data_dir", data_dir)

    def relative_to_data_dir(self, path: Path) -> Path:
        """Return a data-dir-relative path, rejecting paths outside the data dir."""
        try:
            return path.resolve().relative_to(self._data_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"result codec path must be inside the Furu data dir: {path}"
            ) from exc

    def data_dir_path(self, relative_path: str | Path) -> Path:
        """Resolve a data-dir-relative path, rejecting absolute or escaping paths."""
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(
                f"result codec data-relative path must be relative: {path}"
            )

        resolved = (self._data_dir / path).resolve()
        if not resolved.is_relative_to(self._data_dir.resolve()):
            raise ValueError(
                f"result codec data-relative path escapes data dir: {path}"
            )
        return resolved


class ResultCodec[T](ABC):
    load_after_dump: ClassVar[bool] = False

    @final
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
        context: ResultCodecContext,
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *, context: ResultCodecContext) -> T:
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
        context: ResultCodecContext,
    ) -> None:
        value.write_parquet(context.artifact_dir / "data.parquet")

    @classmethod
    def load(cls, *, context: ResultCodecContext) -> pl.DataFrame:
        import polars as pl

        return pl.read_parquet(context.artifact_dir / "data.parquet")


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
        context: ResultCodecContext,
    ) -> None:
        import numpy as np

        np.save(context.artifact_dir / "data.npy", value, allow_pickle=False)

    @classmethod
    def load(cls, *, context: ResultCodecContext) -> np.ndarray[Any, Any]:
        import numpy as np

        return np.load(context.artifact_dir / "data.npy", allow_pickle=False)


@dataclass(frozen=True)
class ResultRegistry:
    codecs: tuple[type[ResultCodec], ...] = ()

    def with_codec(self, codec: type[ResultCodec]) -> Self:
        return type(self)(codecs=(codec, *self.codecs))

    def find_codec(self, value: object) -> type[ResultCodec] | None:
        for codec in self.codecs:
            if codec.matches(value):
                return codec
        return None

    @classmethod
    @cache
    def default(cls) -> ResultRegistry:
        configured_codecs: list[type[ResultCodec]] = []
        for codec_id in get_config().result.codecs:
            codec = resolve_fully_qualified_name(codec_id)
            if not isinstance(codec, type) or not issubclass(codec, ResultCodec):
                raise TypeError(
                    f"Configured result codec {codec_id!r} is not a ResultCodec"
                )
            if codec.dependencies_available():
                configured_codecs.append(codec)
        built_in_codecs = tuple(
            codec
            for codec in (PolarsParquetCodec, NumpyNpyCodec)
            if codec.dependencies_available()
        )
        return cls(codecs=(*configured_codecs, *built_in_codecs))
