from __future__ import annotations

import importlib.util
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from functools import cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

from furu.utils import fully_qualified_name


class _ResultCodec[T](ABC):
    reload_value_after_dump: ClassVar[bool] = False

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


class ResultCodecMeta(ABCMeta):
    _auto_registered_codecs: list[type[ResultCodec]] = []
    _auto_registered_codecs_lock = Lock()

    def __init__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        is_root_codec_class = not any(
            isinstance(base, ResultCodecMeta) for base in bases
        )
        if is_root_codec_class:
            return
        if not namespace.get("auto_register", True):
            return
        if getattr(cls, "__abstractmethods__", None):
            return

        cls.auto_register = True
        with ResultCodecMeta._auto_registered_codecs_lock:
            ResultCodecMeta._auto_registered_codecs.append(cast(type[ResultCodec], cls))

        ResultCodecMeta._default_codec_layers.cache_clear()

    @classmethod
    def auto_registered_codecs(mcls) -> tuple[type[ResultCodec], ...]:
        with mcls._auto_registered_codecs_lock:
            return tuple(reversed(mcls._auto_registered_codecs))

    @classmethod
    @cache
    def _default_codec_layers(
        mcls,
    ) -> tuple[tuple[type[ResultCodec], ...], tuple[type[ResultCodec], ...]]:
        auto_registered_codecs = tuple(
            codec
            for codec in mcls.auto_registered_codecs()
            if codec.dependencies_available()
        )
        built_in_codecs = tuple(
            codec
            for codec in (PolarsParquetCodec, NumpyNpyCodec)
            if codec.dependencies_available()
        )
        return auto_registered_codecs, built_in_codecs

    @classmethod
    def find_codec(
        mcls,
        value: object,
        result_codecs: tuple[type[_ResultCodec], ...],
    ) -> type[_ResultCodec] | None:
        auto_registered_codecs, built_in_codecs = mcls._default_codec_layers()
        if codec := _find_single_codec_match(
            value,
            result_codecs,
            layer_name="result codecs",
        ):
            return codec
        if codec := _find_single_codec_match(
            value,
            auto_registered_codecs,
            layer_name="auto-registered codec registry",
        ):
            return codec
        for codec in built_in_codecs:
            if codec.matches(value):
                return codec
        return None


class ResultCodec[T](_ResultCodec[T], metaclass=ResultCodecMeta):
    auto_register: ClassVar[bool] = True

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


class DataDirResultCodec[T](_ResultCodec[T]):
    @classmethod
    @abstractmethod
    def dump(
        cls,
        value: T,
        *,
        artifact_dir: Path,
        path_relative_to_data_dir: Callable[[Path], str],
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        *,
        artifact_dir: Path,
        path_in_data_dir: Callable[[str], Path],
    ) -> T:
        pass


class PolarsParquetCodec(ResultCodec["pl.DataFrame"]):
    auto_register: ClassVar[bool] = False

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
    auto_register: ClassVar[bool] = False

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

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


def _find_single_codec_match(
    value: object,
    codecs: tuple[type[_ResultCodec], ...],
    *,
    layer_name: str,
) -> type[_ResultCodec] | None:
    matching_codec: type[_ResultCodec] | None = None
    for codec in codecs:
        if not codec.matches(value):
            continue
        if matching_codec is None:
            matching_codec = codec
            continue

        raise TypeError(
            f"{layer_name} matched multiple codecs: "
            f"{matching_codec.__name__}, {codec.__name__}"
        )
    return matching_codec
