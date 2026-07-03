from __future__ import annotations

import importlib.util
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

from furu.utils import fully_qualified_name


class CodecMeta(ABCMeta):
    _auto_registered_codecs: list[type[Codec]] = []
    _auto_registered_codecs_lock = Lock()

    def __init__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        is_root_codec_class = not any(isinstance(base, CodecMeta) for base in bases)
        if is_root_codec_class:
            return
        if not namespace.get("auto_register", True):
            return
        if getattr(cls, "__abstractmethods__", None):
            return

        cls.auto_register = True
        with CodecMeta._auto_registered_codecs_lock:
            CodecMeta._auto_registered_codecs.append(cast(type[Codec], cls))

        CodecMeta._default_codec_layers.cache_clear()

    @classmethod
    def auto_registered_codecs(mcls) -> tuple[type[Codec], ...]:
        with mcls._auto_registered_codecs_lock:
            return tuple(reversed(mcls._auto_registered_codecs))

    @classmethod
    @cache
    def _default_codec_layers(
        mcls,
    ) -> tuple[tuple[type[Codec], ...], tuple[type[Codec], ...]]:
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
        result_codecs: tuple[type[Codec], ...],
    ) -> type[Codec] | None:
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


class Codec[T](ABC, metaclass=CodecMeta):
    """Persists one heavy value under an artifact directory.

    ``save`` writes the value's bytes under ``artifact_dir`` and returns a
    metadata mapping that the runtime stores inline in the wrapper JSON; any
    ``Path`` in that mapping is recorded relative to the run's data dir and
    handed back resolved to an absolute path in ``load``. ``matches`` stays a
    classmethod so the registry can dispatch on a value without constructing a
    codec.
    """

    auto_register: ClassVar[bool] = True
    reload_value_after_save: ClassVar[bool] = False

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

    @abstractmethod
    def save(self, value: T, artifact_dir: Path) -> Mapping[str, object]:
        pass

    @abstractmethod
    def load(self, metadata: Mapping[str, object], artifact_dir: Path) -> T:
        pass


class PolarsParquetCodec(Codec["pl.DataFrame"]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("polars") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import polars as pl

        return isinstance(value, pl.DataFrame)

    def save(self, value: pl.DataFrame, artifact_dir: Path) -> Mapping[str, object]:
        value.write_parquet(artifact_dir / "data.parquet")
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_dir: Path
    ) -> pl.DataFrame:
        import polars as pl

        return pl.read_parquet(artifact_dir / "data.parquet")


class NumpyNpyCodec(Codec["np.ndarray[Any, Any]"]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("numpy") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import numpy as np

        return isinstance(value, np.ndarray)

    def save(
        self, value: np.ndarray[Any, Any], artifact_dir: Path
    ) -> Mapping[str, object]:
        import numpy as np

        np.save(artifact_dir / "data.npy", value, allow_pickle=False)
        return {}

    def load(
        self, metadata: Mapping[str, object], artifact_dir: Path
    ) -> np.ndarray[Any, Any]:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


def _find_single_codec_match(
    value: object,
    codecs: tuple[type[Codec], ...],
    *,
    layer_name: str,
) -> type[Codec] | None:
    matching_codec: type[Codec] | None = None
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
