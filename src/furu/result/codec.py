from __future__ import annotations

import importlib.util
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, final

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

from furu.utils import fully_qualified_name


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

        registry_cls = globals().get("ResultRegistry")
        if registry_cls is not None:
            registry_cls.default.cache_clear()

    @classmethod
    def auto_registered_codecs(mcls) -> tuple[type[ResultCodec], ...]:
        with mcls._auto_registered_codecs_lock:
            return tuple(reversed(mcls._auto_registered_codecs))


class ResultCodec[T](ABC, metaclass=ResultCodecMeta):
    auto_register: ClassVar[bool] = True
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
        artifact_dir: Path,
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *, artifact_dir: Path) -> T:
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


@dataclass(frozen=True)
class ResultRegistry:
    explicit_codecs: tuple[type[ResultCodec], ...] = ()
    auto_registered_codecs: tuple[type[ResultCodec], ...] = ()
    built_in_codecs: tuple[type[ResultCodec], ...] = ()

    def with_codec(self, codec: type[ResultCodec]) -> Self:
        return type(self)(
            explicit_codecs=(codec, *self.explicit_codecs),
            auto_registered_codecs=self.auto_registered_codecs,
            built_in_codecs=self.built_in_codecs,
        )

    def find_codec(self, value: object) -> type[ResultCodec] | None:
        if codec := _find_single_codec_match(
            value,
            self.explicit_codecs,
            layer_name="explicit codec registry",
        ):
            return codec
        if codec := _find_single_codec_match(
            value,
            self.auto_registered_codecs,
            layer_name="auto-registered codec registry",
        ):
            return codec
        for codec in self.built_in_codecs:
            if codec.matches(value):
                return codec
        return None

    @classmethod
    @cache
    def default(cls) -> ResultRegistry:
        auto_registered_codecs = tuple(
            codec
            for codec in ResultCodecMeta.auto_registered_codecs()
            if codec.dependencies_available()
        )
        built_in_codecs = tuple(
            codec
            for codec in (PolarsParquetCodec, NumpyNpyCodec)
            if codec.dependencies_available()
        )
        return cls(
            auto_registered_codecs=auto_registered_codecs,
            built_in_codecs=built_in_codecs,
        )


def _find_single_codec_match(
    value: object,
    codecs: tuple[type[ResultCodec], ...],
    *,
    layer_name: str,
) -> type[ResultCodec] | None:
    matching_codec: type[ResultCodec] | None = None
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
