from __future__ import annotations

import importlib.util
import io
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cache
from itertools import accumulate
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

from furu.utils import fully_qualified_name


class CodecMeta(ABCMeta):
    _auto_registered_codecs: ClassVar[list[type[Codec]]] = []
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
    auto_register: ClassVar[bool] = True

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
    def save(cls, value: T, artifact_directory: Path) -> Mapping[str, object]:
        pass

    @classmethod
    @abstractmethod
    def load(cls, metadata: Mapping[str, object], artifact_directory: Path) -> T:
        pass

    @classmethod
    def save_many(
        cls, values: Sequence[T], artifact_directory: Path
    ) -> Sequence[Mapping[str, object]] | None:
        """Save the values of many results into one directory.

        Returns one metadata per value, in order. Returning None, before
        writing anything, has each value saved by save() into a directory of
        its own.
        """
        return None

    @classmethod
    def load_many(
        cls, metadatas: Sequence[Mapping[str, object]], artifact_directory: Path
    ) -> Sequence[T]:
        """Load values that save_many() saved, one per metadata, in order."""
        return [cls.load(metadata, artifact_directory) for metadata in metadatas]


class PolarsParquetCodec(Codec["pl.DataFrame"]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("polars") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import polars as pl

        return isinstance(value, pl.DataFrame)

    @classmethod
    def save(
        cls, value: pl.DataFrame, artifact_directory: Path
    ) -> Mapping[str, object]:
        value.write_parquet(artifact_directory / "data.parquet")
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> pl.DataFrame:
        import polars as pl

        return pl.read_parquet(artifact_directory / "data.parquet")

    @classmethod
    def save_many(
        cls, values: Sequence[pl.DataFrame], artifact_directory: Path
    ) -> Sequence[Mapping[str, object]] | None:
        import polars as pl

        schema = values[0].schema
        if any(value.schema != schema for value in values):
            return None
        # Small row groups let a read of a few tables skip the rest of the file.
        pl.concat(values).write_parquet(
            artifact_directory / "data.parquet", row_group_size=16_384
        )
        offsets = accumulate((value.height for value in values), initial=0)
        return [
            {"offset": offset, "length": value.height}
            for offset, value in zip(offsets, values)
        ]

    @classmethod
    def load_many(
        cls, metadatas: Sequence[Mapping[str, object]], artifact_directory: Path
    ) -> Sequence[pl.DataFrame]:
        import polars as pl

        spans = [
            (cast(int, metadata["offset"]), cast(int, metadata["length"]))
            for metadata in metadatas
        ]
        start = min(offset for offset, _ in spans)
        end = max(offset + length for offset, length in spans)
        table = (
            pl.scan_parquet(artifact_directory / "data.parquet")
            .slice(start, end - start)
            .collect()
        )
        return [table.slice(offset - start, length) for offset, length in spans]


class NumpyNpyCodec(Codec["np.ndarray[Any, Any]"]):
    auto_register: ClassVar[bool] = False

    @classmethod
    def dependencies_available(cls) -> bool:
        return importlib.util.find_spec("numpy") is not None

    @classmethod
    def matches(cls, value: object) -> bool:
        import numpy as np

        return isinstance(value, np.ndarray)

    @classmethod
    def save(
        cls, value: np.ndarray[Any, Any], artifact_directory: Path
    ) -> Mapping[str, object]:
        import numpy as np

        np.save(artifact_directory / "data.npy", value, allow_pickle=False)
        return {}

    @classmethod
    def load(
        cls, metadata: Mapping[str, object], artifact_directory: Path
    ) -> np.ndarray[Any, Any]:
        import numpy as np

        return np.load(artifact_directory / "data.npy", allow_pickle=False)

    @classmethod
    def save_many(
        cls, values: Sequence[np.ndarray[Any, Any]], artifact_directory: Path
    ) -> Sequence[Mapping[str, object]] | None:
        import numpy as np

        first = values[0]
        if first.dtype.hasobject or any(
            value.shape != first.shape
            or value.dtype != first.dtype
            # Its mask would be dropped.
            or isinstance(value, np.ma.MaskedArray)
            for value in values
        ):
            return None
        header = np.lib.format.header_data_from_array_1_0(first)
        header |= {"shape": (len(values), *first.shape), "fortran_order": False}
        encoded = io.BytesIO()
        try:
            np.lib.format.write_array_header_1_0(encoded, header)
        except ValueError:
            # A field name outside latin-1, or a header over 64 KiB, needs a
            # later format version, which only np.save() picks.
            return None
        # Written row by row, so the batch is never copied into one array.
        with open(artifact_directory / "data.npy", "wb") as file:
            file.write(encoded.getvalue())
            file.writelines(np.asarray(value).tobytes() for value in values)
        return [{"row": row} for row in range(len(values))]

    @classmethod
    def load_many(
        cls, metadatas: Sequence[Mapping[str, object]], artifact_directory: Path
    ) -> Sequence[np.ndarray[Any, Any]]:
        import numpy as np

        stacked = np.load(
            artifact_directory / "data.npy", mmap_mode="r", allow_pickle=False
        )
        # The ellipsis keeps a 0-d row an array, with its dtype.
        return [
            np.array(stacked[cast(int, metadata["row"]), ...]) for metadata in metadatas
        ]


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
