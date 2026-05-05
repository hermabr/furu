from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from furu.utils import fully_qualified_name


class ResultCodec(ABC):
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

        assert isinstance(value, np.ndarray), (
            f"NumpyNpyCodec.dump expected np.ndarray, got {type(value).__name__}"
        )
        np.save(artifact_dir / "data.npy", value, allow_pickle=False)

    @classmethod
    def load(cls, *, artifact_dir: Path) -> object:
        import numpy as np

        return np.load(artifact_dir / "data.npy", allow_pickle=False)


_DEFAULT_CODECS: Final[dict[str, type[ResultCodec]]] = {
    c.codec_id(): c for c in [NumpyNpyCodec] if c.dependencies_available()
}
