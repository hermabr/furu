import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
)

from furu.config import config
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import JsonValue, _hash_dict_deterministically, fully_qualified_name

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    @dataclass_transform(kw_only_default=True, frozen_default=True)
    class _FuruDataclassTransform:
        pass
else:

    class _FuruDataclassTransform:
        pass


class Furu[T](_FuruDataclassTransform, ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Furu:
            return
        if "__dataclass_params__" not in cls.__dict__:
            dataclass(frozen=True, kw_only=True)(cls)

    def load_or_create(self) -> T:
        if self._result_path.exists():
            # TODO: validation that its up to date and valid
            with open(self._result_path, "rb") as f:
                return pickle.load(f)

        # TODO: initialize the state
        # TODO: wrap this in a try/catch
        # TODO: add file locking and keepalive in a different process
        self._internal_furu_dir.mkdir(exist_ok=True, parents=True)
        result = self._create()

        if (
            tmp_result_path := self._result_path.with_suffix(".tmp.pkl")
        ).exists():  # clean up old tmp path
            tmp_result_path.unlink()

        with open(tmp_result_path, "wb") as f:
            # TODO: maybe it would be more correct to dump to a _tmp file and then rename?
            pickle.dump(result, f)
            # TODO: do i need f.flush and os.fsync?
        os.replace(tmp_result_path, self._result_path)

        return result

    def exists(self) -> bool:
        return self._result_path.exists()

    def try_load(self) -> T:
        if self._result_path.exists():
            # TODO: validation that its up to date and valid
            with open(self._result_path, "rb") as f:
                return pickle.load(f)
        raise NotImplementedError(
            "TODO: decide if i should throw or return error value"
        )

    @property
    def _result_path(self) -> Path:
        return self.data_dir / "result.pkl"

    @abstractmethod
    def _create(self) -> T:
        raise NotImplementedError("TODO")

    @cache
    def to_json(
        self,
    ) -> JsonValue:
        return _to_json(self)

    @classmethod
    def from_json(self) -> Self:
        raise NotImplementedError("TODO")

    @cached_property
    def artifact_hash(  # TODO: should this be __hash__?
        self,
    ) -> str:
        return _hash_dict_deterministically(self.to_json())

    @cached_property
    def schema(
        self,
    ) -> JsonValue:
        return _schema_type(type(self), set())

    @cached_property
    def schema_hash(self) -> str:
        return _hash_dict_deterministically(self.schema)

    @cached_property
    def data_dir(self) -> Path:
        return (
            config.directories.data
            / Path(*fully_qualified_name(type(self)).split("."))
            / self.schema_hash
            / self.artifact_hash
        )

    @cached_property
    def _internal_furu_dir(self) -> Path:
        return self.data_dir / ".furu"
