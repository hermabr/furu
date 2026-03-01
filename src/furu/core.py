import pickle
import secrets
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    assert_never,
)

from furu.config import config
from furu.locking import run_with_lease_and_pickle_result
from furu.metadata import CompletedMetadata, RunningMetadata
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import JsonValue, Ok, _hash_dict_deterministically, fully_qualified_name

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

        # TODO: have the logs be per-run with a run id

        # TODO: initialize the state
        # TODO: add file locking and keepalive in a different process
        self._internal_furu_dir.mkdir(
            exist_ok=True, parents=True
        )  # TODO: decide if i should make the directory here or inside the cached property itself

        def create_wrapper() -> T:  # TODO: better name
            metadata = RunningMetadata(
                artifact=self.to_json(),
                artifact_hash=self.artifact_hash,
                schema_=self.schema,
                schema_hash=self.schema_hash,
                data_path=self.data_dir.resolve(),
                started_at=datetime.now(),  # TODO: make this timezone aware?
            )
            self._metadata_path.write_text(metadata.model_dump_json(indent=2))

            try:
                result = self._create()
            except BaseException as exc:
                with (  # TODO: log this to the regular log file
                    (
                        self._internal_furu_dir
                        / f"error-{datetime.now():%y%m%d_%H-%M-%S}-{secrets.token_hex(4)}.log"  # TODO: make this part of the regular error
                    ).open("a", encoding="utf-8") as f
                ):
                    f.write("Traceback (most recent call last):\n")
                    f.writelines(
                        traceback.format_list(
                            traceback.extract_stack()[:-1]
                            + traceback.extract_tb(exc.__traceback__)
                        )
                    )
                    f.writelines(traceback.format_exception_only(type(exc), exc))
                    f.write("\n=== Debug Details (with locals) ===\n")
                    f.writelines(
                        traceback.TracebackException.from_exception(
                            exc, capture_locals=True
                        ).format(chain=True)
                    )
                raise

            completed_metadata = CompletedMetadata(
                artifact=metadata.artifact,
                artifact_hash=metadata.artifact_hash,
                schema_=metadata.schema_,
                schema_hash=metadata.schema_hash,
                data_path=metadata.data_path,
                started_at=metadata.started_at,
                completed_at=datetime.now(),
            )
            self._metadata_path.write_text(completed_metadata.model_dump_json(indent=2))

            return result

        maybe_result = run_with_lease_and_pickle_result(
            create_wrapper,
            lock_path=self._internal_furu_dir / "compute.lock",
            result_path=self._result_path,
        )

        match maybe_result:
            case Ok(result=result):
                return result
            case "lost-lock":
                raise NotImplementedError("TODO: Handle this")
            case "missing-tmp":
                raise NotImplementedError("TODO: Handle this")
            case x:
                assert_never(x)

    def status(
        self,
    ) -> Literal[
        "completed", "missing", "running", "failed"
    ]:  # TODO: add queued/waiting state?
        if self.is_completed():
            return "completed"
        raise NotImplementedError("TODO")

    def is_completed(
        self,
    ) -> bool:  # TODO: maybe this should check the is self.status is completed? (in that case status cant check if self.is_completed)
        return self._result_path.exists()

    def try_load(self) -> T:  # TODO: make a better name for this
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
    def to_json(  # TODO: make sure this doesn't prevent garbage collection
        self,
    ) -> JsonValue:
        return _to_json(self)

    @classmethod
    def from_json(cls) -> Self:
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

    @cached_property  # TODO: decide if something like this should be cached_property or simply property
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

    @cached_property
    def _metadata_path(self) -> Path:
        return self._internal_furu_dir / "metadata.json"
