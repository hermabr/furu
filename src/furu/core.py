import os
import pickle
import secrets
import shutil
import traceback
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache, cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
)

from furu.config import config
from furu.logging import _scoped_log_file, get_logger
from furu.locking import LockLostError, lock

# from furu.locking import run_with_lease_and_pickle_result
from furu.metadata import RunningMetadata
from furu.schema import schema_type as _schema_type
from furu.serialize import to_json as _to_json
from furu.utils import (
    JsonValue,
    _hash_dict_deterministically,
    _nfs_safe_unique_name,
    fully_qualified_name,
)

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

    def load_or_create(self, use_lock: bool = True) -> T:
        logger = get_logger()

        if self._result_path.exists():
            logger.info(
                "loading cached result for %s from %s",
                self._log_label,
                self._result_path,
            )
            # TODO: validation that its up to date and valid
            with open(self._result_path, "rb") as f:
                return pickle.load(f)

        self._internal_furu_dir.mkdir(exist_ok=True, parents=True)

        try:
            logger.info("calling %s.load_or_create()", self._log_label)
            with _scoped_log_file(self._log_path):
                logger.debug("load_or_create start")

                with (
                    lock(self._internal_furu_dir / "compute.lock")
                    if use_lock
                    else nullcontext()
                ) as has_lock:
                    if self._result_path.exists():
                        logger.info(
                            "loading cached result after waiting from %s",
                            self._result_path,
                        )
                        with open(self._result_path, "rb") as f:
                            return pickle.load(f)

                    metadata = RunningMetadata(
                        artifact=self.to_json(),
                        artifact_hash=self.artifact_hash,
                        schema_=self.schema,
                        schema_hash=self.schema_hash,
                        data_path=self.data_dir.resolve(),
                        started_at=datetime.now(timezone.utc),
                    )
                    self._metadata_path.write_text(metadata.model_dump_json(indent=2))
                    logger.debug("running _create()")
                    result = self._create()
                    logger.debug("_create() returned")

                    completed_metadata = metadata.to_complete()

                    if has_lock and not has_lock():
                        raise LockLostError(
                            f"lost lock at {self._internal_furu_dir / 'compute.lock'} "
                            "before writing final result"
                        )

                    tmp_result_path = self._result_path.with_suffix(".tmp.pkl")
                    with tmp_result_path.open("wb") as f:
                        pickle.dump(
                            result,
                            f,
                        )
                        f.flush()  # TODO: Do i need this and the os.fsync?
                        os.fsync(f.fileno())

                    if has_lock and not has_lock():
                        raise LockLostError(
                            f"lost lock at {self._internal_furu_dir / 'compute.lock'} "
                            "after writing temporary result"
                        )

                    tmp_result_path.rename(self._result_path)
                    self._metadata_path.write_text(
                        completed_metadata.model_dump_json(indent=2)
                    )
                    logger.debug("stored result at %s", self._result_path)

                    logger.debug("load_or_create complete")
            logger.info("%s.load_or_create() returned", self._log_label)

        except BaseException as exc:
            with _scoped_log_file(self._log_path):
                logger.exception("load_or_create failed")
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

        return result

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

    def delete(self, mode: Literal["prompt", "force"] = "prompt") -> bool:
        if not self.data_dir.exists():
            return False

        self._internal_furu_dir.mkdir(exist_ok=True, parents=True)

        tombstone_path: Path | None = None
        try:
            with lock(self._internal_furu_dir / "compute.lock"):
                if not self.data_dir.exists():
                    return False

                if mode == "prompt" and (
                    input(f"Do you want to delete {self.data_dir}? [y/N] ")
                    .strip()
                    .lower()
                    != "y"
                ):
                    return False

                tombstone_path = _nfs_safe_unique_name(self.data_dir, name="deleting")
                self.data_dir.rename(tombstone_path)
        except LockLostError:
            if tombstone_path is None:
                raise

        assert tombstone_path is not None
        shutil.rmtree(tombstone_path)
        return True

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

    @cached_property
    def _log_path(self) -> Path:
        return self._internal_furu_dir / "run.log"

    @cached_property
    def _log_label(self) -> str:
        return f"{fully_qualified_name(type(self))}:{self.artifact_hash}"
