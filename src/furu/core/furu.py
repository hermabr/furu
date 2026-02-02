import datetime
import getpass
import inspect
import os
import signal
import socket
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from types import FrameType
from typing import (
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)

import chz
import submitit
from chz.field import Field as ChzField
from typing_extensions import dataclass_transform

from ..adapters import SubmititAdapter
from ..adapters.submitit import SubmititJob
from ..config import FURU_CONFIG
from ..errors import (
    MISSING,
    FuruComputeError,
    FuruLockNotAcquired,
    FuruValidationError,
    FuruWaitTimeout,
)
from ..runtime import current_holder
from ..runtime.logging import enter_holder, get_logger, log, write_separator
from ..runtime.tracebacks import format_traceback
from ..runtime.overrides import has_override, lookup_override
from ..schema import schema_key_from_cls
from ..serialization import FuruSerializer
from ..serialization.serializer import JsonValue
from ..aliases import collect_aliases
from ..migration import (
    FuruRef,
    MigrationReport,
    current as _current_refs,
    migrate as _migrate,
    migrate_one as _migrate_one,
    resolve_original_ref,
    stale as _stale_refs,
)
from ..storage import (
    FuruMetadata,
    MetadataManager,
    MigrationManager,
    MigrationRecord,
    StateManager,
    StateOwner,
)
from ..storage.state import (
    _FuruState,
    _OwnerDict,
    _StateAttemptQueued,
    _StateAttemptRunning,
    _StateResultAbsent,
    _StateResultFailed,
    _StateResultSuccess,
    compute_lock,
)


class _SubmititEnvInfo(TypedDict, total=False):
    """Environment info collected for submitit jobs."""

    backend: str
    slurm_job_id: str | None
    pid: int
    host: str
    user: str
    started_at: str
    command: str


class _CallerInfo(TypedDict, total=False):
    """Caller location info for logging."""

    furu_caller_file: str
    furu_caller_line: int


@dataclass_transform(
    field_specifiers=(chz.field,), kw_only_default=True, frozen_default=True
)
class Furu[T](ABC):
    """
    Base class for cached computations with provenance tracking.

    Subclasses must implement:
    - _create(self) -> T
    - _load(self) -> T
    """

    MISSING = MISSING

    # Configuration (can be overridden in subclasses)
    version_controlled: ClassVar[bool] = False

    # Maximum time to wait for result (seconds). Default: 10 minutes.
    _max_wait_time_sec: float = 600.0

    def __init_subclass__(
        cls,
        *,
        version_controlled: bool | None = None,
        version: str | None = None,
        typecheck: bool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "Furu" and cls.__module__ == __name__:
            return

        # Python 3.14+ may not populate `__annotations__` in `cls.__dict__` (PEP 649).
        # `chz` expects annotations to exist for every `chz.field()` attribute, so we
        # materialize them and (as a last resort) fill missing ones with `Any`.
        try:
            annotations = dict(getattr(cls, "__annotations__", {}) or {})
        except Exception:
            annotations = {}

        try:
            materialized = inspect.get_annotations(cls, eval_str=False)
        except TypeError:  # pragma: no cover
            materialized = inspect.get_annotations(cls)
        except Exception:
            materialized = {}

        if materialized:
            annotations.update(materialized)

        FieldType: type[object] | None
        try:
            from chz.field import Field as _ChzField
        except Exception:  # pragma: no cover
            FieldType = None
        else:
            FieldType = _ChzField

        if FieldType is not None:
            for field_name, value in cls.__dict__.items():
                if isinstance(value, FieldType) and field_name not in annotations:
                    annotations[field_name] = Any

        if annotations:
            type.__setattr__(cls, "__annotations__", annotations)

        chz_kwargs: dict[str, str | bool] = {}
        if version is not None:
            chz_kwargs["version"] = version
        if typecheck is not None:
            chz_kwargs["typecheck"] = typecheck
        chz.chz(cls, **chz_kwargs)

        if version_controlled is not None:
            setattr(cls, "version_controlled", version_controlled)

    @classmethod
    def _namespace(cls) -> Path:
        module = getattr(cls, "__module__", None)
        qualname = getattr(cls, "__qualname__", cls.__name__)
        if not module or module == "__main__":
            raise ValueError(
                "Cannot derive Furu namespace from __main__; define the class in an importable module."
            )
        if "<locals>" in qualname:
            raise ValueError(
                "Cannot derive Furu namespace for a local class; define it at module scope."
            )
        return Path(*module.split("."), *qualname.split("."))

    @abstractmethod
    def _create(self: Self) -> T:
        """Compute and save the result (implement in subclass)."""
        raise NotImplementedError(
            f"{self.__class__.__name__}._create() not implemented"
        )

    @abstractmethod
    def _load(self: Self) -> T:
        """Load the result from disk (implement in subclass)."""
        raise NotImplementedError(f"{self.__class__.__name__}._load() not implemented")

    def _validate(self: Self) -> bool:
        """
        Validate that result is complete and correct (override if needed).

        Return False or raise FuruValidationError to mark artifacts as invalid.
        """
        return True

    def _dependencies(self: Self) -> "DependencySpec | None":
        """Return extra dependencies not captured by fields."""
        return None

    def _executor_spec_key(self: Self) -> str:
        return "default"

    def _get_dependencies(self: Self, *, recursive: bool = True) -> list["Furu"]:
        """Collect Furu dependencies from fields and `_dependencies()`."""
        seen = {self.furu_hash}
        dependencies: list[Furu] = []
        _collect_dependencies(self, dependencies, seen, recursive=recursive)
        return dependencies

    def _dependency_hashes(self: Self) -> list[str]:
        dependencies = _direct_dependencies(self)
        if not dependencies:
            return []

        digests: set[str] = set()
        for dependency in dependencies:
            if dependency is self:
                raise ValueError("Furu dependencies cannot include self")
            digests.add(dependency.furu_hash)
        return sorted(digests)

    def _invalidate_cached_success(self: Self, directory: Path, *, reason: str) -> None:
        logger = get_logger()
        logger.warning(
            "invalidate %s %s %s (%s)",
            self.__class__.__name__,
            self.furu_hash,
            directory,
            reason,
        )

        StateManager.get_success_marker_path(directory).unlink(missing_ok=True)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

        def mutate(state: _FuruState) -> None:
            state.result = _StateResultAbsent(status="absent")

        StateManager.update_state(directory, mutate)
        StateManager.append_event(
            directory, {"type": "result_invalidated", "reason": reason, "at": now}
        )

    def _prepare_executor_rerun(self: Self, directory: Path) -> None:
        if not self._always_rerun():
            return
        if not directory.exists():
            return
        migration = self._alias_record(directory)
        if migration is not None and self._alias_is_active(directory, migration):
            self._maybe_detach_alias(
                directory=directory,
                record=migration,
                reason="always_rerun",
            )
        state = StateManager.read_state(directory)
        if isinstance(state.result, _StateResultSuccess):
            self._invalidate_cached_success(directory, reason="always_rerun enabled")

    @cached_property
    def furu_hash(self: Self) -> str:
        """Return the stable content hash for this Furu object."""
        return FuruSerializer.compute_hash(self)

    def _always_rerun(self: Self) -> bool:
        if FURU_CONFIG.always_rerun_all:
            return True
        if not FURU_CONFIG.always_rerun:
            return False
        qualname = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        return qualname in FURU_CONFIG.always_rerun

    def _base_furu_dir(self: Self) -> Path:
        root = FURU_CONFIG.get_root(self.version_controlled)
        return root / self.__class__._namespace() / self.furu_hash

    @cached_property
    def furu_dir(self: Self) -> Path:
        """Get the directory for this Furu object."""
        directory = self._base_furu_dir()
        migration = self._alias_record(directory)
        if migration is not None:
            target_dir = self._alias_target_dir(directory, migration)
            if target_dir is not None:
                return target_dir
        return directory

    @property
    def raw_dir(self: Self) -> Path:
        """
        Get the raw directory for Furu.

        This is intended for large, non-versioned byproducts or inputs.
        """
        return FURU_CONFIG.raw_dir

    def to_dict(self: Self) -> JsonValue:
        """Convert to dictionary."""
        return FuruSerializer.to_dict(self)

    @classmethod
    def from_dict(cls, data: JsonValue) -> "Furu":
        """Reconstruct from dictionary."""
        return FuruSerializer.from_dict(data)

    def to_python(self: Self, multiline: bool = True) -> str:
        """
        Serialize this Furu object to Python source code.
        
        Parameters:
        	multiline (bool): If True, produce a multi-line, human-readable representation; if False, produce a compact single-line representation.
        
        Returns:
        	python_code (str): Python source code that reconstructs the object.
        """
        return FuruSerializer.to_python(self, multiline=multiline)

    @classmethod
    def schema_key(cls) -> tuple[str, ...]:
        """
        Compute the stable schema key for the given class's serialization schema.
        
        Parameters:
            cls (type): The class whose schema key will be derived.
        
        Returns:
            schema_key (tuple[str, ...]): Tuple of strings representing the stable schema key for the class.
        """
        return schema_key_from_cls(cls)

    @classmethod
    def migrate(
        cls,
        *,
        from_schema: Iterable[str] | None = None,
        from_drop: Iterable[str] | None = None,
        from_add: Iterable[str] | None = None,
        from_hash: str | None = None,
        from_furu_obj: dict[str, JsonValue] | None = None,
        from_namespace: str | None = None,
        default_field: Iterable[str] | None = None,
        set_field: Mapping[str, JsonValue] | None = None,
        drop_field: Iterable[str] | None = None,
        rename_field: Mapping[str, str] | None = None,
        include_alias_sources: bool = False,
        conflict: Literal["throw", "skip"] = "throw",
        origin: str | None = None,
        note: str | None = None,
    ) -> MigrationReport:
        """
        Perform a schema-aware migration of existing stored instances of this Furu subclass.
        
        Parameters:
            from_schema (Iterable[str] | None): Names of schema fields to treat as the source schema for migration selection.
            from_drop (Iterable[str] | None): Field names to drop from source schema when matching migration candidates.
            from_add (Iterable[str] | None): Field names to add to source schema when matching migration candidates.
            from_hash (str | None): Specific source furu hash to migrate from.
            from_furu_obj (dict[str, JsonValue] | None): A single source object provided as a dict to migrate into this class's schema.
            from_namespace (str | None): Namespace filter for candidate sources (limits which stored objects are considered).
            default_field (Iterable[str] | None): Field names to populate with default values when performing the migration.
            set_field (Mapping[str, JsonValue] | None): Mapping of field names to values to set on migrated objects.
            drop_field (Iterable[str] | None): Field names to remove from migrated objects.
            rename_field (Mapping[str, str] | None): Mapping of old field names to new field names for renaming during migration.
            include_alias_sources (bool): If true, include alias-origin objects as migration sources.
            conflict (Literal["throw","skip"]): How to handle conflicting migrations: `"throw"` to raise on conflict, `"skip"` to ignore.
            origin (str | None): Optional string describing the origin of this migration (stored with migration records).
            note (str | None): Optional human-readable note to attach to migration records.
        
        Returns:
            MigrationReport: A report summarizing which objects were migrated, skipped, or failed and any side effects performed.
        """
        return _migrate(
            cls,
            from_schema=from_schema,
            from_drop=from_drop,
            from_add=from_add,
            from_hash=from_hash,
            from_furu_obj=from_furu_obj,
            from_namespace=from_namespace,
            default_field=default_field,
            set_field=set_field,
            drop_field=drop_field,
            rename_field=rename_field,
            include_alias_sources=include_alias_sources,
            conflict=conflict,
            origin=origin,
            note=note,
        )

    @classmethod
    def migrate_one(
        cls,
        *,
        from_hash: str,
        from_namespace: str | None = None,
        default_field: Iterable[str] | None = None,
        set_field: Mapping[str, JsonValue] | None = None,
        drop_field: Iterable[str] | None = None,
        rename_field: Mapping[str, str] | None = None,
        include_alias_sources: bool = False,
        conflict: Literal["throw", "skip"] = "throw",
        origin: str | None = None,
        note: str | None = None,
    ) -> MigrationRecord | None:
        """
        Attempt a single migration from a specific previous Furu version to this class's schema.
        
        Parameters:
            from_hash (str): Hash of the source Furu to migrate from.
            from_namespace (str | None): Optional namespace to restrict the source; defaults to None.
            default_field (Iterable[str] | None): Field names to fill with default values when missing in the source.
            set_field (Mapping[str, JsonValue] | None): Field values to set on the migrated object.
            drop_field (Iterable[str] | None): Field names to drop from the migrated result.
            rename_field (Mapping[str, str] | None): Mapping of old field names to new field names.
            include_alias_sources (bool): If True, consider alias records as possible migration sources.
            conflict (Literal["throw", "skip"]): Conflict resolution strategy; "throw" to raise on conflict, "skip" to ignore conflicting sources.
            origin (str | None): Optional origin identifier to record in the migration metadata.
            note (str | None): Optional human-readable note to record with the migration.
        
        Returns:
            MigrationRecord | None: A MigrationRecord describing the performed migration if one was applied, or None if no applicable source was found or migration was skipped.
        """
        return _migrate_one(
            cls,
            from_hash=from_hash,
            from_namespace=from_namespace,
            default_field=default_field,
            set_field=set_field,
            drop_field=drop_field,
            rename_field=rename_field,
            include_alias_sources=include_alias_sources,
            conflict=conflict,
            origin=origin,
            note=note,
        )

    @classmethod
    def stale(cls, *, namespace: str | None = None) -> list[FuruRef]:
        """
        List stale references for this Furu class.
        
        Parameters:
            namespace (str | None): If provided, restrict results to the given namespace.
        
        Returns:
            stale_refs (list[FuruRef]): A list of FuruRef objects representing stale references for this class.
        """
        return _stale_refs(cls, namespace=namespace)

    @classmethod
    def current(cls, *, namespace: str | None = None) -> list[FuruRef]:
        """
        List current Furu references for the class, optionally filtered by namespace.
        
        Parameters:
            namespace (str | None): Namespace to restrict the search. If `None`, the class's default namespace is used.
        
        Returns:
            list[FuruRef]: Current (non-stale) references for this Furu class.
        """
        return _current_refs(cls, namespace=namespace)

    def log(self: Self, message: str, *, level: str = "INFO") -> Path:
        """
        Log a message to this Furu object's log file.
        
        Parameters:
            level (str): Logging level name (e.g., "INFO", "ERROR").
        
        Returns:
            Path: Path to the log file where the message was written.
        """
        return log(message, level=level)

    def _exists_quiet(self: Self) -> bool:
        if has_override(self.furu_hash):
            return True
        directory = self._base_furu_dir()
        success_dir = self._success_marker_dir(directory)
        if success_dir is None:
            return False
        try:
            return self._validate()
        except FuruValidationError as exc:
            logger = get_logger()
            logger.warning(
                "exists %s -> false (validate invalid for %s: %s)",
                directory,
                f"{self.__class__.__name__}({self.furu_hash})",
                exc,
            )
            return False
        except Exception as exc:
            logger = get_logger()
            logger.exception(
                "exists %s -> false (validate crashed for %s: %s)",
                directory,
                f"{self.__class__.__name__}({self.furu_hash})",
                exc,
            )
            return False

    def exists(self: Self) -> bool:
        """Check if result exists and is valid."""
        logger = get_logger()
        directory = self._base_furu_dir()
        if has_override(self.furu_hash):
            logger.info("exists %s -> true (override)", directory)
            return True
        success_dir = self._success_marker_dir(directory)
        if success_dir is None:
            logger.info("exists %s -> false", directory)
            return False

        ok = self._validate()
        logger.info("exists %s -> %s", directory, "true" if ok else "false")
        return ok

    def get_metadata(self: Self) -> "FuruMetadata":
        """Get metadata for this object."""
        directory = self._base_furu_dir()
        return MetadataManager.read_metadata(directory)

    def get_migration_record(self: Self) -> MigrationRecord | None:
        """
        Return the migration record associated with this Furu instance, if any.
        
        Returns:
            MigrationRecord | None: The migration record for this object's base directory, or `None` if no record exists.
        """
        return MigrationManager.read_migration(self._base_furu_dir())

    def is_alias(self: Self) -> bool:
        """
        Check whether this Furu's base directory has an alias migration record.
        
        Returns:
            True if an alias migration record exists for the object's base directory, False otherwise.
        """
        record = self._alias_record(self._base_furu_dir())
        return record is not None

    def original(self: Self) -> FuruRef:
        """
        Get a FuruRef pointing to the original (non-aliased) storage directory for this Furu.
        
        Returns:
            FuruRef: Reference to the original Furu, containing namespace, furu_hash, root kind, and the canonical directory for the object.
        """
        base_dir = self._base_furu_dir()
        ref = FuruRef(
            namespace=".".join(self.__class__._namespace().parts),
            furu_hash=self.furu_hash,
            root=MigrationManager.root_kind_for_dir(base_dir),
            directory=base_dir,
        )
        return resolve_original_ref(ref)

    def aliases(self: Self, *, include_inactive: bool = True) -> list[FuruRef]:
        """
        Collect alias references that point to this object's original Furu.
        
        Parameters:
        	include_inactive (bool): If True, include aliases that are inactive/overwritten; if False, include only currently active aliases.
        
        Returns:
        	list[FuruRef]: Alias targets for this Furu's original, sorted by `furu_hash`.
        """
        original_ref = self.original()
        alias_index = collect_aliases(include_inactive=include_inactive)
        records = alias_index.get(
            (original_ref.namespace, original_ref.furu_hash, original_ref.root),
            [],
        )
        refs = [
            FuruRef(
                namespace=record.to_namespace,
                furu_hash=record.to_hash,
                root=record.to_root,
                directory=MigrationManager.resolve_dir(record, target="to"),
            )
            for record in records
        ]
        refs.sort(key=lambda item: item.furu_hash)
        return refs

    def get(self: Self, *, force: bool = False) -> T:
        """
        Load result if it exists, computing if necessary.

        Args:
            force: Allow computation inside executor contexts if the spec matches.

        Returns:
            Loaded or computed result.

        Raises:
            FuruComputeError: If computation fails with detailed error information
        """
        has_override_value, override_value = lookup_override(self.furu_hash)
        if has_override_value:
            return cast(T, override_value)
        from furu.errors import (
            FuruExecutionError,
            FuruMissingArtifact,
            FuruSpecMismatch,
        )
        from furu.execution.context import EXEC_CONTEXT

        ctx = EXEC_CONTEXT.get()
        if ctx.mode == "executor":
            logger = get_logger()
            parent_holder = current_holder()
            has_parent = parent_holder is not None and parent_holder is not self
            needs_holder = parent_holder is None or has_parent
            caller_info: _CallerInfo = {}
            if has_parent:
                caller_info = self._get_caller_info()

            def _executor_get() -> T:
                directory = self._base_furu_dir()
                if force:
                    if (
                        ctx.current_node_hash is None
                        or self.furu_hash != ctx.current_node_hash
                    ):
                        raise FuruExecutionError(
                            "force=True not allowed: only the current node may compute in executor mode. "
                            f"current_node_hash={ctx.current_node_hash!r} "
                            f"obj={self.__class__.__name__}({self.furu_hash})",
                            hints=[
                                "Declare this object as a dependency instead of calling dep.get(force=True).",
                                "Inside executor mode, use get(force=True) only on the node being executed.",
                            ],
                        )
                    self._prepare_executor_rerun(directory)

                exists_ok = self._exists_quiet()
                if exists_ok and not (force and self._always_rerun()):
                    return self._load()

                if force and not exists_ok:
                    state = self.get_state(directory)
                    if isinstance(state.result, _StateResultSuccess):
                        self._invalidate_cached_success(
                            directory, reason="_validate returned false (executor)"
                        )

                if not force:
                    raise FuruMissingArtifact(
                        "Missing artifact "
                        f"{self.__class__.__name__}({self.furu_hash}) in executor mode. "
                        f"Requested by {ctx.current_node_hash}. Declare it as a dependency."
                    )

                required = self._executor_spec_key()
                if ctx.spec_key is None or required != ctx.spec_key:
                    raise FuruSpecMismatch(
                        "force=True not allowed: "
                        f"required={required!r} != worker={ctx.spec_key!r} (v1 exact match)"
                    )

                StateManager.ensure_internal_dir(directory)
                status, created_here, result = self._run_locally(
                    start_time=time.time(),
                    allow_failed=FURU_CONFIG.retry_failed,
                    executor_mode=True,
                )
                if status == "success":
                    if created_here:
                        return cast(T, result)
                    return self._load()

                raise self._build_failed_state_error(
                    self._base_furu_dir(),
                    None,
                    message="Computation previously failed",
                )

            if has_parent:
                logger.debug(
                    "dep: begin %s %s %s",
                    self.__class__.__name__,
                    self.furu_hash,
                    self._base_furu_dir(),
                    extra=caller_info,
                )

            ok = False
            try:
                if needs_holder:
                    with enter_holder(self):
                        result = _executor_get()
                else:
                    result = _executor_get()
                ok = True
                return result
            finally:
                if has_parent:
                    logger.debug(
                        "dep: end %s %s (%s)",
                        self.__class__.__name__,
                        self.furu_hash,
                        "ok" if ok else "error",
                        extra=caller_info,
                    )

        return self._get_impl_interactive(force=force)

    def _get_impl_interactive(self: Self, *, force: bool) -> T:
        logger = get_logger()
        parent_holder = current_holder()
        has_parent = parent_holder is not None and parent_holder is not self
        caller_info = self._get_caller_info()
        retry_failed_effective = FURU_CONFIG.retry_failed
        if has_parent:
            logger.debug(
                "dep: begin %s %s %s",
                self.__class__.__name__,
                self.furu_hash,
                self._base_furu_dir(),
                extra=caller_info,
            )

        ok = False
        try:
            with enter_holder(self):
                start_time = time.time()
                base_dir = self._base_furu_dir()
                directory = base_dir
                migration = self._alias_record(base_dir)
                alias_active = False
                base_marker = StateManager.success_marker_exists(base_dir)

                if (
                    migration is not None
                    and migration.kind == "alias"
                    and migration.overwritten_at is None
                    and not base_marker
                ):
                    target_dir = self._alias_target_dir(
                        base_dir, migration, base_marker=base_marker
                    )
                    if target_dir is not None:
                        alias_active = True
                        directory = target_dir
                    else:
                        self._maybe_detach_alias(
                            directory=base_dir,
                            record=migration,
                            reason="original_not_success",
                        )
                        migration = MigrationManager.read_migration(base_dir)

                if alias_active and self._always_rerun():
                    if migration is not None:
                        self._maybe_detach_alias(
                            directory=base_dir,
                            record=migration,
                            reason="always_rerun",
                        )
                    migration = MigrationManager.read_migration(base_dir)
                    alias_active = False
                    directory = base_dir

                # Optimistic read: if state is already good, we don't need to reconcile (write lock)
                # Optimization: Check for success marker first to avoid reading state.json
                # This is much faster for cache hits (11x speedup on check).
                success_marker = StateManager.get_success_marker_path(directory)
                if success_marker.is_file():
                    # We have a success marker. Check if we can use it.
                    if self._always_rerun():
                        self._invalidate_cached_success(
                            directory, reason="always_rerun enabled"
                        )
                        # Fall through to normal load
                    else:
                        try:
                            if not self._validate():
                                self._invalidate_cached_success(
                                    directory, reason="_validate returned false"
                                )
                                # Fall through
                            else:
                                # Valid success! Return immediately.
                                # Since we didn't read state, we skip the logging below for speed
                                # or we can log a minimal message if needed.
                                ok = True
                                self._log_console_start(action_color="green")
                                return self._load()
                        except Exception as e:
                            self._invalidate_cached_success(
                                directory,
                                reason=f"_validate raised {type(e).__name__}: {e}",
                            )
                            # Fall through

                state0 = StateManager.read_state(directory)

                if (
                    isinstance(state0.result, _StateResultFailed)
                    and not retry_failed_effective
                ):
                    raise self._build_failed_state_error(
                        directory,
                        state0,
                        message="Computation previously failed",
                    )

                if isinstance(state0.result, _StateResultSuccess):
                    # Double check logic if we fell through to here (e.g. race condition or invalidation above)
                    if self._always_rerun():
                        self._invalidate_cached_success(
                            directory, reason="always_rerun enabled"
                        )
                        state0 = StateManager.read_state(directory)
                    else:
                        try:
                            if not self._validate():
                                self._invalidate_cached_success(
                                    directory, reason="_validate returned false"
                                )
                                state0 = StateManager.read_state(directory)
                        except Exception as e:
                            self._invalidate_cached_success(
                                directory,
                                reason=f"_validate raised {type(e).__name__}: {e}",
                            )
                            state0 = StateManager.read_state(directory)

                attempt0 = state0.attempt
                if isinstance(state0.result, _StateResultSuccess):
                    decision = "success->load"
                    action_color = "green"
                elif isinstance(attempt0, (_StateAttemptQueued, _StateAttemptRunning)):
                    decision = f"{attempt0.status}->wait"
                    action_color = "yellow"
                else:
                    decision = "create"
                    action_color = "blue"

                # Cache hits can be extremely noisy in pipelines; keep logs for state
                # transitions (create/wait) and error cases, but suppress repeated
                # "success->load" lines and the raw separator on successful loads.
                self._log_console_start(
                    action_color=action_color,
                    caller_info=caller_info,
                )

                if decision != "success->load":
                    if decision == "create":
                        StateManager.ensure_internal_dir(directory)
                    write_separator()
                    logger.debug(
                        "get %s %s %s (%s)",
                        self.__class__.__name__,
                        self.furu_hash,
                        directory,
                        decision,
                        extra={
                            "furu_action_color": action_color,
                            **caller_info,
                        },
                    )

                # Fast path: already successful
                state_now = StateManager.read_state(directory)
                if isinstance(state_now.result, _StateResultSuccess):
                    try:
                        result = self._load()
                        ok = True
                        return result
                    except Exception as e:
                        # Ensure there is still a clear marker in logs for unexpected
                        # failures even when we suppressed the cache-hit header line.
                        write_separator()
                        logger.error(
                            "get %s %s (load failed)",
                            self.__class__.__name__,
                            self.furu_hash,
                        )
                        raise FuruComputeError(
                            f"Failed to load result from {directory}",
                            StateManager.get_state_path(directory),
                            e,
                        ) from e

                status, created_here, result = self._run_locally(
                    start_time=start_time,
                    allow_failed=retry_failed_effective,
                    executor_mode=False,
                )
                if status == "success":
                    ok = True
                    if created_here:
                        logger.debug(
                            "get: %s created -> return",
                            self.__class__.__name__,
                        )
                        return cast(T, result)
                    logger.debug(
                        "get: %s success -> _load()",
                        self.__class__.__name__,
                    )
                    return self._load()

                raise self._build_failed_state_error(
                    directory,
                    None,
                    message="Computation previously failed",
                )
        finally:
            if has_parent:
                logger.debug(
                    "dep: end %s %s (%s)",
                    self.__class__.__name__,
                    self.furu_hash,
                    "ok" if ok else "error",
                    extra=caller_info,
                )

    @staticmethod
    def _get_caller_info() -> _CallerInfo:
        frame = sys._getframe(1)
        caller_info: _CallerInfo = {}
        if frame is not None:
            # Walk up the stack to find the caller outside of furu package
            furu_pkg_dir = str(Path(__file__).parent.parent)
            while frame is not None:
                filename = frame.f_code.co_filename
                # Skip frames from within the furu package
                if not filename.startswith(furu_pkg_dir):
                    caller_info = {
                        "furu_caller_file": filename,
                        "furu_caller_line": frame.f_lineno,
                    }
                    break
                frame = frame.f_back
        return caller_info

    def _log_console_start(
        self, action_color: str, caller_info: _CallerInfo | None = None
    ) -> None:
        """Log the start of get to console with caller info."""
        logger = get_logger()
        if caller_info is None:
            caller_info = self._get_caller_info()

        logger.info(
            "get %s %s",
            self.__class__.__name__,
            self.furu_hash,
            extra={
                "furu_console_only": True,
                "furu_action_color": action_color,
                **caller_info,
            },
        )

    def _add_exception_breadcrumbs(self, exc: BaseException, directory: Path) -> None:
        if not hasattr(exc, "add_note"):
            return
        note = f"Furu dir: {directory}"
        exc.add_note(note)

    @staticmethod
    def _failed_state_hints() -> list[str]:
        return [
            "To retry this failed artifact: set FURU_RETRY_FAILED=1 or call get() again.",
            "To inspect details: open the furu dir shown above.",
        ]

    def _build_failed_state_error(
        self,
        directory: Path,
        state: _FuruState | None,
        *,
        message: str,
    ) -> FuruComputeError:
        current_state = state or StateManager.read_state(directory)
        attempt = current_state.attempt
        error = getattr(attempt, "error", None) if attempt is not None else None
        return FuruComputeError(
            message,
            StateManager.get_state_path(directory),
            recorded_error_type=getattr(error, "type", None),
            recorded_error_message=getattr(error, "message", None),
            recorded_traceback=getattr(error, "traceback", None),
            hints=self._failed_state_hints(),
        )

    def _effective_max_wait_time_sec(self) -> float | None:
        if FURU_CONFIG.max_wait_time_sec is not None:
            return FURU_CONFIG.max_wait_time_sec
        return self._max_wait_time_sec

    def _check_timeout(self, start_time: float) -> None:
        """Check if operation has timed out."""
        max_wait_time = self._effective_max_wait_time_sec()
        if max_wait_time is not None:
            if time.time() - start_time > max_wait_time:
                raise FuruWaitTimeout(
                    f"Furu operation timed out after {max_wait_time} seconds."
                )

    def _is_migrated_state(self, directory: Path) -> bool:
        record = self._alias_record(directory)
        return record is not None and self._alias_is_active(directory, record)

    def _migration_target_dir(self, directory: Path) -> Path | None:
        record = self._alias_record(directory)
        if record is None:
            return None
        return MigrationManager.resolve_dir(record, target="from")

    def _resolve_effective_dir(self) -> Path:
        return self._base_furu_dir()

    def get_state(self, directory: Path | None = None) -> _FuruState:
        """Return the alias-aware state for this Furu directory."""
        base_dir = directory or self._base_furu_dir()
        record = self._alias_record(base_dir)
        if record is None:
            return StateManager.read_state(base_dir)
        target_dir = self._alias_target_dir(base_dir, record)
        if target_dir is None:
            return StateManager.read_state(base_dir)
        return StateManager.read_state(target_dir)

    def _alias_record(self, directory: Path) -> MigrationRecord | None:
        record = MigrationManager.read_migration(directory)
        if record is None or record.kind != "alias":
            return None
        return record

    def _alias_target_dir(
        self,
        directory: Path,
        record: MigrationRecord,
        *,
        base_marker: bool | None = None,
    ) -> Path | None:
        if record.overwritten_at is not None:
            return None
        if base_marker is None:
            base_marker = StateManager.success_marker_exists(directory)
        if base_marker:
            return None
        target = MigrationManager.resolve_dir(record, target="from")
        if StateManager.success_marker_exists(target):
            return target
        return None

    def _success_marker_dir(self, directory: Path) -> Path | None:
        base_marker = StateManager.success_marker_exists(directory)
        record = self._alias_record(directory)
        if record is None:
            return directory if base_marker else None
        target_dir = self._alias_target_dir(directory, record, base_marker=base_marker)
        if target_dir is not None:
            return target_dir
        return directory if base_marker else None

    def _alias_is_active(self, directory: Path, record: MigrationRecord) -> bool:
        return self._alias_target_dir(directory, record) is not None

    def _maybe_detach_alias(
        self: Self,
        *,
        directory: Path,
        record: MigrationRecord,
        reason: str,
    ) -> None:
        if record.overwritten_at is not None:
            return
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        record.overwritten_at = now
        MigrationManager.write_migration(record, directory)
        target = MigrationManager.resolve_dir(record, target="from")
        target_record = MigrationManager.read_migration(target)
        if target_record is not None:
            target_record.overwritten_at = now
            MigrationManager.write_migration(target_record, target)
        event: dict[str, str | int] = {
            "type": "migration_overwrite",
            "policy": record.policy,
            "from_namespace": record.from_namespace,
            "from_hash": record.from_hash,
            "to_namespace": record.to_namespace,
            "to_hash": record.to_hash,
            "reason": reason,
        }
        StateManager.append_event(directory, event.copy())
        StateManager.append_event(target, event.copy())

    def _submit_once(
        self,
        adapter: SubmititAdapter,
        directory: Path,
        on_job_id: Callable[[str], None] | None,
        *,
        allow_failed: bool,
    ) -> SubmititJob | None:
        """Submit job once without waiting (fire-and-forget mode)."""
        logger = get_logger()
        StateManager.ensure_internal_dir(directory)
        self._reconcile(directory, adapter=adapter)
        state = StateManager.read_state(directory)
        attempt = state.attempt
        if (
            isinstance(attempt, (_StateAttemptQueued, _StateAttemptRunning))
            and attempt.backend == "submitit"
        ):
            job = adapter.load_job(directory)
            if job is not None:
                return job

        # Try to acquire submit lock
        lock_path = StateManager.get_lock_path(directory, StateManager.SUBMIT_LOCK)
        lock_fd = StateManager.try_lock(lock_path)

        if lock_fd is None:
            # Someone else is submitting, wait briefly and return their job
            logger.debug(
                "submit: waiting for submit lock %s %s %s",
                self.__class__.__name__,
                self.furu_hash,
                directory,
            )
            time.sleep(0.5)
            return adapter.load_job(directory)

        attempt_id: str | None = None
        try:
            # Create metadata
            metadata = MetadataManager.create_metadata(self, directory)
            MetadataManager.write_metadata(metadata, directory)

            env_info = MetadataManager.collect_environment_info()
            owner_state = StateOwner(
                pid=env_info.pid,
                host=env_info.hostname,
                hostname=env_info.hostname,
                user=env_info.user,
                command=env_info.command,
                timestamp=env_info.timestamp,
                python_version=env_info.python_version,
                executable=env_info.executable,
                platform=env_info.platform,
            )
            owner_payload: _OwnerDict = {
                "pid": owner_state.pid,
                "host": owner_state.host,
                "hostname": owner_state.hostname,
                "user": owner_state.user,
                "command": owner_state.command,
                "timestamp": owner_state.timestamp,
                "python_version": owner_state.python_version,
                "executable": owner_state.executable,
                "platform": owner_state.platform,
            }
            attempt_id = StateManager.start_attempt_queued(
                directory,
                backend="submitit",
                lease_duration_sec=FURU_CONFIG.lease_duration_sec,
                owner=owner_payload,
                scheduler={},
            )

            job = adapter.submit(lambda: self._worker_entry(allow_failed=allow_failed))

            # Save job handle and watch for job ID
            adapter.pickle_job(job, directory)
            adapter.watch_job_id(
                job,
                directory,
                attempt_id=attempt_id,
                callback=on_job_id,
            )

            return job
        except Exception as e:
            if attempt_id is not None:
                StateManager.finish_attempt_failed(
                    directory,
                    attempt_id=attempt_id,  # type: ignore[arg-type]
                    error={
                        "type": type(e).__name__,
                        "message": f"Failed to submit: {e}",
                    },
                )
            else:

                def mutate(state: _FuruState) -> None:
                    state.result = _StateResultFailed(status="failed")

                StateManager.update_state(directory, mutate)
            raise FuruComputeError(
                "Failed to submit job",
                StateManager.get_state_path(directory),
                e,
            ) from e
        finally:
            StateManager.release_lock(lock_fd, lock_path)

    def _worker_entry(self: Self, *, allow_failed: bool | None = None) -> None:
        """Entry point for worker process (called by submitit or locally)."""
        with enter_holder(self):
            logger = get_logger()
            # Ensure executor semantics apply to *all* work in the worker, not
            # just `_create()`. This prevents accidental dependency computation
            # (e.g., from within `_validate()` or metadata hooks).
            from furu.execution.context import EXEC_CONTEXT, ExecContext

            exec_token = EXEC_CONTEXT.set(
                ExecContext(
                    mode="executor",
                    spec_key=self._executor_spec_key(),
                    backend="submitit",
                    current_node_hash=self.furu_hash,
                )
            )
            try:
                directory = self._base_furu_dir()
                StateManager.ensure_internal_dir(directory)
                always_rerun = self._always_rerun()
                needs_success_invalidation = False
                if not always_rerun:
                    exists_ok = self._exists_quiet()
                    if not exists_ok:
                        state = self.get_state(directory)
                        if isinstance(state.result, _StateResultSuccess):
                            needs_success_invalidation = True

                env_info = self._collect_submitit_env()
                allow_failed_effective = (
                    allow_failed
                    if allow_failed is not None
                    else FURU_CONFIG.retry_failed
                )
                allow_success = always_rerun or needs_success_invalidation

                try:
                    with compute_lock(
                        directory,
                        backend="submitit",
                        lease_duration_sec=FURU_CONFIG.lease_duration_sec,
                        heartbeat_interval_sec=FURU_CONFIG.heartbeat_interval_sec,
                        owner={
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "user": getpass.getuser(),
                            "command": " ".join(sys.argv) if sys.argv else "<unknown>",
                        },
                        scheduler={
                            "backend": env_info.get("backend"),
                            "job_id": env_info.get("slurm_job_id"),
                        },
                        max_wait_time_sec=None,  # Workers wait indefinitely
                        poll_interval_sec=FURU_CONFIG.poll_interval,
                        wait_log_every_sec=FURU_CONFIG.wait_log_every_sec,
                        reconcile_fn=lambda d: self._reconcile(d),
                        allow_failed=allow_failed_effective,
                        allow_success=allow_success,
                    ) as ctx:
                        self._prepare_executor_rerun(directory)
                        if not always_rerun:
                            exists_ok = self._exists_quiet()
                            if not exists_ok:
                                state = self.get_state(directory)
                                if isinstance(state.result, _StateResultSuccess):
                                    self._invalidate_cached_success(
                                        directory,
                                        reason="_validate returned false (worker)",
                                    )

                        stage = "metadata"
                        try:
                            # Refresh metadata (now safe - attempt is already recorded)
                            metadata = MetadataManager.create_metadata(self, directory)
                            MetadataManager.write_metadata(metadata, directory)

                            # Set up signal handlers
                            stage = "signal handler setup"
                            self._setup_signal_handlers(
                                directory,
                                ctx.stop_heartbeat,
                                attempt_id=ctx.attempt_id,
                            )

                            stage = "_create"
                            # Run computation
                            logger.debug(
                                "_create: begin %s %s %s",
                                self.__class__.__name__,
                                self.furu_hash,
                                directory,
                            )
                            self._create()
                            logger.debug(
                                "_create: ok %s %s %s",
                                self.__class__.__name__,
                                self.furu_hash,
                                directory,
                            )
                            StateManager.write_success_marker(
                                directory, attempt_id=ctx.attempt_id
                            )
                            StateManager.finish_attempt_success(
                                directory, attempt_id=ctx.attempt_id
                            )
                            logger.info(
                                "_create ok %s %s",
                                self.__class__.__name__,
                                self.furu_hash,
                                extra={"furu_console_only": True},
                            )
                        except Exception as e:
                            if stage == "_create":
                                logger.error(
                                    "_create failed %s %s %s",
                                    self.__class__.__name__,
                                    self.furu_hash,
                                    directory,
                                    extra={"furu_file_only": True},
                                )
                            else:
                                logger.error(
                                    "attempt failed (%s) %s %s %s",
                                    stage,
                                    self.__class__.__name__,
                                    self.furu_hash,
                                    directory,
                                    extra={"furu_file_only": True},
                                )
                            logger.error(
                                "%s",
                                format_traceback(e),
                                extra={"furu_file_only": True},
                            )

                            tb = "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                            StateManager.finish_attempt_failed(
                                directory,
                                attempt_id=ctx.attempt_id,
                                error={
                                    "type": type(e).__name__,
                                    "message": str(e),
                                    "traceback": tb,
                                },
                            )
                            self._add_exception_breadcrumbs(e, directory)
                            if stage != "_create":
                                message = (
                                    "Failed to create metadata"
                                    if stage == "metadata"
                                    else "Failed to set up signal handlers"
                                )
                                raise FuruComputeError(
                                    message,
                                    StateManager.get_state_path(directory),
                                    e,
                                ) from e
                            raise
                except FuruLockNotAcquired as exc:
                    # Experiment already completed; succeed if success, fail if failed.
                    state = StateManager.read_state(directory)
                    state_path = StateManager.get_state_path(directory)
                    attempt = state.attempt
                    attempt_info = "no active attempt"
                    if attempt is not None:
                        attempt_info = (
                            f"attempt {attempt.id} status {attempt.status} "
                            f"backend {attempt.backend}"
                        )
                    hints = [
                        f"Furu hash: {self.furu_hash}",
                        f"Directory: {directory}",
                        f"State file: {state_path}",
                        f"Attempt: {attempt_info}",
                    ]
                    if isinstance(state.result, _StateResultSuccess):
                        return
                    if isinstance(state.result, _StateResultFailed):
                        if allow_failed_effective:
                            return
                        raise FuruComputeError(
                            "Worker refused to run: experiment already failed",
                            state_path,
                            exc,
                            hints=hints,
                        ) from exc
                    raise FuruLockNotAcquired(
                        "Worker refused to run: experiment already running elsewhere",
                        hints=hints,
                    ) from exc
            finally:
                EXEC_CONTEXT.reset(exec_token)

    def _collect_submitit_env(self: Self) -> _SubmititEnvInfo:
        """Collect submitit/slurm environment information."""
        slurm_id = os.getenv("SLURM_JOB_ID")

        info: _SubmititEnvInfo = {
            "backend": "slurm" if slurm_id else "local",
            "slurm_job_id": slurm_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command": " ".join(sys.argv) if sys.argv else "<unknown>",
        }

        # Only call submitit.JobEnvironment() when actually in a submitit job
        if slurm_id:
            env = submitit.JobEnvironment()
            info["backend"] = "submitit"
            info["slurm_job_id"] = str(getattr(env, "job_id", slurm_id))

        return info

    def _run_locally(
        self: Self,
        start_time: float,
        *,
        allow_failed: bool,
        executor_mode: bool = False,
    ) -> tuple[str, bool, T | None]:
        """Run computation locally, returning (status, created_here, result)."""
        logger = get_logger()
        directory = self._base_furu_dir()

        # Calculate remaining time for the lock wait
        max_wait: float | None = None
        max_wait_time = self._effective_max_wait_time_sec()
        if max_wait_time is not None:
            elapsed = time.time() - start_time
            max_wait = max(0.0, max_wait_time - elapsed)

        try:
            with compute_lock(
                directory,
                backend="local",
                lease_duration_sec=FURU_CONFIG.lease_duration_sec,
                heartbeat_interval_sec=FURU_CONFIG.heartbeat_interval_sec,
                owner={
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "user": getpass.getuser(),
                    "command": " ".join(sys.argv) if sys.argv else "<unknown>",
                },
                scheduler={},
                max_wait_time_sec=max_wait,
                poll_interval_sec=FURU_CONFIG.poll_interval,
                wait_log_every_sec=FURU_CONFIG.wait_log_every_sec,
                reconcile_fn=lambda d: self._reconcile(d),
                allow_failed=allow_failed,
            ) as ctx:
                stage = "metadata"
                try:
                    # Create metadata (now safe - attempt is already recorded)
                    metadata = MetadataManager.create_metadata(self, directory)
                    MetadataManager.write_metadata(metadata, directory)

                    # Set up preemption handler
                    stage = "signal handler setup"
                    self._setup_signal_handlers(
                        directory, ctx.stop_heartbeat, attempt_id=ctx.attempt_id
                    )

                    stage = "_create"
                    # Run the computation
                    logger.debug(
                        "_create: begin %s %s %s",
                        self.__class__.__name__,
                        self.furu_hash,
                        directory,
                    )
                    token = None
                    if executor_mode:
                        from furu.execution.context import EXEC_CONTEXT, ExecContext

                        token = EXEC_CONTEXT.set(
                            ExecContext(
                                mode="executor",
                                spec_key=self._executor_spec_key(),
                                backend="local",
                                current_node_hash=self.furu_hash,
                            )
                        )
                    try:
                        result = self._create()
                    finally:
                        if token is not None:
                            EXEC_CONTEXT.reset(token)
                    logger.debug(
                        "_create: ok %s %s %s",
                        self.__class__.__name__,
                        self.furu_hash,
                        directory,
                    )
                    StateManager.write_success_marker(
                        directory, attempt_id=ctx.attempt_id
                    )
                    StateManager.finish_attempt_success(
                        directory, attempt_id=ctx.attempt_id
                    )
                    logger.info(
                        "_create ok %s %s",
                        self.__class__.__name__,
                        self.furu_hash,
                        extra={"furu_console_only": True},
                    )
                    return "success", True, result
                except Exception as e:
                    if stage == "_create":
                        logger.error(
                            "_create failed %s %s %s",
                            self.__class__.__name__,
                            self.furu_hash,
                            directory,
                            extra={"furu_file_only": True},
                        )
                    else:
                        logger.error(
                            "attempt failed (%s) %s %s %s",
                            stage,
                            self.__class__.__name__,
                            self.furu_hash,
                            directory,
                            extra={"furu_file_only": True},
                        )
                    logger.error(
                        "%s", format_traceback(e), extra={"furu_file_only": True}
                    )

                    # Record failure (plain text in file)
                    tb = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                    StateManager.finish_attempt_failed(
                        directory,
                        attempt_id=ctx.attempt_id,
                        error={
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": tb,
                        },
                    )
                    self._add_exception_breadcrumbs(e, directory)
                    if stage != "_create":
                        message = (
                            "Failed to create metadata"
                            if stage == "metadata"
                            else "Failed to set up signal handlers"
                        )
                        raise FuruComputeError(
                            message,
                            StateManager.get_state_path(directory),
                            e,
                        ) from e
                    raise
        except FuruLockNotAcquired:
            # Lock couldn't be acquired because experiment already completed
            state = StateManager.read_state(directory)
            if isinstance(state.result, _StateResultSuccess):
                return "success", False, None
            if isinstance(state.result, _StateResultFailed):
                return "failed", False, None
            # Shouldn't happen, but re-raise if state is unexpected
            raise

    def _reconcile(
        self: Self, directory: Path, *, adapter: SubmititAdapter | None = None
    ) -> None:
        if adapter is None:
            StateManager.reconcile(directory)
            return

        StateManager.reconcile(
            directory,
            submitit_probe=lambda state: adapter.probe(directory, state),
        )

    def _setup_signal_handlers(
        self,
        directory: Path,
        stop_heartbeat: Callable[[], None],
        *,
        attempt_id: str,
    ) -> None:
        """Set up signal handlers for graceful preemption."""
        if threading.current_thread() is not threading.main_thread():
            return

        def handle_signal(signum: int, frame: FrameType | None) -> None:
            try:
                StateManager.finish_attempt_preempted(
                    directory,
                    attempt_id=attempt_id,
                    error={"type": "signal", "message": f"signal:{signum}"},
                )
            finally:
                stop_heartbeat()
                exit_code = 143 if signum == signal.SIGTERM else 130
                os._exit(exit_code)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handle_signal)


class DependencyChzSpec(Protocol):
    __chz_fields__: dict[str, ChzField]


DependencySequence: TypeAlias = Sequence[Furu]
DependencySet: TypeAlias = AbstractSet[Furu]
DependencyMapping: TypeAlias = Mapping[str, Furu]
DependencyCollection: TypeAlias = DependencySequence | DependencySet | DependencyMapping
DependencyValue: TypeAlias = Furu | DependencyCollection
DependencySpec: TypeAlias = DependencyValue | DependencyChzSpec
DependencyLeaf: TypeAlias = str | int | float | bool | None | Path | bytes
DependencyScanValue: TypeAlias = (
    DependencyLeaf
    | Furu
    | Mapping[Hashable, "DependencyScanValue"]
    | Sequence["DependencyScanValue"]
    | AbstractSet["DependencyScanValue"]
    | DependencyChzSpec
)


def _collect_dependencies(
    obj: Furu,
    dependencies: list[Furu],
    seen: set[str],
    *,
    recursive: bool,
) -> None:
    for dependency in _direct_dependencies(obj):
        digest = dependency.furu_hash
        if digest in seen:
            continue
        seen.add(digest)
        dependencies.append(dependency)
        if recursive:
            _collect_dependencies(
                dependency,
                dependencies,
                seen,
                recursive=recursive,
            )


def _direct_dependencies(obj: Furu) -> list[Furu]:
    dependencies: list[Furu] = []
    for field in chz.chz_fields(obj).values():
        value = cast(DependencyScanValue, getattr(obj, field.logical_name))
        dependencies.extend(_collect_dependencies_from_value(value))
    extra = obj._dependencies()
    if extra is not None:
        dependencies.extend(_collect_dependencies_from_spec(extra, path="dependencies"))
    return dependencies


def _collect_dependencies_from_value(value: DependencyScanValue) -> list[Furu]:
    dependencies: list[Furu] = []
    if isinstance(value, Furu):
        dependencies.append(value)
        return dependencies
    if isinstance(value, dict):
        mapping = cast(Mapping[Hashable, DependencyScanValue], value)
        for item in mapping.values():
            dependencies.extend(_collect_dependencies_from_value(item))
        return dependencies
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[DependencyScanValue], value)
        for item in sequence:
            dependencies.extend(_collect_dependencies_from_value(item))
        return dependencies
    if isinstance(value, (set, frozenset)):
        items = _sorted_dependency_set(cast(AbstractSet[DependencyScanValue], value))
        for item in items:
            dependencies.extend(_collect_dependencies_from_value(item))
        return dependencies
    if chz.is_chz(value):
        for field in chz.chz_fields(value).values():
            field_value = cast(DependencyScanValue, getattr(value, field.logical_name))
            dependencies.extend(_collect_dependencies_from_value(field_value))
    return dependencies


def _collect_dependencies_from_spec(value: DependencySpec, path: str) -> list[Furu]:
    if isinstance(value, Furu):
        return [value]
    if isinstance(value, dict):
        return _collect_dependencies_from_mapping(
            cast(Mapping[Hashable, DependencyValue], value),
            path,
        )
    if isinstance(value, (list, tuple)):
        return _collect_dependencies_from_sequence(
            cast(Sequence[DependencyValue], value),
            path,
        )
    if isinstance(value, (set, frozenset)):
        return _collect_dependencies_from_set(
            cast(AbstractSet[DependencyValue], value),
            path,
        )
    if chz.is_chz(value):
        dependencies: list[Furu] = []
        for field in chz.chz_fields(value).values():
            field_value = getattr(value, field.logical_name)
            field_path = f"{path}.{field.logical_name}"
            dependencies.extend(
                _collect_dependencies_from_value_spec(field_value, field_path)
            )
        return dependencies
    raise _dependency_type_error(path, value)


def _collect_dependencies_from_value_spec(
    value: DependencyValue,
    path: str,
) -> list[Furu]:
    if isinstance(value, Furu):
        return [value]
    if isinstance(value, dict):
        return _collect_dependencies_from_mapping(
            cast(Mapping[Hashable, DependencyValue], value),
            path,
        )
    if isinstance(value, (list, tuple)):
        return _collect_dependencies_from_sequence(
            cast(Sequence[DependencyValue], value),
            path,
        )
    if isinstance(value, (set, frozenset)):
        return _collect_dependencies_from_set(
            cast(AbstractSet[DependencyValue], value),
            path,
        )
    raise _dependency_type_error(path, value)


def _collect_dependencies_from_mapping(
    mapping: Mapping[Hashable, DependencyValue],
    path: str,
) -> list[Furu]:
    dependencies: list[Furu] = []
    for key, item in mapping.items():
        if not isinstance(item, Furu):
            raise _dependency_type_error(f"{path}[{key!r}]", item)
        dependencies.append(item)
    return dependencies


def _collect_dependencies_from_sequence(
    sequence: Sequence[DependencyValue],
    path: str,
) -> list[Furu]:
    dependencies: list[Furu] = []
    for index, item in enumerate(sequence):
        if not isinstance(item, Furu):
            raise _dependency_type_error(f"{path}[{index}]", item)
        dependencies.append(item)
    return dependencies


def _collect_dependencies_from_set(
    values: AbstractSet[DependencyValue],
    path: str,
) -> list[Furu]:
    dependencies: list[Furu] = []
    ordered = sorted(
        list(cast(AbstractSet[DependencyScanValue], values)),
        key=_dependency_sort_key,
    )
    for index, item in enumerate(ordered):
        if not isinstance(item, Furu):
            raise _dependency_type_error(f"{path}[{index}]", item)
        dependencies.append(item)
    return dependencies


def _sorted_dependency_set(
    values: AbstractSet[DependencyScanValue],
) -> list[DependencyScanValue]:
    return sorted(list(values), key=_dependency_sort_key)


def _dependency_sort_key(value: DependencyScanValue) -> tuple[int, str]:
    if isinstance(value, Furu):
        return (0, cast(str, value.furu_hash))
    return (1, f"{type(value).__name__}:{value!r}")


def _dependency_type_error(
    path: str,
    value: DependencySpec | DependencyValue | DependencyScanValue,
) -> TypeError:
    return TypeError(
        f"{path} must be a Furu instance or a collection of Furu instances; "
        f"got {type(value).__name__}"
    )


_H = TypeVar("_H", bound=Furu, covariant=True)