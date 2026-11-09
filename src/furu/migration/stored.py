from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from furu._declared_types import declared_result_type
from furu.code_trace import valid_version
from furu.constants import FIELDSMARKER
from furu.metadata import ArtifactSpec
from furu.migration.resolution import (
    _apply_child_moves,
    _apply_steps,
    _ChildMove,
    _class_resolution,
    _ClassResolution,
    _Covered,
)
from furu.migration.steps import MigrationStep
from furu.result.bundle import load_result_bundle
from furu.storage._layout import data_dir_in, result_dir_in, spec_path_in
from furu.utils import JsonFields, _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec


@dataclass(slots=True)
class _SourceIndex:
    """The identity directories of one covered schema directory, by their fields
    migrated to the current schema and serialized.

    An identity directory's spec.json never changes, so entries never go stale.
    Directories published later are picked up by rescanning on a miss, and only
    when the schema directory's mtime says its listing changed: listing a large
    directory for every missing spec would dominate planning.
    """

    by_key: dict[str, list[Path]] = field(default_factory=dict)
    indexed: set[Path] = field(default_factory=set)
    # The schema directory's mtime at the last scan that left nothing to
    # revisit; None rescans on the next miss.
    scanned_mtime_ns: int | None = None

    def rescan(self, resolution: _ClassResolution, covered: _Covered) -> bool:
        """Index identity directories that appeared; False if none did."""
        try:
            mtime_ns = covered.schema_directory.stat().st_mtime_ns
        except OSError:
            return False
        if mtime_ns == self.scanned_mtime_ns:
            return False
        # Nested instances of the class itself share the outer generation.
        moves = {
            **covered.child_moves,
            covered.generation.class_name: _ChildMove(
                resolution.own, covered.generation
            ),
        }
        found = False
        settled = True
        for identity_dir in covered.schema_directory.iterdir():
            if identity_dir in self.indexed or not identity_dir.is_dir():
                continue
            spec_path = spec_path_in(identity_dir)
            if not spec_path.exists():
                # Its spec.json may still be on its way, which would not touch
                # the schema directory's mtime again.
                settled = False
                continue
            artifact = ArtifactSpec.model_validate_json(
                spec_path.read_text(encoding="utf-8")
            )
            fields = cast(JsonFields, artifact.artifact_data[FIELDSMARKER])
            fields = {
                name: _apply_child_moves(value, moves) for name, value in fields.items()
            }
            fields = _apply_steps(resolution.own, covered.generation, fields)
            self.by_key.setdefault(_stable_json_dump(fields), []).append(identity_dir)
            self.indexed.add(identity_dir)
            found = True
        self.scanned_mtime_ns = mtime_ns if settled else None
        return found


_SOURCE_INDEXES: dict[tuple[type, Path], _SourceIndex] = {}


def _find_source(obj: Spec, resolution: _ClassResolution) -> Path | None:
    """A valid version in an older, covered schema that migrates to obj."""
    target_fields = cast(JsonFields, obj._artifact_data[FIELDSMARKER])
    target_key = _stable_json_dump(target_fields)
    for covered in resolution.covered:
        if any(
            # Serialized so NaN defaults compare equal, like the key above.
            _stable_json_dump(target_fields[name]) != _stable_json_dump(value)
            for name, value in covered.generation.pinned.items()
        ):
            continue
        index = _SOURCE_INDEXES.setdefault(
            (type(obj), covered.schema_directory), _SourceIndex()
        )
        while True:
            for identity_dir in sorted(index.by_key.get(target_key, ())):
                version = valid_version(
                    identity_dir, None, storage_root=obj._metadata.storage
                )
                if version is not None:
                    return version
            if not index.rescan(resolution, covered):
                break
    return None


def own_version(obj: Spec) -> Path | None:
    return valid_version(
        obj._base_dir,
        obj._metadata.code_version,
        storage_root=obj._metadata.storage,
    )


def stored_version(obj: Spec) -> Path | None:
    """The version directory holding obj's result: its own, or a migration source.

    The one definition of "done"; a pure query that writes nothing.
    """
    if (version := own_version(obj)) is not None:
        return version
    return _find_source(obj, _class_resolution(obj))


def load_stored_result[T](obj: Spec[T], version: Path) -> T:
    steps: tuple[MigrationStep, ...] = ()
    if version.parent != obj._base_dir:
        resolution = _class_resolution(obj)
        covered = next(
            (
                entry
                for entry in resolution.covered
                if entry.schema_directory == version.parent.parent
            ),
            None,
        )
        # stored_version only hands out covered sources.
        assert covered is not None, f"{version} is not covered by {obj._log_label}"
        steps = tuple(
            resolution.own.steps[index] for index in covered.generation.pending
        )
    value = load_result_bundle(
        result_dir_in(version),
        data_dir=data_dir_in(version),
        declared_type=declared_result_type(type(obj)),
        rewrites=[
            step.result_rewrite for step in steps if step.result_rewrite is not None
        ],
    )
    return cast(T, value)
