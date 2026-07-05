from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast

from furu.constants import CLASSMARKER, FIELDSMARKER
from furu.migration.links import _find_source
from furu.migration.resolution import _class_resolution, _ClassResolution
from furu.migration.steps import Stale
from furu.storage._layout import schema_snapshot_path_in_schema_directory
from furu.utils import JsonValue, _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec


def _orphaned_directories(resolution: _ClassResolution) -> list[Path]:
    return [
        directory
        for directory in resolution.orphaned
        if schema_snapshot_path_in_schema_directory(directory).exists()
    ]


def sideways_status(obj: Spec[Any]) -> Literal["done", "stale", "missing"]:
    resolution = _class_resolution(obj)
    if _find_source(obj, resolution) is not None:
        return "done"
    if _orphaned_directories(resolution):
        return "stale"
    return "missing"


def _is_class_node(value: JsonValue) -> TypeGuard[dict[str, JsonValue]]:
    return isinstance(value, dict) and CLASSMARKER in value and FIELDSMARKER in value


def _field_diff(
    old_fields: Mapping[str, JsonValue],
    new_fields: Mapping[str, JsonValue],
    prefix: str,
    owner: str | None,
) -> Iterator[tuple[str, str | None]]:
    """Yield (diff line, owner) pairs; owner names the embedded class whose
    migration chain would cover the line, None for the spec's own fields."""
    for name in sorted(old_fields.keys() | new_fields.keys()):
        path = f"{prefix}{name}"
        if name not in old_fields:
            yield f"  + {path}: {_stable_json_dump(new_fields[name])}", owner
        elif name not in new_fields:
            yield f"  - {path}: {_stable_json_dump(old_fields[name])}", owner
        elif old_fields[name] != new_fields[name]:
            old, new = old_fields[name], new_fields[name]
            if _is_class_node(old) and _is_class_node(new):
                # The diff lives inside an embedded class; its innermost owner
                # is where the migration chain belongs.
                inner = cast(str, new[CLASSMARKER]).rsplit(".", 1)[-1]
                if old[CLASSMARKER] != new[CLASSMARKER]:
                    yield (
                        f"  ~ {path}: {old[CLASSMARKER]} -> {new[CLASSMARKER]}",
                        inner,
                    )
                yield from _field_diff(
                    cast(dict[str, JsonValue], old[FIELDSMARKER]),
                    cast(dict[str, JsonValue], new[FIELDSMARKER]),
                    f"{path}.",
                    inner,
                )
            else:
                yield (
                    f"  ~ {path}: {_stable_json_dump(old)} -> {_stable_json_dump(new)}",
                    owner,
                )


def raise_if_stale(obj: Spec[Any]) -> None:
    orphaned = _orphaned_directories(_class_resolution(obj))
    if not orphaned:
        return
    current_schema = cast(dict[str, JsonValue], obj._schema_data)
    current_fields = cast(dict[str, JsonValue], current_schema[FIELDSMARKER])
    lines = [
        f"{type(obj).__name__} is stale: the store holds results under "
        f"{len(orphaned)} other schema(s) with no migration chain to the "
        "current schema."
    ]
    owners: set[str] = set()
    for directory in orphaned:
        snapshot_path = schema_snapshot_path_in_schema_directory(directory)
        snapshot = cast(
            dict[str, JsonValue],
            json.loads(snapshot_path.read_text(encoding="utf-8")),
        )
        lines.append(f"\norphaned: {directory}")
        old_fields = cast(dict[str, JsonValue], snapshot.get(FIELDSMARKER, {}))
        for line, owner in _field_diff(old_fields, current_fields, "", None):
            lines.append(line)
            if owner is not None:
                owners.add(owner)
    if owners:
        named = ", ".join(sorted(owners))
        chains = " / ".join(f"{name}.migrations" for name in sorted(owners))
        lines.append(
            f"\nThe change is inside embedded {named}. Declare the chain once "
            f"on {chains}; every spec embedding it will pick it up."
        )
    lines.append(
        f"\nEither declare a migration chain on {type(obj).__name__}.migrations "
        "(Renamed/Added/MovedFrom/Retyped/Rewrite), mark the change as breaking "
        "(breaking=True) to recompute instead of reuse, or discard the orphaned "
        "results by deleting the directory above."
    )
    raise Stale("\n".join(lines))
