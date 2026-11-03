from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Literal, cast

from furu.constants import CLASSMARKER, FIELDSMARKER
from furu.migration.links import _find_source
from furu.migration.resolution import _class_resolution, _ClassResolution, _Orphaned
from furu.migration.steps import Stale
from furu.storage._layout import schema_snapshot_path_in
from furu.utils import JsonValue, _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec


def _orphaned_directories(resolution: _ClassResolution) -> list[_Orphaned]:
    return [
        orphaned
        for orphaned in resolution.orphaned
        if schema_snapshot_path_in(orphaned.schema_directory).exists()
    ]


def sideways_status(obj: Spec) -> Literal["done", "stale", "missing"]:
    resolution = _class_resolution(obj)
    if _find_source(obj, resolution) is not None:
        return "done"
    if _orphaned_directories(resolution):
        return "stale"
    return "missing"


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
            if (
                isinstance(old, dict)
                and CLASSMARKER in old
                and FIELDSMARKER in old
                and isinstance(new, dict)
                and CLASSMARKER in new
                and FIELDSMARKER in new
            ):
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


def raise_if_stale(obj: Spec) -> None:
    orphaned = _orphaned_directories(_class_resolution(obj))
    if not orphaned:
        return
    current_schema = cast(dict[str, JsonValue], obj._schema_data)
    current_fields = cast(dict[str, JsonValue], current_schema[FIELDSMARKER])
    lines = [
        (
            f"{type(obj).__name__} is stale: the store holds results under "
            f"{len(orphaned)} other schema(s) with no migration chain to the "
            "current schema."
        )
    ]
    owners: set[str] = set()
    for entry in orphaned:
        directory = entry.schema_directory
        snapshot_path = schema_snapshot_path_in(directory)
        snapshot = cast(
            dict[str, JsonValue],
            json.loads(snapshot_path.read_text(encoding="utf-8")),
        )
        lines.append(f"\norphaned: {directory}")
        if entry.clash is not None:
            lines.append(f"  {entry.clash}")
        for renamed in entry.renamed:
            lines.append(f"  {renamed}")
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
    if any(entry.clash is not None for entry in orphaned):
        lines.append(
            "\nA conflict means two branches changed the same thing in parallel, "
            "so no one order of steps fits both. Order the steps the way the "
            "branch whose results you keep made them (the other branch's results "
            "are then recomputed), or use one of the options below."
        )
    if any(entry.renamed for entry in orphaned):
        lines.append(
            "\nA recorded step that differs from a declared one only by its "
            "function or was= type usually means the step was renamed or edited "
            "after those results were stored, so furu can't tell whether they "
            "already took it. If the step still does the same thing, restore its "
            "old name to keep reusing them; otherwise delete the directory to "
            "recompute them."
        )
    lines.append(
        f"\nEither declare a migration chain on {type(obj).__name__}.migrations "
        "(Renamed/Added/MovedFrom/Retyped/Rewrite), mark the change as breaking "
        "(breaking=True) to recompute instead of reuse, or discard the orphaned "
        "results by deleting the directory above."
    )
    raise Stale("\n".join(lines))
