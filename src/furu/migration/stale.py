from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from furu.constants import FIELDSMARKER
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
    for directory in orphaned:
        snapshot_path = schema_snapshot_path_in_schema_directory(directory)
        snapshot = cast(
            dict[str, JsonValue],
            json.loads(snapshot_path.read_text(encoding="utf-8")),
        )
        lines.append(f"\norphaned: {directory}")
        old_fields = cast(dict[str, JsonValue], snapshot.get(FIELDSMARKER, {}))
        for name in sorted(old_fields.keys() | current_fields.keys()):
            if name not in old_fields:
                lines.append(f"  + {name}: {_stable_json_dump(current_fields[name])}")
            elif name not in current_fields:
                lines.append(f"  - {name}: {_stable_json_dump(old_fields[name])}")
            elif old_fields[name] != current_fields[name]:
                lines.append(
                    f"  ~ {name}: {_stable_json_dump(old_fields[name])} "
                    f"-> {_stable_json_dump(current_fields[name])}"
                )
    lines.append(
        f"\nEither declare a migration chain on {type(obj).__name__}.migrations "
        "(Renamed/Added/MovedFrom/Retyped/Rewrite), or discard the orphaned "
        "results by deleting the directory above."
    )
    raise Stale("\n".join(lines))
