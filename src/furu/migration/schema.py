"""schema.ndjson: what a class folder records about the class stored in it.

The first line holds the names the class part is a hash of. Each later line is
one ``Added`` step and the default it declared, for a field or dependency
method of the class or for a field of a dataclass its specs hold, or one set
of field names that such a dataclass was stored with.

A spec at a declared default has the id of a result stored before the field
existed, so a spec at an edited default would load a result computed at
another value. The recorded default is what the source is compared with.

A dataclass's field names are part of the value of the field that holds it,
so the class part does not cover them. They are recorded before a spec that
holds the dataclass is created, and the recorded names are what a launch
compares the source with, as it compares a spec class with its folders.

The file only grows, by whole lines: a writer holds schema.lock and a reader
takes no lock. The first line recorded for a step is the one that counts.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furu._declared_types import cached_type_hints
from furu.locking import short_lock
from furu.migration.steps import MigrationError, describe, naming, stored_name
from furu.serializer.artifact import to_json
from furu.storage._layout import schema_lock_path_in, schema_path_in
from furu.utils import JsonValue, _stable_json_dump

if TYPE_CHECKING:
    from furu.core import Spec

# The nested dataclass that declares it, as its name and class part, and the
# step's name. Both are None for the spec class, whose class part is the folder.
type _Step = tuple[str | None, str | None, str]

# The dataclasses that values of each spec class have held, in this process.
held: defaultdict[type, set[type]] = defaultdict(set)

# What this process has checked: a class, its folder, how many dataclasses its
# values had held by then, and whether a folder with no file was given one.
_checked: set[tuple[type, Path, int, bool]] = set()

# The lines of nested names this process has seen in a class folder's file: the
# folder, the dataclass's stored name and its first names.
_nested_recorded: set[tuple[Path, str, tuple[str, ...]]] = set()


def _declared_defaults(cls: type[Spec]) -> dict[_Step, tuple[str, JsonValue]]:
    """Each default the source declares, as written down, with a label for its step."""
    declared: dict[_Step, tuple[str, JsonValue]] = {}
    for owner in (cls, *sorted(held[cls], key=stored_name)):
        hints = cached_type_hints(owner)
        nested = (
            (None, None)
            if owner is cls
            else (stored_name(owner), naming(owner).class_part)
        )
        for attribute, name, added in naming(owner).fields:
            if added is not None:
                declared[*nested, name] = (
                    f"{owner.__name__}.migrations: {describe(added)}",
                    to_json(
                        added.default,
                        hints.get(attribute, Any),
                        cls.artifact_serializers,
                        set(),
                    ),
                )
    return declared


def _read(path: Path) -> list[dict[str, Any]] | None:
    """The file's whole lines; None when there is no file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    lines = []
    for line in text.splitlines():
        try:
            lines.append(json.loads(line))
        except ValueError:  # blank, or cut short when its writer was killed
            continue
    return lines


def _append(path: Path, lines: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as file:
        # On a line of its own even when the writer before this one was killed
        # in the middle of a line.
        if file.tell():
            file.write("\n")
        file.write("".join(_stable_json_dump(line) + "\n" for line in lines))
        file.flush()
        os.fsync(file.fileno())


def _recorded_defaults(lines: list[dict[str, Any]]) -> dict[_Step, JsonValue]:
    recorded: dict[_Step, JsonValue] = {}
    for line in lines:
        if "added" in line:
            step = (line.get("nested"), line.get("class_part"), line["added"])
            recorded.setdefault(step, line["default"])
    return recorded


def check_defaults(class_dir: Path, cls: type[Spec], *, record: bool = False) -> None:
    """Stop when a default the source declares is not the one recorded.

    A declared default with no line gets one. A folder with no file gets one
    only with ``record``, when the folder is made: until then nothing in it was
    stored under any default.
    """
    key = (cls, class_dir, len(held[cls]))
    if (*key, record) in _checked:
        return
    declared = _declared_defaults(cls)
    path = schema_path_in(class_dir)
    lines = _read(path) if declared or record else None
    if lines is not None or record:
        if not lines or declared.keys() - _recorded_defaults(lines).keys():
            with short_lock(schema_lock_path_in(class_dir)):
                lines = _read(path) or []
                recorded = _recorded_defaults(lines)
                new: list[dict[str, Any]] = [
                    {"added": name, "default": default}
                    | ({} if nested is None else {"nested": nested, "class_part": part})
                    for (nested, part, name), (_, default) in declared.items()
                    if (nested, part, name) not in recorded
                ]
                if not lines:
                    new.insert(0, {"names": list(naming(cls).first_names)})
                if new:
                    _append(path, new)
                lines += new
        recorded = _recorded_defaults(lines)
        for step, (label, default) in declared.items():
            if _stable_json_dump(recorded[step]) != _stable_json_dump(default):
                raise MigrationError(
                    f"{label} declares the default {_stable_json_dump(default)}, "
                    f"but {path} recorded {_stable_json_dump(recorded[step])} "
                    "when the step was first seen. A spec at the declared "
                    "default is named as it was before the field existed, so it "
                    "would load a result computed at the recorded one. Put the "
                    "recorded default back in the step."
                )
        _checked.add((*key, True))
    # A folder with no file may get one from a checkout that declares another
    # default, so a declared default is looked for again.
    if lines is not None or not declared:
        _checked.add((*key, False))


def stored_names(class_dir: Path) -> list[str] | None:
    """The names a class folder's results are stored under, when its file says."""
    for line in _read(schema_path_in(class_dir)) or ():
        if "names" in line and "nested" not in line:
            return line["names"]
    return None


def nested_names(class_dir: Path) -> dict[str, set[tuple[str, ...]]]:
    """Each set of first names a dataclass was stored with in a class folder."""
    recorded: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    for line in _read(schema_path_in(class_dir)) or ():
        if "names" in line and "nested" in line:
            recorded[line["nested"]].add(tuple(line["names"]))
    return recorded


def record_nested_names(class_dir: Path, holds: dict[type, tuple[str, ...]]) -> None:
    """Give the names each held dataclass has now a line, if they have none.

    ``holds`` says where a spec stored in ``class_dir`` holds each: the fields
    that lead to it.
    """
    lines = {
        (class_dir, stored_name(nested), naming(nested).first_names): at
        for nested, at in holds.items()
    }
    if lines.keys() <= _nested_recorded:
        return

    def unrecorded() -> list[dict[str, Any]]:
        recorded = nested_names(class_dir)
        return [
            {"nested": name, "at": list(at), "names": list(names)}
            for (_, name, names), at in lines.items()
            if names not in recorded.get(name, ())
        ]

    if unrecorded():
        with short_lock(schema_lock_path_in(class_dir)):
            if new := unrecorded():
                _append(schema_path_in(class_dir), new)
    _nested_recorded.update(lines)
