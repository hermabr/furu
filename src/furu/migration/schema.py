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
takes no lock. The first line that records a default is the one that counts,
so a writer that lost the lock while it held it cannot change one.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furu._declared_types import cached_type_hints
from furu.config import get_config
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


# A lock holder is done in milliseconds, so a lock this old was left by a killed one.
_STALE_S = 60.0
_POLL_S = 0.05


def _missing(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESTALE)


@contextmanager
def short_lock(path: Path) -> Generator[None]:
    """Hold ``path`` while the body runs: creating the file is taking the lock.

    Nothing keeps the lock alive, so a holder that stops for longer than
    ``_STALE_S`` can find that another took it. What the lock guards has to
    stay right then too.
    """
    mine = secrets.token_hex(8)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > _STALE_S:
                    path.unlink()
                    continue
            except OSError as exc:  # released since
                if not _missing(exc):
                    raise
            time.sleep(_POLL_S)
    # Closed before the body: a file removed while it is open stays on the
    # file server under another name. Writing a token identifies this holder
    # even if the filesystem reuses the inode number after a stale break.
    try:
        os.write(fd, mine.encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        # Another's by now if this one was taken for stale.
        try:
            if path.read_text(encoding="utf-8") == mine:
                path.unlink()
        except OSError as exc:
            if not _missing(exc):
                raise


def _read(path: Path) -> list[dict[str, Any]] | None:
    """The file's whole lines; None when there is no file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        if _missing(exc):
            return None
        raise
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
    """The default each step was first recorded with."""
    recorded: dict[_Step, JsonValue] = {}
    for line in lines:
        if "added" in line:
            step = (line.get("nested"), line.get("class_part"), line["added"])
            recorded.setdefault(step, line["default"])
    return recorded


def _layered(class_dir: Path) -> list[dict[str, Any]]:
    """The lines of a class folder's file and, in an overlay, of the one under it."""
    return [
        line
        for layer in get_config().layers(class_dir)
        for line in _read(schema_path_in(layer)) or ()
    ]


def check_defaults(class_dir: Path, cls: type[Spec], *, record: bool = False) -> None:
    """Stop when a default the source declares is not the one recorded.

    A declared default with no line gets one. A folder with no file gets one
    only with ``record``, when the folder is made: until then nothing in it was
    stored under any default. In an overlay the file under it is only read,
    and a default recorded in either has to match.
    """
    key = (cls, class_dir, len(held[cls]))
    if (*key, record) in _checked:
        return
    declared = _declared_defaults(cls)
    path = schema_path_in(class_dir)
    layers = get_config().layers(class_dir) if declared or record else ()
    files = {schema_path_in(layer): _read(schema_path_in(layer)) for layer in layers}
    found = any(lines is not None for lines in files.values())

    def unrecorded() -> list[dict[str, Any]]:
        recorded = _recorded_defaults(
            [line for lines in files.values() for line in lines or ()]
        )
        return [
            {"added": name, "default": default}
            | ({} if nested is None else {"nested": nested, "class_part": part})
            for (nested, part, name), (_, default) in declared.items()
            if (nested, part, name) not in recorded
        ]

    if found or record:
        if (record and not files[path]) or unrecorded():
            class_dir.mkdir(parents=True, exist_ok=True)  # new in an overlay
            with short_lock(schema_lock_path_in(class_dir)):
                files[path] = _read(path)
                new = unrecorded()
                if not files[path]:  # every file starts with the names
                    new.insert(0, {"names": list(naming(cls).first_names)})
                if new:
                    _append(path, new)
                    # What counts is what the file holds: a writer that lost
                    # the lock while it held it can have recorded first.
                    files[path] = _read(path)
        for file, lines in files.items():
            recorded = _recorded_defaults(lines or [])
            for step, (label, default) in declared.items():
                if step in recorded and (
                    _stable_json_dump(recorded[step]) != _stable_json_dump(default)
                ):
                    raise MigrationError(
                        f"{label} declares the default {_stable_json_dump(default)}, "
                        f"but {file} recorded {_stable_json_dump(recorded[step])} "
                        "when the step was first seen. A spec at the declared "
                        "default is named as it was before the field existed, so it "
                        "would load a result computed at the recorded one. Put the "
                        "recorded default back in the step."
                    )
        _checked.add((*key, True))
    # A folder with no file may get one from a checkout that declares another
    # default, so a declared default is looked for again.
    if found or record or not declared:
        _checked.add((*key, False))


def stored_names(class_dir: Path) -> list[str] | None:
    """The names a class folder's results are stored under, when its file says."""
    for line in _layered(class_dir):
        if "names" in line and "nested" not in line:
            return line["names"]
    return None


def nested_names(class_dir: Path) -> dict[str, set[tuple[str, ...]]]:
    """Each set of first names a dataclass was stored with in a class folder."""
    recorded: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    for line in _layered(class_dir):
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
