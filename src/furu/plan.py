"""Plan lines and plan files: each spec of a launch described once.

A launch writes ``plans/<launch id>.plan`` before anything runs. It holds a
line for each spec the launch runs and for each spec those reach through their
fields, all the way down, so that every spec a create() holds can be built
from lines. A worker gets its job's lines in the job message and never reads
the file. Specs that a create() asks for while it runs are not in the launch's
plan: the process running it saves their lines in a file of its own,
``plans/<launch id>/<worker>.plan``, before it tells anyone.
What the lines look like is in ``furu.protocol.plan_text``.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import fields
from itertools import chain
from pathlib import Path
from typing import Any, cast

from furu._declared_types import cached_type_hints
from furu._tree import specs_in
from furu.config import get_config
from furu.constants import KINDMARKER, VALUEMARKER
from furu.core import Spec
from furu.dependencies import creation, returned_attributes
from furu.migration.schema import held
from furu.migration.steps import naming
from furu.protocol.plan_text import File, Line, dump, pointer
from furu.provenance import SubmitProvenance
from furu.serializer.artifact import (
    _PLAIN,
    OWN_LINE_BYTES,
    check_names,
    from_json,
    remembered,
    to_json,
)
from furu.utils import (
    JsonValue,
    _fingerprint,
    _hash_dict_deterministically,
    fully_qualified_name,
    resolve_fully_qualified_name,
)


def lines(
    roots: Iterable[Spec], runs: Collection[str] = (), *, reach: bool = True
) -> Iterator[Line]:
    """The lines of ``roots`` and of every spec they reach through fields, each once.

    ``runs`` holds the ids of the specs that the launch computes. Their lines
    say so, and what their dependency methods returned is written out. The
    table of moved classes comes last, when a class has moved. Without
    ``reach``, the specs that ``roots`` hold get no lines.
    """
    written: set[str] = set()
    # The fingerprints of field values whose specs are pending or written.
    walked: set[str] = set()
    classes: set[type] = set()
    pending = list(roots)[::-1]
    while pending:
        obj = pending.pop()
        if (spec_id := obj.object_id) in written:
            continue
        written.add(spec_id)
        cls = type(obj)
        classes.add(cls)
        hints = cached_type_hints(cls)
        returned = returned_attributes(cls)
        serializers, nested = obj.artifact_serializers, held[cls]
        fields: dict[str, JsonValue] = {}
        left_out: dict[str, JsonValue] = {}
        for attribute, name, added in naming(cls).fields:
            declared = hints.get(attribute, Any)
            default = (
                None
                if added is None
                else to_json(added.default, declared, serializers, nested)
            )
            if attribute in returned:
                fingerprint = obj._returned.get(attribute)
                if added is not None and fingerprint in (
                    None,
                    _hash_dict_deterministically(default),
                ):
                    left_out[name] = default
                    continue
                assert fingerprint is not None
                kind = "returned"
                new = spec_id in runs and fingerprint not in written
                value = (
                    to_json(getattr(obj, attribute), declared, serializers, nested)
                    if new
                    else None
                )
            else:
                held_value = getattr(obj, attribute)
                fingerprint = remembered(held_value, cls, attribute)
                if added is None and fingerprint in walked:
                    # Written for a spec that holds the same value.
                    fields[name] = {KINDMARKER: "value", VALUEMARKER: fingerprint}
                    continue
                value = to_json(held_value, declared, serializers, nested)
                text = dump(value)
                if added is not None and text == dump(default):
                    left_out[name] = default
                    continue
                if reach and type(held_value) not in _PLAIN:
                    pending += specs_in(held_value)[::-1]
                if len(text) <= OWN_LINE_BYTES:
                    fields[name] = value
                    continue
                fingerprint = _fingerprint(text)
                walked.add(fingerprint)
                kind = "value"
                new = fingerprint not in written
            fields[name] = {KINDMARKER: kind, VALUEMARKER: fingerprint}
            if new:
                written.add(fingerprint)
                yield {"value": fingerprint, "is": value}
        line: Line = {"spec": spec_id, "fields": fields}
        if left_out:
            line["left_out"] = left_out
        if spec_id in runs:
            line["runs"] = True
        yield line
    if moved := {
        old: fully_qualified_name(moved_cls)
        for cls in classes
        for moved_cls in (cls, *held[cls])
        if (old := naming(moved_cls).moved_from) is not None
    }:
        yield {"moved": moved}


class Specs:
    """The specs that lines describe, each built the first time it is asked for.

    A spec gets its id from its line and is not hashed. Its line's fingerprint
    of what each dependency method returned is compared when the field is read.
    """

    def __init__(self, lines: Iterable[Line]) -> None:
        self._lines: dict[str, Line] = {}
        self._values: dict[str, JsonValue] = {}
        self._moved: dict[str, str] = {}
        self._built: dict[str, Spec] = {}
        for line in lines:
            match line:
                case {"spec": spec_id}:
                    self._lines[spec_id] = line
                case {"value": fingerprint, "is": value}:
                    self._values[fingerprint] = value
                case {"moved": moved}:
                    self._moved |= moved

    def __getitem__(self, spec_id: str) -> Spec:
        if spec_id not in self._built:
            self._built[spec_id] = self._build(spec_id)
        return self._built[spec_id]

    def _build(self, spec_id: str) -> Spec:
        line = self._lines[spec_id]
        written: dict[str, JsonValue] = line["fields"]
        noted: dict[str, JsonValue] = line.get("left_out", {})
        stored, _, artifact_hash = spec_id.split(":")
        cls = cast(
            type[Spec], resolve_fully_qualified_name(self._moved.get(stored, stored))
        )
        # The steps forwards: the line fits only a class with the same names.
        check_names(cls, written)
        hints = cached_type_hints(cls)
        takes = {f.name for f in fields(cls) if f.init}
        values: dict[str, Any] = {}
        returned: dict[str, str] = {}
        for attribute, name, added in naming(cls).fields:
            declared = hints.get(attribute, Any)
            default = (
                None
                if added is None
                else to_json(
                    added.default, declared, cls.artifact_serializers, held[cls]
                )
            )
            value = written.get(name)
            points_to = pointer(value)
            if attribute in returned_attributes(cls):
                returned[attribute] = points_to or _hash_dict_deterministically(default)
            elif attribute not in takes:
                continue
            elif name in written:
                values[attribute] = from_json(
                    value if points_to is None else self._values[points_to],
                    declared,
                    held[cls],
                    self.__getitem__,
                    self._moved,
                )
            else:
                # Filled from the declared step, never from the note.
                assert added is not None
                if name in noted and dump(noted[name]) != dump(default):
                    raise ValueError(
                        f"the line of {spec_id} left {name} out at "
                        f"{dump(noted[name])}, and {cls.__name__}.migrations "
                        f"declares the default {dump(default)}"
                    )
                values[attribute] = added.default
        obj = cls(**values)
        vars(obj).update(_named=(artifact_hash, returned), _from_line=True)
        return obj


def write(
    path: Path, asked: Sequence[Spec], described: Iterable[Line], code: SubmitProvenance
) -> list[str]:
    """Save a launch's plan: what it was asked for, its code, and its lines.

    Returns the lines as written, which is how the manager is sent them.
    """
    header = {
        "asked": [obj.object_id for obj in asked],
        "code": {
            "repository": code.git.remote,
            "commit": code.git.commit,
            "lock": code.environment.uv_lock_hash,
        },
    }
    lines = list(map(dump, described))
    path.parent.mkdir(parents=True, exist_ok=True)
    file = File(path)
    file.write(chain([dump(header)], lines))
    file.close()
    return lines


def listing() -> Path:
    """The plan file that lists the specs whose create() runs in this context."""
    if (listed := creation().plan) is None:
        raise RuntimeError("only the create() of specs that a plan lists has a listing")
    return listed


def launch() -> str:
    """The id of the launch whose plan, or whose folder of found specs, holds ``listing()``."""
    plans = get_config().directories.anchored().plans
    return listing().relative_to(plans).parts[0].removesuffix(".plan")


# The file this process created in each launch's folder.
_own: dict[Path, File] = {}
os.register_at_fork(after_in_child=_own.clear)


def save_found(found: Sequence[Line]) -> Path:
    """Save the lines of specs that a running create() asked for and that are not stored.

    They go to this process's own file in the launch's folder, which the
    specs' runs then name as their plan.
    """
    folder = get_config().directories.anchored().plans / launch()
    if (file := _own.get(folder)) is None:
        file = _own[folder] = File.new(folder, ".plan")
    file.add([{"asked_by": sorted(creation().specs)}, *found])
    return file.path
