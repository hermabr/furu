"""The record of what ran: files in ``provenance/`` that outlive plans and results.

A launch writes ``provenance/launch-<id>`` once, before anything runs: where
it was started from, and what the lines of the specs it runs point to by
fingerprint. Each process that runs create() calls keeps one file of its own,
which begins with a line about the process and gets a line for each run just
before the run is published. A process whose create() asks for specs that no
launch listed does for them what a launch does, in its own file.

    {"launch": id, "git": {...}, "environment": {...}, "submitted": {...}}
    {"worker": {"host": ..., "cpus": ..., "gpus": [...], ...}}
    {"metadata": fingerprint, "is": {"environment": {...}, ...}}
    {"value": fingerprint, "is": value}
    {"run": name, "storage": root, "version": folder, "launch": id,
     "metadata": fingerprint, "start": at, "end": at, "specs": [line, ...],
     "loaded": [id, ...]}

A run's line holds the plan line of each of its specs: its own fields and the
ids it points to, never the specs upstream. Those are followed to their own
lines, in whichever file has them, and a fingerprint to a value line. So no
file here may be deleted. ``storage`` and ``version`` make the run's path
together with the ids. ``loaded`` lists the specs whose results the create()
loaded and that none of the run's lines point to.

The files are written as plan files are: compressed as they grow, on disk
after every addition, and added to only by the process that created them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import fields
from itertools import chain
from pathlib import Path

from furu import plan
from furu._tree import specs_in
from furu.config import get_config
from furu.core import Spec
from furu.dependencies import returned_attributes
from furu.execution.launch import IGNORE_STORED
from furu.plan import Line
from furu.protocol import plan_text
from furu.protocol.messages import ProcessSettings
from furu.provenance import SubmitProvenance, worker
from furu.spec_metadata import Metadata
from furu.utils import _hash_dict_deterministically


def _process(metadata: Metadata) -> Line:
    """What a Metadata says of the process that runs create(), as a line.

    Where the results go and which code counts are in each run's line.
    """
    how = ProcessSettings.from_metadata(metadata).model_dump(mode="json")
    return {"metadata": _hash_dict_deterministically(how), "is": how}


def _described(running: Sequence[Spec], shared: list[Line]) -> Iterator[Line]:
    """The plan lines of ``running`` and of what they reach.

    Once they are taken, ``shared`` holds what the record lines of ``running``
    point to by fingerprint: each distinct Metadata, what their dependency
    methods returned, and each value over 1 KB that one of them holds.
    """
    # Compared before any is fingerprinted: a launch has few distinct ones,
    # however many specs it runs.
    distinct = {
        (
            tuple(sorted(metadata.environment.items())),
            metadata.required_environment_variables,
            metadata.reuse,
        ): metadata
        for metadata in (obj._metadata for obj in running)
    }
    shared += map(_process, distinct.values())
    # The value lines that no spec that runs has pointed to yet.
    values: dict[str, Line] = {}
    for line in plan.lines(running, {obj.object_id for obj in running}):
        if "value" in line:
            values[line["value"]] = line
        elif line.get("runs"):
            for fingerprint in map(plan_text.pointer, line["fields"].values()):
                if fingerprint in values:
                    shared.append(values.pop(fingerprint))
        yield line


def launch(
    plan_path: Path,
    asked: Sequence[Spec],
    running: Sequence[Spec],
    facts: SubmitProvenance,
) -> list[str]:
    """Save a launch's plan and the file its workers share, before anything runs.

    Returns the plan's lines as written.
    """
    shared: list[Line] = []
    lines = plan.write(plan_path, asked, _described(running, shared), facts)
    header: Line = {"launch": plan_path.stem, **facts.model_dump(mode="json")}
    if ignore := os.environ.get(IGNORE_STORED):
        header["ignore_stored"] = ignore
    folder = get_config().directories.anchored().provenance
    folder.mkdir(parents=True, exist_ok=True)
    file = plan_text.File(folder / f"launch-{plan_path.stem}")
    file.add([header, *shared])
    file.close()
    return lines


# The file this process created in each provenance folder.
_own: dict[Path, plan_text.File] = {}
os.register_at_fork(after_in_child=_own.clear)


def own() -> plan_text.File:
    """This process's file in provenance/, which begins with a line about the process.

    It is created the first time the process starts an attempt, so a process
    that never does leaves no file.
    """
    folder = get_config().directories.anchored().provenance
    if (file := _own.get(folder)) is None:
        file = _own[folder] = plan_text.File.new(folder)
        file.add([{"worker": worker()}])
    return file


def found(running: Sequence[Spec]) -> tuple[Path, list[Line]]:
    """Save what describes specs that a running create() asked for and that are not stored.

    ``running`` are those specs and what is missing under them. What their
    lines point to goes to this process's record first and the lines to its
    plan file after that, so no found run can finish before its values are on
    disk. Returns the plan file and the lines.
    """
    shared: list[Line] = []
    lines = list(_described(running, shared))
    own().add(shared)
    return plan.save_found(lines), lines


def batch(
    group: Sequence[Spec],
    *,
    run: str,
    version: str,
    start: str,
    end: str,
    loaded: Sequence[Spec],
) -> None:
    """Add a finished run to this process's record, before the rename that publishes it.

    ``run`` is the name of its folder and ``version`` the code-version folder
    it is published into. ``loaded`` are the specs whose results its create()
    loaded.
    """
    lead = group[0]
    line: Line = {
        "run": run,
        "storage": str(lead._metadata.storage),
        "version": version,
        "launch": plan.launch(),
        "metadata": _process(lead._metadata)["metadata"],
        "start": start,
        "end": end,
        "specs": [spec for spec in plan.lines(group, reach=False) if "spec" in spec],
    }
    if loaded:
        beyond = dict.fromkeys(obj.object_id for obj in loaded)
        # What is in vars() first, which calls nothing: a dependency method
        # whose field create() never read runs only when a loaded spec is left over.
        read = (
            (obj, f.name) for obj in group for f in fields(obj) if f.name in vars(obj)
        )
        rest = ((obj, name) for obj in group for name in returned_attributes(type(obj)))
        for holder, name in chain(read, rest):
            if not beyond:
                break
            for held in specs_in(getattr(holder, name)):
                beyond.pop(held.object_id, None)
        if beyond:
            line["loaded"] = list(beyond)
    own().add([line])
