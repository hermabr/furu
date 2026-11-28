"""Where a spec's result is stored, and reading it back.

A spec without ``@furu.batched`` is a run with one row, named by its hash, so
it is found by name: one file check per live code version. A batched spec is a
row in some run of its class, so it is found by listing the class's runs/
folders and reading each run's events.jsonl, whose first line lists the specs
the run holds. No index is kept.

With an overlay a class has two folders, and each is looked in as above: the
overlay's first, then the real store's.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from furu._batched import is_batched
from furu.code_trace import versions
from furu.config import CodeVersion, layers
from furu.dependencies import record_loaded
from furu.migration.schema import check_defaults
from furu.result.bundle import load_result_bundle
from furu.storage._layout import (
    BATCH,
    FIXED,
    data_dir_in,
    events_path_in,
    list_names,
    result_dir_in,
    run_dir_in,
    runs_dir_in,
    version_dirs_in,
)
from furu.storage.events import parse_events

if TYPE_CHECKING:
    from furu.core import Spec


class Row(NamedTuple):
    """A stored result: a published run and the spec's row in it."""

    run: Path
    index: int


# A spec's row in the run that was published first: when the run's attempt
# ended, its name, the row and the run folder. They compare in that order, so
# every reader picks the same run when two hold the spec.
type _First = tuple[str, str, int, Path]

# What this process has read under each code version: the runs, and the first
# row of each spec in them. A published run never changes, so it is read once.
_READ: dict[Path, tuple[set[str], dict[str, _First]]] = {}
_READERS = 16


def _events(runs: Sequence[Path]) -> list[bytes]:
    """The events.jsonl of each run; empty for one deleted after it was listed."""
    texts = []
    for run in runs:
        try:
            # Unbuffered: each system call hands the interpreter to another
            # thread, and this makes half as many.
            with open(events_path_in(run), "rb", buffering=0) as file:
                texts.append(file.read())
        except FileNotFoundError:
            texts.append(b"")
    return texts


def look_again(class_dir: Path) -> None:
    """Make this machine forget which runs of a class it found missing.

    It remembers for up to a minute that a name was not there, until the
    folder above the name is opened or listed. Opening does not read the
    names, of which runs/ can hold very many.
    """
    for layer in layers(class_dir):
        for version in version_dirs_in(layer):
            for folder in (version, runs_dir_in(version)):
                with suppress(OSError):
                    os.close(os.open(folder, os.O_RDONLY | os.O_DIRECTORY))


def _listed(version: Path, batches: bool) -> dict[str, _First]:
    """The first row of each spec in a code version's runs, as a listing shows now.

    With ``batches``, only the batch runs are read.
    """
    runs = runs_dir_in(version)
    names = set(list_names(runs))
    read, first = _READ.setdefault(version, (set(), {}))
    if read - names:
        # A run was deleted, and a spec it held may be in another one.
        read.clear()
        first.clear()
    todo = {name for name in names - read if not batches or name.startswith(BATCH)}
    if new := [runs / name for name in sorted(todo)]:
        # The threads only wait for the files: each takes its share of them,
        # and the lines are parsed once all are in, since threads that parse
        # hold each other up.
        shares = [new[start::_READERS] for start in range(min(_READERS, len(new)))]
        with ThreadPoolExecutor(_READERS) as pool:
            texts = list(pool.map(_events, shares))
        for run, text in zip(chain(*shares), chain(*texts)):
            events = parse_events(text)
            ended = [event["at"] for event in events if event.get("outcome") == "ok"]
            if not ended:
                continue
            name = run.name
            read.add(name)
            for index, spec in enumerate(events[0]["specs"]):
                row = (ended[-1], name, index, run)
                if first.setdefault(spec, row) > row:
                    first[spec] = row
    return first


class Stored:
    """One look at what is stored.

    A look lists each class folder and each runs/ folder once, however many
    specs it is asked about, so asking about a whole launch costs one listing
    per class. Take a new look to see what was published since.
    """

    def __init__(self) -> None:
        self._live: dict[tuple[Path, str], list[Path]] = {}
        self._listed: dict[tuple[Path, bool], dict[str, _First]] = {}

    def _versions(self, class_dir: Path, code_version: CodeVersion) -> list[Path]:
        """The code versions in a class folder whose runs count: the live ones."""
        if (class_dir, code_version) not in self._live:
            self._live[class_dir, code_version] = (
                [class_dir / FIXED]
                if code_version == "fixed"
                else [version for version, why in versions(class_dir) if why is None]
            )
        return self._live[class_dir, code_version]

    def row(self, obj: Spec) -> Row | None:
        """Where obj's result is, if one is stored and its code still matches."""
        # Named first: that is what finds the dataclasses the spec holds, whose
        # defaults are checked too. The script that wrote a spec's line checked.
        name = obj._artifact_hash
        if not obj._from_line:
            check_defaults(obj._class_dir, type(obj))
        if is_batched(type(obj)):
            return self.listed(obj)
        for class_dir in layers(obj._class_dir):
            held = self.named(class_dir, obj._metadata.code_version, name)
            if len(held) > 1:  # the one published first, as for a batch
                return self.listed(obj)
            if held:
                return Row(held[0], 0)
        return None

    def named(
        self, class_dir: Path, code_version: CodeVersion, name: str
    ) -> list[Path]:
        """The runs called ``name`` in one class folder, whose code still matches."""
        return [
            run
            for version in self._versions(class_dir, code_version)
            if (run := run_dir_in(version, name)).is_dir()
        ]

    def listed(self, obj: Spec, *, batches: bool = False) -> Row | None:
        """obj's row in the run published first, among every run of its class.

        With ``batches``, among the batch runs only: one listing, and a read of
        each batch run this process has not read yet.
        """
        for class_dir in layers(obj._class_dir):
            held = []
            for version in self._versions(class_dir, obj._metadata.code_version):
                if (version, batches) not in self._listed:
                    self._listed[version, batches] = _listed(version, batches)
                if (
                    first := self._listed[version, batches].get(obj._artifact_hash)
                ) is not None:
                    held.append(first)
            if held:
                _, _, index, run = min(held)
                return Row(run, index)
        return None


def stored_row(obj: Spec) -> Row | None:
    """Where obj's result is, as a look taken now sees it."""
    return Stored().row(obj)


def why_rerun(class_dir: Path, run: str) -> str | None:
    """Why the newest run of this name under traced code no longer counts, if any."""
    outdated = [
        (folder.stat().st_mtime_ns, why)
        for layer in layers(class_dir)
        for version, why in versions(layer)
        if why is not None and (folder := run_dir_in(version, run)).is_dir()
    ]
    return max(outdated)[1] if outdated else None


def load_rows(objs: Sequence[Spec], rows: Sequence[Row]) -> list[object]:
    """Each spec's result from its row, reading every run's files once.

    Rows of one run are read together, so a codec is handed the rows that
    share a file in one call. Each run counts as loaded by the create() that
    is running.
    """
    by_run: dict[Path, dict[int, Spec]] = {}
    for obj, row in zip(objs, rows, strict=True):
        by_run.setdefault(row.run, {})[row.index] = obj
    values: dict[Row, object] = {}
    for run, held in by_run.items():
        record_loaded(run, held.values())
        loaded = load_result_bundle(
            result_dir_in(run),
            list(held),
            data_dir=data_dir_in(run),
            spec_classes=[type(obj) for obj in held.values()],
        )
        values.update(zip((Row(run, index) for index in held), loaded, strict=True))
    return [values[row] for row in rows]
