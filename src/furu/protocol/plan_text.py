"""The text of plan lines and plan files, which scripts, the manager and workers share.

A line holds a spec as it was hashed: an added field left out at its declared
default, old field names, the old class name. So nobody who builds a spec
from a line hashes it again: its id is in the line. The fields that were left
out are noted beside the hashed ones.

    {"spec": id, "fields": {name: value}, "left_out": {name: default}, "runs": true}
    {"value": fingerprint, "is": value}
    {"moved": {old class name: where the class lives now}}

A field value whose written form is over 1 KB is a line of its own, and each
spec that holds it points to that line by the value's fingerprint. What a
``@furu.dependency`` method returned is always a pointer, of a kind of its
own, with the fingerprint that went into the spec's id. The returned value has
a line only for a spec that runs, which says what the spec waits for; whoever
builds the spec from its line calls the method and compares.
"""

from __future__ import annotations

import json
import os
import re
import socket
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from furu.constants import KINDMARKER, VALUEMARKER
from furu.utils import JsonValue, _stable_json_dump

type Line = dict[str, Any]

dump = _stable_json_dump


def pointer(value: JsonValue) -> str | None:
    """The fingerprint of the value line that a written field value points to, if it does."""
    if isinstance(value, dict) and value.get(KINDMARKER) in ("value", "returned"):
        return cast(str, value[VALUEMARKER])
    return None


def key(line: Line) -> str:
    """What other lines call this one: a spec's id, a value's fingerprint."""
    return line.get("spec") or line.get("value") or "moved"


# How a dumped line names a spec or a value line. A string never matches:
# its quotes are escaped.
_HOLDS = re.compile(
    rf'\{{"{re.escape(KINDMARKER)}":"(spec|value|returned)",'
    rf'"{re.escape(VALUEMARKER)}":"([^"]*)"\}}'
)


def holds(text: str, *, returned: bool = True) -> list[str]:
    """The ids and fingerprints that a dumped line points to, in the order written.

    Without ``returned``, the pointers to what dependency methods returned
    are left out: whoever builds the spec gets that by calling the method.
    """
    return [
        named for kind, named in _HOLDS.findall(text) if returned or kind != "returned"
    ]


class File:
    """A file of lines in one compressed stream, on disk after every addition.

    Only the process that created the file adds to it, and a reader takes the
    lines up to where the stream stops. ``zcat`` prints them.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("xb")
        self._pack = zlib.compressobj(wbits=31)

    @classmethod
    def new(cls, folder: Path, suffix: str = "") -> File:
        """Create this process's file in ``folder``.

        It is named for the process's Slurm job, or for its machine and
        process id. One name can stand for several processes, one after
        another, and a process never adds to a file it did not create: when
        the name is taken it adds -2, then -3.
        """
        folder.mkdir(parents=True, exist_ok=True)
        name = (
            f"job-{job}"
            if (job := os.environ.get("SLURM_JOB_ID"))
            else f"{socket.gethostname()}-pid-{os.getpid()}"
        )
        taken = 0
        while True:
            try:
                return cls(folder / f"{name}{f'-{taken + 1}' if taken else ''}{suffix}")
            except FileExistsError:
                taken += 1

    def add(self, lines: Iterable[Line]) -> None:
        self.write(map(dump, lines))

    def write(self, texts: Iterable[str]) -> None:
        """Add lines that are dumped already."""
        for text in texts:
            self._file.write(self._pack.compress(f"{text}\n".encode()))
        self._file.write(self._pack.flush(zlib.Z_SYNC_FLUSH))
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.write(self._pack.flush())
        self._file.close()


def read(path: Path) -> list[Line]:
    """The whole lines of a file, also of one that is still being written."""
    text = zlib.decompressobj(wbits=31).decompress(path.read_bytes())
    lines = []
    for line in text.splitlines():
        try:
            lines.append(json.loads(line))
        except ValueError:  # cut short when its writer was killed
            continue
    return lines
