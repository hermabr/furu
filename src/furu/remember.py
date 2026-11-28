"""What ``@furu.dependency(remember=True)`` methods returned, kept between launches.

A spec's id needs a fingerprint of what each of its dependency methods
returned, so naming a spec calls them. A method with ``remember=True`` is
called only when something it depends on has changed. Its call is traced as a
create() is, and a line says what it returned:

    <class folder>/remembered/launch-<id>.jsonl
        {"method": name, "fields": fingerprint, "returned": fingerprint, "trace": name}
    <class folder>/remembered/traces/<name>.json
        the functions the method ran and the code versions of the results it loaded

``fields`` is a fingerprint of the spec's other fields. A line holds while the
code its trace names is unchanged and every code version the trace names is
live. What the method returned is not saved, only its fingerprint. A launch
writes one file per class folder once it has named its specs, also when it
has nothing to run, and no file changes afterwards. Deleting the folder is
always safe: the next launch calls the methods again.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path

from furu.code_trace import Trace, why_trace_outdated
from furu.config import layers
from furu.storage._layout import remembered_dir_in
from furu.utils import _stable_json_dump, atomic_write_text

# The method's name and the fingerprint of the spec's other fields.
type _Key = tuple[str, str]

# The fingerprint of what each method returned, by class folder: from the
# lines that held when this process first read the folder, the newest under
# each key, and from what it has added since.
_LINES: dict[Path, dict[_Key, str]] = {}
# What this process has not saved yet, by class folder: the traces of the
# calls it made, by name, and their lines.
_UNSAVED: dict[Path, tuple[dict[str, Trace], list[str]]] = {}


def _trace_path(class_dir: Path, name: str) -> Path:
    return remembered_dir_in(class_dir) / "traces" / f"{name}.json"


def lookup(class_dir: Path, method: str, fields: str, storage_root: Path) -> str | None:
    """The fingerprint of what ``method`` returned for a spec with these fields.

    None when no line says that still holds: one whose method ran code that
    has changed since, or loaded a result that is out of date, does not.
    """
    if (lines := _LINES.get(class_dir)) is None:
        lines = _LINES[class_dir] = {}
        texts = []
        # Under an overlay the store's lines first: the overlay's are newer.
        for layer in reversed(layers(class_dir)):
            folder = remembered_dir_in(layer)
            try:
                files = sorted(folder.glob("launch-*.jsonl"), key=os.path.getmtime)
                texts += [file.read_text(encoding="utf-8") for file in files]
            except FileNotFoundError:  # deleted meanwhile, which is always allowed
                pass
        holds: dict[str, bool] = {}
        for text in texts:
            for line in map(json.loads, text.splitlines()):
                if (trace := line["trace"]) not in holds:
                    path = _trace_path(class_dir, trace)
                    holds[trace] = why_trace_outdated(path, storage_root) is None
                if holds[trace]:
                    lines[line["method"], line["fields"]] = line["returned"]
    return lines.get((method, fields))


def add(class_dir: Path, method: str, fields: str, returned: str, trace: Trace) -> None:
    """Remember what a call of ``method`` returned, once `lookup` found nothing.

    The next launch in this process saves it.
    """
    _LINES[class_dir][method, fields] = returned
    traces, lines = _UNSAVED.setdefault(class_dir, ({}, []))
    name = trace.version
    traces[name] = trace
    lines.append(
        _stable_json_dump(
            {"method": method, "fields": fields, "returned": returned, "trace": name}
        )
    )


def save(launch: str) -> None:
    """Write what this process has not saved: one file of lines per class folder, named for the launch.

    The traces go first, so that no line is read before its trace is there.
    """
    while _UNSAVED:
        class_dir, (traces, lines) = _UNSAVED.popitem()
        folder = remembered_dir_in(class_dir)
        # Storage that cannot be written to remembers nothing, and still loads.
        with suppress(OSError):
            (folder / "traces").mkdir(parents=True, exist_ok=True)
            for name, trace in traces.items():
                if not (path := _trace_path(class_dir, name)).exists():
                    atomic_write_text(
                        path, trace.model_dump_json(indent=2, exclude_none=True)
                    )
            atomic_write_text(
                folder / f"launch-{launch}.jsonl",
                "".join(f"{line}\n" for line in lines),
            )
