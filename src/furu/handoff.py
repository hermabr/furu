"""Handing a pipeline to someone: an export of the record, and a rebuild from it.

An export starts from the specs you name and follows their lines in
``provenance/`` all the way down: to the specs they hold, to the values written
apart from them and what their dependency methods returned, and to the specs
whose results their create() loaded. Each line it reaches goes into one file,
written as a plan is, which names the commit it was made at. Only a spec that
finished has a line, so a half-finished pipeline is handed over as the commit
and the launch command.

Whoever gets the file checks out that commit and rebuilds: furu recomputes
every id from its line, stops when one differs, and builds every spec in the
file. A result does not change unless a field does, so the one commit stands
for however many it took to build the pipeline. What comes out is the same
computation, not the same bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from furu import plan
from furu._tree import specs_in
from furu.config import get_config
from furu.core import Spec
from furu.execution.load_or_create import build
from furu.logging import get_logger
from furu.migration.steps import moved_classes
from furu.plan import Line
from furu.provenance import capture_submit_provenance
from furu.utils import _fingerprint, _hash_dict_deterministically, fully_qualified_name

if TYPE_CHECKING:
    from furu.worker.backends.protocol import WorkerBackend


def _recorded(start: Sequence[str]) -> list[Line]:
    """The lines in provenance/ of the specs ``start`` names and of all they lead to, each once.

    The record has no index, so every file in it is read. A spec that ran more
    than once is described by the run that ended last.
    """
    folder = get_config().directories.anchored().provenance
    # Spec lines by id and value lines by fingerprint, and the run of each spec.
    lines: dict[str, Line] = {}
    runs: dict[str, Line] = {}
    for file in folder.glob("[!.]*"):
        for line in plan.read(file):
            if "value" in line:
                lines[line["value"]] = line
            for spec in line.get("specs", ()):
                name = spec["spec"]
                if name not in runs or runs[name]["end"] < line["end"]:
                    lines[name], runs[name] = spec, line
    reached: dict[str, Line] = {}
    todo = list(start)[::-1]
    while todo:
        if (name := todo.pop()) in reached:
            continue
        if name not in lines:
            raise RuntimeError(
                f"{name} has no line under {folder}. The record holds the "
                "specs that finished and what their lines point to; hand a "
                "half-finished pipeline over as the commit and the launch "
                "command."
            )
        reached[name] = lines[name]
        todo += plan.holds(plan.dump(lines[name]))
        todo += runs.get(name, {}).get("loaded", ())
    return list(reached.values())


def export(tree: object, /, path: Path) -> None:
    """Write what the specs in ``tree`` were built from, all the way down, into one file.

    The file names this checkout's commit, which `rebuild` is to run at, so
    the checkout has to be committed and pushed.
    """
    asked = specs_in(tree)
    code = capture_submit_provenance()
    lines = _recorded([obj.object_id for obj in asked])
    # A class that a line names by an old name is found through this table,
    # which covers the classes this process has named.
    if moved := {
        old: fully_qualified_name(cls)
        for old, cls in moved_classes.items()
        if any(
            f'"{old}:' in text or f'"{old}"' in text for text in map(plan.dump, lines)
        )
    }:
        lines.append({"moved": moved})
    plan.write(path, asked, lines, code)
    get_logger().info(
        "exported %d specs, to rebuild at commit %s",
        sum("spec" in line for line in lines),
        code.git.commit,
        extra={"path": path},
    )


def rebuild(path: Path, *, on: Sequence[WorkerBackend] | None = None) -> None:
    """Build every spec in an export, after recomputing each one's id.

    An id is recomputed from the line's fields, the values the line points to
    and the names this checkout's class gives it. One that differs from what
    the file says stops the rebuild before anything runs.
    """
    header, *lines = plan.read(path)
    values = {line["value"]: line["is"] for line in lines if "value" in line}
    specs = plan.Specs(lines)
    described: list[Spec] = []
    for line in lines:
        if (spec_id := line.get("spec")) is None:
            continue
        try:
            obj = specs[spec_id]
            fingerprints = {
                name: _fingerprint(
                    plan.dump(
                        value if (to := plan.pointer(value)) is None else values[to]
                    )
                )
                for name, value in line["fields"].items()
            }
        except Exception as exc:
            exc.add_note(f"while reading {spec_id} from {path}")
            raise
        found = ":".join(
            [
                obj._stored_name,
                obj._class_part,
                _hash_dict_deterministically(fingerprints),
            ]
        )
        if found != spec_id:
            raise ValueError(
                f"{path} holds a line for {spec_id}, which gives the id "
                f"{found}. Either the file was changed, or this checkout names "
                f"the spec differently than commit {header['code']['commit']}, "
                "where it was exported."
            )
        described.append(obj)
    build(described, on=on)
