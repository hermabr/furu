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

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from furu import plan
from furu._tree import specs_in
from furu.config import get_config
from furu.core import Spec
from furu.dependencies import returned_attributes
from furu.execution.load_or_create import build
from furu.logging import get_logger
from furu.migration.schema import held
from furu.migration.steps import naming
from furu.plan import Line
from furu.provenance import capture_submit_provenance
from furu.serializer.artifact import field_fingerprints
from furu.utils import _hash_dict_deterministically

if TYPE_CHECKING:
    from furu.worker.backends.protocol import WorkerBackend


def _recorded(start: Sequence[str]) -> list[Line]:
    """The lines in provenance/ of the specs ``start`` names and of all they lead to, each once.

    The record has no index, so every file in it is read. A spec that ran more
    than once is described by the run that ended last. The tables of moved
    classes are merged into one, which comes last.
    """
    folder = get_config().directories.anchored().provenance
    # Spec lines by id and value lines by fingerprint, and the run of each spec.
    lines: dict[str, Line] = {}
    runs: dict[str, Line] = {}
    moved: dict[str, str] = {}
    for file in folder.glob("[!.]*"):
        for line in plan.read(file):
            moved |= line.get("moved", {})
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
    return [*reached.values(), *([{"moved": moved}] if moved else ())]


def export(tree: object, /, path: Path) -> None:
    """Write what the specs in ``tree`` were built from, all the way down, into one file.

    The file names this checkout's commit, which `rebuild` is to run at, so
    the checkout has to be committed and pushed.
    """
    asked = specs_in(tree)
    code = capture_submit_provenance()
    lines = _recorded([obj.object_id for obj in asked])
    plan.write(path, asked, lines, code)
    get_logger().info(
        "exported %d specs, to rebuild at commit %s",
        sum("spec" in line for line in lines),
        code.git.commit,
        extra={"path": path},
    )


def rebuild(path: Path, *, on: Sequence[WorkerBackend] | None = None) -> None:
    """Build every spec in an export, after recomputing each one's id.

    An id is recomputed from the spec that this checkout builds from the line,
    as any spec is named. One that differs from what the file says stops the
    rebuild before anything runs.
    """
    # Read strictly: an export is never still being written, so one that was
    # cut short is refused.
    header, *lines = map(json.loads, gzip.decompress(path.read_bytes()).splitlines())
    specs = plan.Specs(lines)
    described: list[Spec] = []
    for line in lines:
        if (spec_id := line.get("spec")) is None:
            continue
        try:
            obj = specs[spec_id]
            cls = type(obj)
            returned = returned_attributes(cls)
            fingerprints = field_fingerprints(
                obj, obj.artifact_serializers, held[cls], leave=returned
            )
        except Exception as exc:
            exc.add_note(f"while reading {spec_id} from {path}")
            raise
        # What a dependency method returned is taken from the line: calling it
        # may load results that are not built yet, and its first read compares.
        for attribute, name, _ in naming(cls).fields:
            if attribute in returned and name in line["fields"]:
                fingerprints[name] = obj._returned[attribute]
        fields = _hash_dict_deterministically(fingerprints)
        found = f"{obj._stored_name}:{obj._class_part}:{fields}"
        if found != spec_id:
            raise ValueError(
                f"{path} holds a line for {spec_id}, which gives the id "
                f"{found}. Either the file was changed, or this checkout names "
                f"the spec differently than commit {header['code']['commit']}, "
                "where it was exported."
            )
        described.append(obj)
    build(described, on=on)
