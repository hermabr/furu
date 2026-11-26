"""What a launch says and asks before anything runs.

A launch is one ``create()`` that finds specs missing, in this process or on
workers. It lists them by stage, and it asks before it leaves stored results
behind: a class with missing specs may have results in a folder the launch
will not look in, because the class changed and no step in ``migrations``
says how. furu cannot tell a step you forgot from results another branch
stored, so the launch stops and names both ways on. ``FURU_IGNORE_STORED``
names the folders one launch may leave behind. Asking stores nothing.

A class that lost ``@furu.batched`` stops the launch too, when a missing spec
of it has a result in a batch folder: without the decorator a spec is looked
for by its own folder name only.

Nothing checks afterwards, with one exception: a spec that a ``create()``
builds is checked where it is built, together with what is missing under it,
for each class the launch never saw.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path

from pydantic import BaseModel

from furu._batched import is_batched
from furu._tree import specs_in
from furu.config import get_config
from furu.core import Spec
from furu.dag import Dag
from furu.dependencies import creation
from furu.logging import get_logger
from furu.migration import schema
from furu.migration.steps import (
    MigrationError,
    class_part_of,
    explained_names,
    naming,
    stored_name,
)
from furu.serializer.artifact import remembered
from furu.storage._layout import REMEMBERED
from furu.storage.stored import Stored

logger = get_logger("run")

IGNORE_STORED = "FURU_IGNORE_STORED"


def missing(objs: Sequence[Spec], dag: Dag | None = None) -> list[Spec]:
    """Add to ``dag`` the specs under ``objs`` that are not stored, and return them.

    The walk goes down through the fields and stops at a stored spec: nothing
    is looked at below one. Each new spec waits for the ones it holds that
    are not stored either, among them the ones ``dag`` had already. A large
    value that the launch has named is walked once, however many specs hold
    it: they wait for the value, and the value for the specs in it.
    """
    dag = Dag() if dag is None else dag
    look = Stored()  # one listing per class, however many specs it has
    new: list[Spec] = []
    seen: set[str] = set()
    holds: dict[str, list[str]] = {}
    values: set[str] = set()
    pending = list(reversed(objs))
    while pending:
        obj = pending.pop()
        if (spec_id := obj.object_id) in seen:
            continue
        seen.add(spec_id)
        if spec_id in dag.nodes:
            holds[spec_id] = []
        elif look.row(obj) is None:
            new.append(obj)
            held = holds[spec_id] = []
            for field in fields(obj):
                value = getattr(obj, field.name)
                fingerprint = remembered(value, type(obj), field.name)
                if fingerprint is None:
                    inside = specs_in(value)
                    held += [ref.object_id for ref in inside]
                else:
                    held.append(fingerprint)
                    if fingerprint in holds:
                        continue
                    values.add(fingerprint)
                    inside = specs_in(value) if fingerprint not in dag.nodes else []
                    holds[fingerprint] = [ref.object_id for ref in inside]
                pending.extend(reversed(inside))
    dag.add(holds, values)
    return new


def stage_table(missing: Iterable[Spec]) -> str:
    """One row per class that has specs missing, in the order the launch reached them."""
    counts = Counter(map(type, missing))
    rows = [("stage", "missing")] + [
        (stored_name(cls), str(count)) for cls, count in counts.items()
    ]
    stage_width, count_width = (max(map(len, column)) for column in zip(*rows))
    return "\n".join(
        f"{stage:<{stage_width}}   {count:>{count_width}}" for stage, count in rows
    )


def _nested_in(
    value: object, at: tuple[str, ...]
) -> Iterator[tuple[type, tuple[str, ...]]]:
    """Each dataclass inside ``value`` and the fields that lead to it.

    A spec is not entered: its class folder records what it holds.
    """
    match value:
        case Spec():
            return
        case dict():
            children = [(at, child) for child in value.values()]
        case list() | tuple() | set() | frozenset():
            children = [(at, child) for child in value]
        case BaseModel():
            names = type(value).model_fields
            children = [((*at, name), getattr(value, name)) for name in names]
        case _ if is_dataclass(value) and not isinstance(value, type):
            yield type(value), at
            children = [
                ((*at, field.name), getattr(value, field.name))
                for field in fields(value)
            ]
        case _:
            return
    for child_at, child in children:
        yield from _nested_in(child, child_at)


def held_in(obj: Spec) -> dict[type, tuple[str, ...]]:
    """Each dataclass a named spec holds in its values, and the fields that lead to it."""
    holds: dict[type, tuple[str, ...]] = {}
    if held := schema.held[type(obj)]:
        for field in fields(obj):
            for nested, at in _nested_in(getattr(obj, field.name), (field.name,)):
                # Naming the spec left out a dataclass that a serializer
                # writes: its field names are not in the id.
                if nested in held:
                    holds.setdefault(nested, at)
    return holds


def _stored(cls: type, names: Collection[str] | None) -> str:
    """How the names results were stored under differ from the ones ``cls`` gives now."""
    if names is None:
        return "stored under field names that no schema.ndjson records"
    now = set(naming(cls).first_names)
    differs = [
        f"{word} the field{'' if len(only) == 1 else 's'} "
        + ", ".join(map(repr, sorted(only)))
        for word, only in (("without", now - set(names)), ("with", set(names) - now))
        if only
    ]
    return "stored " + (" and ".join(differs) or "under the same field names")


def _check_batch_folders(missing: Sequence[Spec]) -> None:
    """Stop when a missing spec without ``@furu.batched`` is a row in a batch folder."""
    look = Stored()
    held: defaultdict[type, dict[Path, None]] = defaultdict(dict)
    for obj in missing:
        if not is_batched(type(obj)) and (row := look.listed(obj, batches=True)):
            held[type(obj)][row.run] = None
    if held:
        lines = [
            f"  {stored_name(cls)}: {', '.join(map(str, [*runs][:3]))}"
            + (f" and {len(runs) - 3} more" if len(runs) > 3 else "")
            for cls, runs in held.items()
        ]
        raise RuntimeError(
            "this launch would compute specs again that have results in batch "
            "folders, stored when their class had @furu.batched\n"
            + "\n".join(lines)
            + "\nA spec without @furu.batched is looked for by its own folder "
            "name. Put the decorator back on create(), or delete the folders."
        )


def ask(missing: Sequence[Spec], *, found: bool = False) -> frozenset[str]:
    """Stop the launch when it would leave stored results behind.

    For each class with missing specs, a folder under the class's name is
    explained when it is the class part the source gives now, or the one from
    before a breaking step. The field names a held dataclass was stored with
    are explained the same way. Anything else raises, unless
    ``FURU_IGNORE_STORED`` names the folder. Returns the stored names of the
    classes it asked about.

    With ``found``, a create() asked for ``missing``. The classes the launch
    asked about are skipped. It could not ask about the others, so there is no
    launch to go on with ``FURU_IGNORE_STORED``: unexplained results fail the
    create().
    """
    ignore: set[str] = set()
    if found:
        skip = creation().asked
        missing = [obj for obj in missing if obj._stored_name not in skip]
    else:
        ignore = {name.strip() for name in os.environ.get(IGNORE_STORED, "").split(",")}
    class_dirs: dict[Path, type[Spec]] = {}
    holds: dict[Path, dict[type, tuple[str, ...]]] = {}
    for obj in missing:
        class_dir = obj._class_dir
        class_dirs.setdefault(class_dir, type(obj))
        for nested, at in held_in(obj).items():
            holds.setdefault(class_dir, {}).setdefault(nested, at)

    # What the source does not explain, under each name the variable can take.
    unexplained: defaultdict[str, list[str]] = defaultdict(list)
    for class_dir, cls in class_dirs.items():
        explained = set(map(class_part_of, explained_names(cls)))
        for parent in get_config().layers(class_dir.parent):
            try:
                folders = set(os.listdir(parent))
            except FileNotFoundError:
                folders = set()
            for folder in folders - explained:
                path = parent / folder
                if os.listdir(path) == [REMEMBERED]:
                    continue  # what dependency methods returned, and no result
                unexplained[f"{stored_name(cls)}:{folder}"].append(
                    f"{stored_name(cls)}: {path} was "
                    f"{_stored(cls, schema.stored_names(path))}"
                    f", and this launch stores in {class_dir}"
                )
        held_here = holds.get(class_dir, {})
        for layer in get_config().layers(class_dir) if held_here else ():
            recorded = schema.nested_names(layer)
            for nested, at in held_here.items():
                seen = recorded.get(stored_name(nested), set())
                for names in seen - explained_names(nested):
                    unexplained[f"{stored_name(cls)}:{class_dir.name}"].append(
                        f"{stored_name(cls)}: {nested.__name__} at {'.'.join(at)} was "
                        f"{_stored(nested, names)} in {layer}, where this launch "
                        "stores too"
                    )

    for name in sorted(ignore - unexplained.keys() - {""}):
        logger.warning(
            "%s names %s, which is no folder this launch would leave behind",
            IGNORE_STORED,
            name,
        )
    if asking := sorted(unexplained.keys() - ignore):
        lines = "".join(
            f"  {line}\n" for name in asking for line in sorted(unexplained[name])
        )
        if found:
            # A worker's failure shows the last line, so that one names the
            # class and the folder.
            raise MigrationError(
                "a create() built a spec of a class that has stored results "
                "the source does not explain. Declare the change with a step "
                "in the class's migrations, or merge the branch that stored "
                f"them. {IGNORE_STORED} does not cover a spec that a create() "
                f"builds.\n{lines.rstrip()}"
            )
        table = "".join(f"    {row}\n" for row in stage_table(missing).splitlines())
        value = ",".join(sorted(unexplained.keys() & ignore) + asking)
        raise MigrationError(
            f"this launch would leave stored results behind\n{table}{lines}"
            "A change of yours takes a step in the class's migrations: "
            "Added(name, default=...) or Renamed(old, to=new) keeps the stored "
            "results, and Added(name, breaking=True) is a fresh start. Results "
            "that another branch stored, or a field you deleted, take "
            f"{IGNORE_STORED}={value} for this one launch."
        )
    for name in sorted(unexplained):
        for line in sorted(unexplained[name]):
            logger.info("ignoring stored results · %s", line)
    _check_batch_folders(missing)
    return frozenset(map(stored_name, class_dirs.values()))
