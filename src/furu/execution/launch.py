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
from collections.abc import Collection, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from pathlib import Path

from pydantic import BaseModel

from furu._batched import is_batched
from furu.core import Spec
from furu.logging import get_logger
from furu.migration import schema
from furu.migration.steps import (
    MigrationError,
    class_part_of,
    explained_names,
    naming,
    stored_name,
)
from furu.storage.stored import Stored

logger = get_logger("run")

IGNORE_STORED = "FURU_IGNORE_STORED"

# The stored names of the classes the launch asked about, where its create()s run.
_asked: ContextVar[frozenset[str]] = ContextVar(
    "furu_launch_asked", default=frozenset()
)


def _stage(cls: type) -> str:
    """A class as its ids and folders name it, without the module."""
    return stored_name(cls).rpartition(".")[2]


def stage_table(missing: Iterable[Spec]) -> str:
    """One row per class that has specs missing, in the order the launch reached them."""
    counts = Counter(map(type, missing))
    rows = [("stage", "missing")] + [
        (_stage(cls), str(count)) for cls, count in counts.items()
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
        if not is_batched(type(obj)) and (row := look.listed(obj)) is not None:
            held[type(obj)][row.run] = None
    if held:
        lines = [
            f"  {_stage(cls)}: {', '.join(map(str, [*runs][:3]))}"
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
        skip = _asked.get()
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
        try:
            folders = set(os.listdir(class_dir.parent))
        except FileNotFoundError:
            folders = set()
        for folder in folders - explained:
            path = class_dir.parent / folder
            unexplained[f"{stored_name(cls)}:{folder}"].append(
                f"{_stage(cls)}: {path} was {_stored(cls, schema.stored_names(path))}"
                f", and this launch stores in {class_dir}"
            )
        held_here = holds.get(class_dir, {})
        recorded = schema.nested_names(class_dir) if held_here else {}
        for nested, at in held_here.items():
            seen = recorded.get(stored_name(nested), set())
            for names in seen - explained_names(nested):
                unexplained[f"{stored_name(cls)}:{class_dir.name}"].append(
                    f"{_stage(cls)}: {nested.__name__} at {'.'.join(at)} was "
                    f"{_stored(nested, names)} in {class_dir}, where this launch "
                    "stores too"
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


@contextmanager
def asked(stages: frozenset[str]) -> Generator[None]:
    """Run create()s of a launch that asked about ``stages``."""
    token = _asked.set(stages)
    try:
        yield
    finally:
        _asked.reset(token)
