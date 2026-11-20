"""The steps a class declares in ``migrations``, and how they name its values.

Changing a class's fields gives every spec of the class a new id. A step says
that a change keeps the stored results: when a spec is named, the steps are
undone, newest first, so the spec has the id it had before the change. Nothing
stored is read, rewritten or moved.

A step with ``breaking=True`` is a fresh start instead: it and every step
listed before it are ignored, so the ids are new.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cache

from furu.utils import _hash_dict_deterministically, fully_qualified_name

_NO_DEFAULT = object()


@dataclass(frozen=True, slots=True)
class Added:
    """A field, or ``@furu.dependency`` method, added to a class that has results.

    ``default`` is the value those results behave as: the name is left out of
    the id while it has that value, so they are still found. Any other value
    is new work. Without one the step has to be ``breaking``.
    """

    name: str
    _: dataclasses.KW_ONLY
    default: object = _NO_DEFAULT
    breaking: bool = False

    def __post_init__(self) -> None:
        if self.breaking == (self.default is not _NO_DEFAULT):
            raise TypeError(
                f"Added({self.name!r}) needs exactly one of default= (the value "
                "the stored results behave as) or breaking=True (a fresh start)"
            )


@dataclass(frozen=True, slots=True)
class Renamed:
    """A renamed field or ``@furu.dependency`` method: the id keeps the old name."""

    old: str
    _: dataclasses.KW_ONLY
    to: str
    breaking: bool = False


@dataclass(frozen=True, slots=True)
class MovedFrom:
    """A renamed or moved class: ids and folders keep the old name, for new results too."""

    fully_qualified_name: str
    _: dataclasses.KW_ONLY
    breaking: bool = False


type MigrationStep = Added | Renamed | MovedFrom


class MigrationError(RuntimeError):
    pass


def describe(step: MigrationStep) -> str:
    match step:
        case Added(name=name, default=default):
            body = repr(name)
            if default is not _NO_DEFAULT:
                body += f", default={default!r}"
        case Renamed(old=old, to=to):
            body = f"{old!r}, to={to!r}"
        case MovedFrom(fully_qualified_name=name):
            body = repr(name)
    return f"{type(step).__name__}({body}{', breaking=True' if step.breaking else ''})"


def class_part_of(first_names: tuple[str, ...]) -> str:
    return _hash_dict_deterministically(first_names)


@dataclass(frozen=True, slots=True)
class Naming:
    """How a class's values are named: as the class was before any declared change."""

    # The names of the fields the class first had, which the class part is a
    # hash of. Whether each is a field or a @furu.dependency method is not part
    # of it.
    first_names: tuple[str, ...]
    class_part: str
    # Per field: its attribute, the name it is hashed under, and the step that
    # added it, if one did.
    fields: tuple[tuple[str, str, Added | None], ...]
    moved_from: str | None


@cache
def naming(cls: type) -> Naming:
    """Undo the steps in ``cls.migrations``, newest first.

    A breaking step is a fresh start: it and every step listed before it are
    ignored, so they can be deleted once their results are.
    """
    assert dataclasses.is_dataclass(cls)
    steps = getattr(cls, "migrations", ())

    # A field's name at this point of the undoing -> its attribute today.
    first = {field.name: field.name for field in dataclasses.fields(cls)}
    added: dict[str, tuple[str, Added]] = {}
    moved_from = None
    for index, step in reversed(tuple(enumerate(steps))):
        if step.breaking:
            break
        problem = None
        match step:
            case Added(name=name):
                if name not in first:
                    problem = f"{name!r} is not a field; fields: {sorted(first)}"
                else:
                    added[name] = (first.pop(name), step)
            case Renamed(old=old, to=to):
                if to not in first:
                    problem = f"{to!r} is not a field; fields: {sorted(first)}"
                elif old in first or old in added:
                    # Two fields hashed under one name would share ids.
                    problem = f"{old!r} already names a field"
                else:
                    first[old] = first.pop(to)
            case MovedFrom(fully_qualified_name=name):
                if steps is not vars(cls).get("migrations"):
                    # Two classes stored under one name would share ids.
                    problem = (
                        f"{cls.__name__} inherits this step and would be stored "
                        f"under the same name as the class that declares it; "
                        f"give {cls.__name__} its own migrations"
                    )
                else:
                    moved_from = name
        if problem:
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] ({describe(step)}): {problem}"
            )

    first_names = tuple(sorted(first))
    return Naming(
        first_names=first_names,
        class_part=class_part_of(first_names),
        fields=(
            *((attribute, name, None) for name, attribute in first.items()),
            *(
                (attribute, name, step)
                for name, (attribute, step) in reversed(added.items())
            ),
        ),
        moved_from=moved_from,
    )


@cache
def explained_names(cls: type) -> frozenset[tuple[str, ...]]:
    """The sets of first names that results of ``cls`` may be stored under.

    The one the source gives now, and the one from before each breaking step,
    which that fresh start left behind on purpose.
    """
    assert dataclasses.is_dataclass(cls)
    names = {field.name for field in dataclasses.fields(cls)}
    explained = set()
    for step in reversed(getattr(cls, "migrations", ())):
        if step.breaking:
            explained.add(tuple(sorted(names)))
        # Steps from before a fresh start are not required to fit the fields.
        match step:
            case Added(name=name):
                names.discard(name)
            case Renamed(old=old, to=to) if to in names:
                names.remove(to)
                names.add(old)
    return frozenset({*explained, tuple(sorted(names))})


@cache
def stored_name(cls: type) -> str:
    """The class name an id and a folder use: the first one, when the class has moved."""
    return naming(cls).moved_from or fully_qualified_name(cls)
