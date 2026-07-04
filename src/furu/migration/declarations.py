"""Validate migration declarations at class creation time."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from furu.migration.generations import _ANY, _WalkError, _walk_generations
from furu.migration.steps import (
    Added,
    MovedFrom,
    Renamed,
    Retyped,
    Rewrite,
    _describe_step,
)

if TYPE_CHECKING:
    from furu.core import Spec


def validate_migration_declaration(cls: type[Spec[Any]]) -> None:
    steps = cls.migrations
    if not isinstance(steps, tuple) or not all(
        isinstance(step, (Renamed, Added, MovedFrom, Retyped, Rewrite))
        for step in steps
    ):
        raise TypeError(
            f"{cls.__name__}.migrations must be a tuple of "
            "Renamed/Added/MovedFrom/Retyped/Rewrite steps"
        )

    fields_by_name = {field.name: field for field in dataclasses.fields(cls)}
    try:
        _, added_current_name = _walk_generations(
            owner=cls.__name__,
            steps=steps,
            class_name="",
            current_fields=dict.fromkeys(fields_by_name, _ANY),
            shape_for=lambda step: _ANY,
        )
    except _WalkError as error:
        raise TypeError(str(error)) from None

    for index, current_name in added_current_name.items():
        field = fields_by_name[current_name]
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ):
            raise TypeError(
                f"{cls.__name__}.migrations[{index}] "
                f"({_describe_step(steps[index])}): field {current_name!r} has no "
                "default; Added can only backfill a field with a default value"
            )
