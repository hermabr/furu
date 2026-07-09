from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, assert_never, cast

from furu.constants import FIELDSMARKER, KINDMARKER
from furu.migration.steps import (
    MigrationError,
    MigrationStep,
    ResultAdded,
    ResultMigration,
    ResultRenamed,
    ResultRewrite,
    _describe_result_migration,
    _describe_step,
)
from furu.result.bundle import WRAPPER_KEY
from furu.utils import JsonFields, JsonValue


def _validate_json_value(value: object, *, description: str) -> JsonValue:
    def is_json_value(node: object) -> bool:
        if node is None or isinstance(node, bool | int | float | str):
            return True
        if isinstance(node, list):
            return all(is_json_value(item) for item in node)
        if isinstance(node, dict):
            return all(
                isinstance(key, str) and is_json_value(item)
                for key, item in node.items()
            )
        return False

    if not is_json_value(value):
        raise MigrationError(
            f"{description} must produce a raw JSON value; got {type(value).__name__}"
        )
    return cast(JsonValue, value)


def _manifest_fields(
    manifest: JsonValue, *, description: str
) -> tuple[JsonFields, dict[str, Any] | None]:
    if not isinstance(manifest, dict):
        raise MigrationError(
            f"{description} needs a mapping, dataclass, or pydantic result at the "
            f"root; the stored result is {type(manifest).__name__}"
        )

    if WRAPPER_KEY not in manifest:
        return dict(manifest), None

    body = manifest[WRAPPER_KEY]
    if (
        isinstance(body, dict)
        and body.get(KINDMARKER) in ("dataclass", "pydantic")
        and isinstance(body.get(FIELDSMARKER), dict)
    ):
        return dict(cast(JsonFields, body[FIELDSMARKER])), cast(dict[str, Any], body)

    kind = body.get(KINDMARKER) if isinstance(body, dict) else None
    raise MigrationError(
        f"{description} needs a mapping, dataclass, or pydantic result at the "
        f"root; the stored result wrapper kind is {kind!r}"
    )


def _replace_manifest_fields(
    manifest: JsonValue,
    fields: JsonFields,
    wrapper_body: dict[str, Any] | None,
) -> JsonValue:
    if wrapper_body is None:
        return fields
    assert isinstance(manifest, dict)
    return {
        **manifest,
        WRAPPER_KEY: {
            **wrapper_body,
            FIELDSMARKER: fields,
        },
    }


def _apply_result_migration(
    manifest: JsonValue, migration: ResultMigration, *, owner: str
) -> JsonValue:
    description = f"{owner}: {_describe_result_migration(migration)}"
    match migration:
        case ResultAdded(field=field, default=default):
            fields, wrapper = _manifest_fields(manifest, description=description)
            if field in fields:
                raise MigrationError(
                    f"{description}: stored result already has field {field!r}; "
                    f"stored fields: {sorted(fields)}"
                )
            fields[field] = copy.deepcopy(
                _validate_json_value(default, description=description)
            )
            return _replace_manifest_fields(manifest, fields, wrapper)
        case ResultRenamed(field=field, to=to):
            fields, wrapper = _manifest_fields(manifest, description=description)
            if field not in fields:
                raise MigrationError(
                    f"{description}: stored result has no field {field!r}; "
                    f"stored fields: {sorted(fields)}"
                )
            if to in fields:
                raise MigrationError(
                    f"{description}: stored result already has target field {to!r}; "
                    f"stored fields: {sorted(fields)}"
                )
            fields[to] = fields.pop(field)
            return _replace_manifest_fields(manifest, fields, wrapper)
        case ResultRewrite(transform=transform):
            rewritten = transform(copy.deepcopy(manifest))
            return _validate_json_value(rewritten, description=description)
        case unreachable:
            assert_never(unreachable)


def _apply_result_migrations(
    manifest: JsonValue,
    steps: Iterable[MigrationStep],
) -> JsonValue:
    migrated = manifest
    for step in steps:
        if step.result is not None:
            migrated = _apply_result_migration(
                migrated,
                step.result,
                owner=_describe_step(step),
            )
    return migrated
