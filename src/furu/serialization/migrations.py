from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from ..runtime.logging import get_logger
from .serializer import JsonValue


_T = TypeVar("_T")


@dataclass(frozen=True)
class MigrationContext:
    fields: dict[str, JsonValue]
    from_class: str
    to_class: str
    from_version: float
    to_version: float


@dataclass(frozen=True)
class FieldRename:
    old: str
    new: str


@dataclass(frozen=True)
class FieldAdd(Generic[_T]):
    name: str
    default: _T | None = None
    default_factory: Callable[[MigrationContext], _T] | None = None


@dataclass(frozen=True)
class Transform:
    func: Callable[[dict[str, JsonValue]], dict[str, JsonValue]]


MigrationStep = FieldRename | FieldAdd[JsonValue] | Transform


@dataclass(frozen=True)
class MigrationSpec:
    from_class: str
    from_version: float
    to_version: float
    steps: list[MigrationStep]
    to_class: str | None = None
    note: str | None = None

    def default_value_for(self, name: str, data: dict[str, JsonValue]) -> JsonValue:
        for step in self.steps:
            if isinstance(step, FieldAdd) and step.name == name:
                if step.default_factory is None:
                    return step.default
                context = MigrationContext(
                    fields={k: v for k, v in data.items() if k != "__class__"},
                    from_class=self.from_class,
                    to_class=self.to_class or self.from_class,
                    from_version=self.from_version,
                    to_version=self.to_version,
                )
                return step.default_factory(context)
        return None


class MigrationRegistry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, float], MigrationSpec] = {}

    def register(
        self, spec: MigrationSpec, *, default_to_class: str | None = None
    ) -> None:
        to_class = spec.to_class or default_to_class
        if to_class is None:
            raise ValueError("MigrationSpec.to_class is required")
        key = (spec.from_class, spec.from_version)
        if key in self._specs:
            raise ValueError(
                f"Duplicate migration for {spec.from_class}@{spec.from_version}"
            )
        normalized = MigrationSpec(
            from_class=spec.from_class,
            from_version=spec.from_version,
            to_version=spec.to_version,
            steps=spec.steps,
            to_class=to_class,
            note=spec.note,
        )
        self._specs[key] = normalized

    def resolve_chain(
        self,
        *,
        from_class: str,
        from_version: float,
        to_class: str | None = None,
        to_version: float | None = None,
    ) -> list[MigrationSpec]:
        chain: list[MigrationSpec] = []
        current_class = from_class
        current_version = from_version
        visited: set[tuple[str, float]] = set()

        while True:
            key = (current_class, current_version)
            if key in visited:
                raise ValueError(
                    f"Migration loop detected for {current_class}@{current_version}"
                )
            visited.add(key)

            spec = self._specs.get(key)
            if spec is None:
                break
            chain.append(spec)
            current_class = spec.to_class or spec.from_class
            current_version = spec.to_version

            if to_class is not None and to_version is not None:
                if current_class == to_class and current_version == to_version:
                    break

        if to_class is not None and to_version is not None:
            if current_class != to_class or current_version != to_version:
                raise ValueError(
                    f"No migration path from {from_class}@{from_version} to {to_class}@{to_version}"
                )

        if len(chain) > 1:
            get_logger().warning(
                "migration: chain length %s from %s@%s",
                len(chain),
                from_class,
                from_version,
            )
        return chain

    def apply_chain(
        self,
        data: dict[str, JsonValue],
        *,
        to_class: str | None = None,
        to_version: float | None = None,
    ) -> tuple[dict[str, JsonValue], list[MigrationSpec]]:
        from_class = _require_class_name(data)
        from_version = _get_version(data)
        chain = self.resolve_chain(
            from_class=from_class,
            from_version=from_version,
            to_class=to_class,
            to_version=to_version,
        )
        result = dict(data)
        for spec in chain:
            result = _apply_spec(result, spec)
        result = _apply_nested_migrations(result, registry=self)
        return result, chain

    def has_migration(self, from_class: str, from_version: float) -> bool:
        return (from_class, from_version) in self._specs


MIGRATION_REGISTRY = MigrationRegistry()


def _require_class_name(data: dict[str, JsonValue]) -> str:
        class_name = data.get("__class__")
        if not isinstance(class_name, str):
            raise ValueError("Serialized Furu object missing __class__")
        return class_name


def _get_version(data: dict[str, JsonValue]) -> float:
    version_value = data.get("furu_version")
    if isinstance(version_value, (float, int)):
        return float(version_value)
    return 0.0


def _apply_spec(
    data: dict[str, JsonValue], spec: MigrationSpec
) -> dict[str, JsonValue]:
    result = dict(data)
    result["__class__"] = spec.to_class or spec.from_class

    for step in spec.steps:
        if isinstance(step, FieldRename):
            if step.old in result:
                result[step.new] = result.pop(step.old)
            continue

        if isinstance(step, FieldAdd):
            if step.name not in result:
                context = MigrationContext(
                    fields={k: v for k, v in result.items() if k != "__class__"},
                    from_class=spec.from_class,
                    to_class=result["__class__"],
                    from_version=spec.from_version,
                    to_version=spec.to_version,
                )
                if step.default_factory is not None:
                    result[step.name] = step.default_factory(context)
                else:
                    result[step.name] = step.default
            continue

        if isinstance(step, Transform):
            result = step.func(result)
            continue

        raise TypeError(f"Unsupported migration step: {step}")

    result["furu_version"] = spec.to_version
    return result


def _apply_nested_migrations(
    data: dict[str, JsonValue], *, registry: MigrationRegistry
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "__class__" in value:
            nested = value
            if registry.has_migration(
                _require_class_name(nested), _get_version(nested)
            ):
                migrated, _ = registry.apply_chain(nested)
                result[key] = migrated
                continue
            result[key] = _apply_nested_migrations(nested, registry=registry)
            continue
        if isinstance(value, list):
            result[key] = [
                _apply_nested_migrations(item, registry=registry)
                if isinstance(item, dict)
                else item
                for item in value
            ]
            continue
        if isinstance(value, dict):
            result[key] = _apply_nested_migrations(value, registry=registry)
            continue
        result[key] = value
    return result
