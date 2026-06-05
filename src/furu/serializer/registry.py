from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from functools import cache
from threading import Lock
from typing import Annotated, Any, ClassVar, cast, final, get_args, get_origin

from furu._declared_types import strip_annotated
from furu.utils import JsonValue, fully_qualified_name


class ArtifactSerializerMeta(ABCMeta):
    _auto_registered_serializers: list[type[ArtifactSerializer]] = []
    _auto_registered_serializers_lock = Lock()

    def __init__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        is_root_serializer_class = not any(
            isinstance(base, ArtifactSerializerMeta) for base in bases
        )
        if is_root_serializer_class:
            return
        if not namespace.get("auto_register", True):
            return
        if getattr(cls, "__abstractmethods__", None):
            return

        cls.auto_register = True
        with ArtifactSerializerMeta._auto_registered_serializers_lock:
            ArtifactSerializerMeta._auto_registered_serializers.append(
                cast(type[ArtifactSerializer], cls)
            )

        ArtifactSerializerMeta._default_serializers.cache_clear()

    @classmethod
    def auto_registered_serializers(mcls) -> tuple[type[ArtifactSerializer], ...]:
        with mcls._auto_registered_serializers_lock:
            return tuple(reversed(mcls._auto_registered_serializers))

    @classmethod
    @cache
    def _default_serializers(mcls) -> tuple[type[ArtifactSerializer], ...]:
        return tuple(
            serializer
            for serializer in mcls.auto_registered_serializers()
            if serializer.dependencies_available()
        )

    @classmethod
    def serializer_for_schema(
        mcls,
        declared_type: object,
        artifact_serializers: tuple[type[ArtifactSerializer], ...],
    ) -> type[ArtifactSerializer] | None:
        if serializer := _annotated_serializer(declared_type):
            return serializer

        declared_type = strip_annotated(declared_type)
        if isinstance(declared_type, type):
            if serializer := _class_serializer(declared_type):
                return serializer

        origin = get_origin(declared_type)
        if isinstance(origin, type):
            if serializer := _class_serializer(origin):
                return serializer

        if serializer := _find_single_serializer_match(
            artifact_serializers,
            matches=lambda serializer: serializer.matches_type(declared_type),
            layer_name="artifact serializers",
            match_kind="schema",
        ):
            return serializer
        return _find_single_serializer_match(
            mcls._default_serializers(),
            matches=lambda serializer: serializer.matches_type(declared_type),
            layer_name="auto-registered serializer registry",
            match_kind="schema",
        )

    @classmethod
    def serializer_for_dump(
        mcls,
        value: object,
        *,
        declared_type: object,
        artifact_serializers: tuple[type[ArtifactSerializer], ...],
    ) -> type[ArtifactSerializer] | None:
        if serializer := _annotated_serializer(declared_type):
            return serializer

        declared_type = strip_annotated(declared_type)
        if isinstance(declared_type, type):
            if serializer := _class_serializer(declared_type):
                return serializer

        if serializer := _class_serializer(type(value)):
            return serializer

        if serializer := _find_single_serializer_match(
            artifact_serializers,
            matches=lambda serializer: serializer.matches(value),
            layer_name="artifact serializers",
            match_kind="dump",
        ):
            return serializer
        return _find_single_serializer_match(
            mcls._default_serializers(),
            matches=lambda serializer: serializer.matches(value),
            layer_name="auto-registered serializer registry",
            match_kind="dump",
        )


class ArtifactSerializer[T](ABC, metaclass=ArtifactSerializerMeta):
    auto_register: ClassVar[bool] = True

    @final
    @classmethod
    def _serializer_id(cls) -> str:
        return fully_qualified_name(cls)

    @classmethod
    def dependencies_available(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def matches(cls, value: object) -> bool:
        pass

    @classmethod
    @abstractmethod
    def matches_type(cls, declared_type: object) -> bool:
        pass

    @classmethod
    @abstractmethod
    def schema(cls, declared_type: object) -> JsonValue:
        pass

    @classmethod
    @abstractmethod
    def dump(cls, value: T, *, declared_type: object) -> JsonValue:
        pass

    @classmethod
    @abstractmethod
    def load(cls, value: JsonValue, *, declared_type: object) -> T:
        pass


def _annotated_serializer(declared_type: object) -> type[ArtifactSerializer] | None:
    if get_origin(declared_type) is not Annotated:
        return None

    for metadata in get_args(declared_type)[1:]:
        if isinstance(metadata, type) and issubclass(metadata, ArtifactSerializer):
            return metadata
    return None


def _class_serializer(cls: type) -> type[ArtifactSerializer] | None:
    _SERIALIZER_ATTR = "__furu_serializer__"
    provider = getattr(cls, _SERIALIZER_ATTR, None)
    if provider is None:
        return None
    if isinstance(provider, type) and issubclass(provider, ArtifactSerializer):
        return provider
    if callable(provider):
        serializer = provider()
        if isinstance(serializer, type) and issubclass(serializer, ArtifactSerializer):
            return serializer

    raise TypeError(
        f"{cls.__module__}.{cls.__qualname__}.{_SERIALIZER_ATTR} must be an ArtifactSerializer "
        "class or a zero-argument callable returning one"
    )


def _find_single_serializer_match(
    serializers: tuple[type[ArtifactSerializer], ...],
    *,
    matches: Callable[[type[ArtifactSerializer]], bool],
    layer_name: str,
    match_kind: str,
) -> type[ArtifactSerializer] | None:
    matching_serializer: type[ArtifactSerializer] | None = None
    for serializer in serializers:
        if not matches(serializer):
            continue
        if matching_serializer is None:
            matching_serializer = serializer
            continue

        raise TypeError(
            f"{layer_name} matched multiple serializers for {match_kind}: "
            f"{matching_serializer.__name__}, {serializer.__name__}"
        )
    return matching_serializer
