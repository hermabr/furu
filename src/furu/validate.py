from dataclasses import MISSING, fields
from typing import Any, Callable, get_type_hints, overload

from furu._declared_types import is_skip_hash

_VALIDATOR_MARKER = "__furu_validator__"
_POST_INIT_WRAPPER_MARKER = "__furu_post_init_wrapper__"
_USER_POST_INIT_ATTR = "__furu_user_post_init__"


@overload
def validate(fn: Callable[[Any], None]) -> Callable[[Any], None]: ...


@overload
def validate() -> Callable[[Callable[[Any], None]], Callable[[Any], None]]: ...


def validate(
    fn: Callable[[Any], None] | None = None,
) -> Callable[[Any], None] | Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    def decorator(inner: Callable[[Any], None]) -> Callable[[Any], None]:
        setattr(inner, _VALIDATOR_MARKER, True)
        return inner

    if fn is None:
        return decorator
    return decorator(fn)


_skip_hash_validated_classes: set[type] = set()


def validate_skip_hash_fields(cls: type) -> None:
    # Resolved lazily (not in __init_subclass__) so forward references and
    # self-referential annotations are defined by the time hints are evaluated.
    if cls in _skip_hash_validated_classes:
        return
    hints = get_type_hints(cls, include_extras=True)
    for field in fields(cls):
        if not is_skip_hash(hints.get(field.name)):
            continue
        if field.default is MISSING and field.default_factory is MISSING:
            raise TypeError(
                f"{cls.__name__}.{field.name} is marked furu.skip_hash and must declare "
                "a default; skip_hash fields are absent from the artifact and cannot be "
                "reconstructed when loading from it"
            )
    _skip_hash_validated_classes.add(cls)


def validate_cls(cls: type) -> None:
    post_init_chain = [
        post_init
        for base in reversed(cls.__mro__[:-1])
        if (
            post_init := base.__dict__.get(_USER_POST_INIT_ATTR)
            or (
                base.__dict__.get("__post_init__")
                if not getattr(
                    base.__dict__.get("__post_init__"),
                    _POST_INIT_WRAPPER_MARKER,
                    False,
                )
                else None
            )
        )
        is not None
    ]
    validators = [
        value
        for base in reversed(cls.__mro__[:-1])
        for value in base.__dict__.values()
        if getattr(value, _VALIDATOR_MARKER, False)
    ]
    if not validators:
        return

    def __post_init__(self, *init_vars: Any) -> None:
        # TODO: runtime validate that post init does not break furu load from artifact logic
        for post_init in post_init_chain:
            post_init(self, *init_vars)
        for validator in validators:
            validator(self)

    user_post_init = cls.__dict__.get("__post_init__")
    if user_post_init is not None and not getattr(
        user_post_init, _POST_INIT_WRAPPER_MARKER, False
    ):
        setattr(cls, _USER_POST_INIT_ATTR, user_post_init)
    setattr(__post_init__, _POST_INIT_WRAPPER_MARKER, True)
    setattr(cls, "__post_init__", __post_init__)
