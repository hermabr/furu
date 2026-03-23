from typing import Any, Callable

_VALIDATOR_MARKER = "__furu_validator__"
_POST_INIT_WRAPPER_MARKER = "__furu_post_init_wrapper__"
_USER_POST_INIT_ATTR = "__furu_user_post_init__"


def validate(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    setattr(fn, _VALIDATOR_MARKER, True)
    return fn


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

    def __post_init__(self) -> None:
        for post_init in post_init_chain:
            post_init(self)
        for validator in validators:
            validator(self)

    user_post_init = cls.__dict__.get("__post_init__")
    if user_post_init is not None and not getattr(
        user_post_init, _POST_INIT_WRAPPER_MARKER, False
    ):
        setattr(cls, _USER_POST_INIT_ATTR, user_post_init)
    setattr(__post_init__, _POST_INIT_WRAPPER_MARKER, True)
    setattr(cls, "__post_init__", __post_init__)
