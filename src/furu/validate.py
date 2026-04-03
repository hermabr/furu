from typing import Any, Callable, Literal

_VALIDATOR_MARKER = "__furu_validator__"
_POST_INIT_WRAPPER_MARKER = "__furu_post_init_wrapper__"
_USER_POST_INIT_ATTR = "__furu_user_post_init__"
_CREATE_MODE_ATTR = "__furu_create_mode__"

type CreateMode = Literal["single", "batch"]


def validate(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    setattr(fn, _VALIDATOR_MARKER, True)
    return fn


def _resolve_create_mode(cls: type) -> CreateMode:
    defines_single = "_create" in cls.__dict__
    defines_batch = "_create_batched" in cls.__dict__

    if defines_single and defines_batch:
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} may define at most one of "
            "_create() or _create_batched()"
        )

    if defines_single:
        mode: CreateMode = "single"
    elif defines_batch:
        mode = "batch"
    else:
        inherited_mode = next(
            (
                getattr(base, _CREATE_MODE_ATTR)
                for base in cls.__mro__[1:]
                if getattr(base, _CREATE_MODE_ATTR, None) in ("single", "batch")
            ),
            None,
        )
        if inherited_mode is None:
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__} must define or inherit exactly "
                "one of _create() or _create_batched()"
            )
        mode = inherited_mode

    setattr(cls, _CREATE_MODE_ATTR, mode)
    return mode


def get_create_mode(cls: type) -> CreateMode:
    mode = getattr(cls, _CREATE_MODE_ATTR, None)
    if mode not in ("single", "batch"):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} does not have a resolved create mode"
        )
    return mode


def validate_cls(cls: type) -> None:
    _resolve_create_mode(cls)

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
        # TODO: runtime validate that post init does not break furu load from artifact logic
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
