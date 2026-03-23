from typing import Any, Callable

_VALIDATOR_MARKER = "__furu_validator__"


def validate(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    setattr(fn, _VALIDATOR_MARKER, True)
    return fn


def install_validation_hooks(cls: type) -> None:
    user_post_init = cls.__dict__.get("__post_init__")
    local_validators = [
        value
        for value in cls.__dict__.values()
        if getattr(value, _VALIDATOR_MARKER, False)
    ]
    if user_post_init is None and not local_validators:
        return
    inherited_post_init = next(
        (
            post_init
            for base in cls.__mro__[1:]
            if (post_init := base.__dict__.get("__post_init__")) is not None
        ),
        None,
    )

    def __post_init__(self, *args: Any, **kwds: Any) -> None:
        if inherited_post_init is not None:
            inherited_post_init(self, *args, **kwds)
        if user_post_init is not None:
            user_post_init(self, *args, **kwds)
        for validator in local_validators:
            validator(self)

    setattr(cls, "__post_init__", __post_init__)
