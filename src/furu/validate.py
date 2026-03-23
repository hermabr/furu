from typing import Any, Callable

_VALIDATOR_MARKER = "__furu_validator__"


def validate(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    setattr(fn, _VALIDATOR_MARKER, True)
    return fn


def validate_cls(cls: type) -> None:
    local_validators = [
        value
        for base in reversed(cls.__mro__[:-1])
        for value in base.__dict__.values()
        if getattr(value, _VALIDATOR_MARKER, False)
    ]
    if not local_validators:
        return

    def __post_init__(self) -> None:
        for validator in local_validators:
            validator(self)

    setattr(cls, "__post_init__", __post_init__)
