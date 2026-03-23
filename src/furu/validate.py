from typing import Any, Callable

from furu.logging import get_logger

_VALIDATOR_MARKER = "__furu_validator__"
logger = get_logger(__name__)


def validate(fn: Callable[[Any], None]) -> Callable[[Any], None]:
    logger.debug("registering validator %s", getattr(fn, "__qualname__", repr(fn)))
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
        logger.debug("no validators found for %s", cls.__qualname__)
        return

    logger.debug(
        "attaching %s validator(s) to %s",
        len(local_validators),
        cls.__qualname__,
    )

    def __post_init__(self) -> None:
        for validator in local_validators:
            logger.debug(
                "running validator %s for %s",
                getattr(validator, "__qualname__", repr(validator)),
                type(self).__qualname__,
            )
            validator(self)

    setattr(cls, "__post_init__", __post_init__)
