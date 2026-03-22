from typing import Any, Callable

_LOCAL_VALIDATORS_ATTR = "__furu_local_validators__"


class validate:
    def __init__(self, fn: Callable[[Any], None]):
        self.fn = fn

    def __set_name__(self, owner: Any, name: str) -> None:
        validators = owner.__dict__.get(_LOCAL_VALIDATORS_ATTR)
        if validators is None:
            validators = []
            setattr(owner, _LOCAL_VALIDATORS_ATTR, validators)
        validators.append(self.fn)
        setattr(owner, name, self.fn)


def _get_local_validators(cls: type) -> list[Callable[[Any], None]]:
    validators = cls.__dict__.get(_LOCAL_VALIDATORS_ATTR)
    if validators is None:
        return []
    return validators


def _find_inherited_post_init(cls: type) -> Callable[..., None] | None:
    for base in cls.__mro__[1:]:
        post_init = base.__dict__.get("__post_init__")
        if post_init is not None:
            return post_init
    return None


def install_validation_hooks(cls: type) -> None:
    user_post_init = cls.__dict__.get("__post_init__")
    local_validators = _get_local_validators(cls)
    if user_post_init is None and not local_validators:
        return
    inherited_post_init = _find_inherited_post_init(cls)

    def __post_init__(self, *args: Any, **kwds: Any) -> None:
        if inherited_post_init is not None:
            inherited_post_init(self, *args, **kwds)
        if user_post_init is not None:
            user_post_init(self, *args, **kwds)
        for validator in local_validators:
            validator(self)

    setattr(cls, "__post_init__", __post_init__)
