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


def install_validation_hooks(cls: type) -> None:
    user_post_init = cls.__dict__.get("__post_init__")
    local_validators = cls.__dict__.get(_LOCAL_VALIDATORS_ATTR, [])
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
