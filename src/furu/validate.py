from typing import Any, Callable


class validate:
    def __init__(self, fn: Callable[[Any], None]):
        self.fn = fn

    def __set_name__(self, owner: Any, name: str) -> None:
        if "__furu_local_validators__" not in owner.__dict__:
            owner.__furu_local_validators__ = []
        owner.__furu_local_validators__.append(self.fn)
        setattr(owner, name, self.fn)

def install_validation_hooks(cls: type) -> None:
    user_post_init = cls.__dict__.get("__post_init__")
    local_validators = cls.__dict__.get("__furu_local_validators__", [])
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
