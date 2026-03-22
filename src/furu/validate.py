from typing import Any, Callable


class validate:
    def __init__(self, fn: Callable[[Any], None]):
        self.fn = fn

    def __set_name__(self, owner: Any, name: str) -> None:
        if "__furu_local_validators__" not in owner.__dict__:
            owner.__furu_local_validators__ = []
        owner.__furu_local_validators__.append(self.fn)
        setattr(owner, name, self.fn)


def run_validation(instance: Any, validators: list[Callable[[Any], None]]) -> None:
    for validator in validators:
        validator(instance)


def find_inherited_post_init(cls: type) -> Any:
    for base in cls.__mro__[1:]:
        post_init = base.__dict__.get("__post_init__")
        if post_init is not None:
            return post_init
    return None


def install_validation_hooks(cls: type) -> None:
    user_post_init = cls.__dict__.get("__post_init__")
    inherited_post_init = find_inherited_post_init(cls)
    local_validators = cls.__dict__.get("__furu_local_validators__", [])
    if user_post_init is None and not local_validators:
        return

    def __post_init__(self, *args: Any, **kwds: Any) -> None:
        if inherited_post_init is not None:
            inherited_post_init(self, *args, **kwds)
        if user_post_init is not None:
            user_post_init(self, *args, **kwds)
        run_validation(self, local_validators)

    setattr(cls, "__post_init__", __post_init__)
