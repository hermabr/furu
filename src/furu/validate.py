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
