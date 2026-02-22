import hashlib
import json
from enum import Enum
from typing import TypeAlias, Union

JsonValue: TypeAlias = Union[
    list["JsonValue"],
    dict[str, "JsonValue"],
    str,
    bool,
    int,
    float,
    None,
]


def fully_qualified_name(tp: type) -> str:
    mod = tp.__module__
    qualname = tp.__qualname__
    if mod == "__main__":  # TODO: allow overwriting
        raise ValueError("Cannot serialize objects from __main__ module")
    elif "<locals>" in mod:  # TODO: allow overwriting
        raise ValueError("TODO: msg")
    elif "." in qualname:
        raise ValueError("TODO: msg")
    elif isinstance(tp, Enum):
        raise ValueError(
            "TODO: support this in the future"
        )  # return f"{mod}.{qualname}.{obj.name}"
    return f"{mod}.{qualname}"


def _stable_json_dump(x: JsonValue) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _hash_dict_deterministically(obj: JsonValue) -> str:
    json_str = _stable_json_dump(obj)

    return hashlib.blake2s(
        json_str.encode(),
        digest_size=10,  # TODO: make this digest size configurable and include a script for estimating likelihood of crashing. right now, i think there is a 1e-08 chance of a collision with 155M items with the same schema and namespace
    ).hexdigest()
