from collections import deque
from dataclasses import dataclass
from typing import Any, cast

from furu.constants import CLASSMARKER, FIELDSMARKER, KINDMARKER
from furu.core import Spec
from furu.utils import JsonValue


@dataclass(frozen=True, kw_only=True)
class FieldDiff:
    path: str
    a: JsonValue
    b: JsonValue


def _as_instance_node(
    value: JsonValue,
) -> tuple[JsonValue, dict[str, JsonValue]] | None:
    if isinstance(value, dict) and value.get(KINDMARKER) == "instance":
        return value[CLASSMARKER], cast("dict[str, JsonValue]", value[FIELDSMARKER])
    return None


def diff(a: Spec[Any], b: Spec[Any]) -> list[FieldDiff]:
    """Identity differences between two specs, one FieldDiff per differing field.

    Walks the same serialized data the field hash is computed from, so skip_hash
    fields are omitted and an empty diff means identical identity. Nested specs
    and dataclasses recurse into dotted paths; collections diff as whole values;
    a class mismatch is reported as a single FieldDiff without recursing below it.
    """
    diffs: list[FieldDiff] = []
    queue: deque[tuple[str, JsonValue, JsonValue]] = deque(
        [("", a._artifact_data_for_hash, b._artifact_data_for_hash)]
    )
    while queue:
        path, a_value, b_value = queue.popleft()
        a_node = _as_instance_node(a_value)
        b_node = _as_instance_node(b_value)
        if a_node is not None and b_node is not None:
            a_class, a_fields = a_node
            b_class, b_fields = b_node
            if a_class != b_class:
                diffs.append(FieldDiff(path=path, a=a_class, b=b_class))
                continue
            for name in {**a_fields, **b_fields}:
                queue.append(
                    (
                        f"{path}.{name}" if path else name,
                        a_fields.get(name),
                        b_fields.get(name),
                    )
                )
        elif a_value != b_value:
            diffs.append(FieldDiff(path=path, a=a_value, b=b_value))
    return diffs
