from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


OverrideValue = Any

_OVERRIDES: ContextVar[dict[str, OverrideValue]] = ContextVar(
    "FURU_RESULT_OVERRIDES",
    default={},
)


def has_override(furu_hash: str) -> bool:
    return furu_hash in _OVERRIDES.get()


def lookup_override(furu_hash: str) -> tuple[bool, OverrideValue]:
    overrides = _OVERRIDES.get()
    if furu_hash in overrides:
        return True, overrides[furu_hash]
    return False, None


@contextmanager
def override_furu_hashes(
    overrides: Mapping[str, OverrideValue],
) -> Generator[None, None, None]:
    current = _OVERRIDES.get()
    merged = dict(current)
    merged.update(overrides)
    token = _OVERRIDES.set(merged)
    try:
        yield
    finally:
        _OVERRIDES.reset(token)
