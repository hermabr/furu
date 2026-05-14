from __future__ import annotations

import contextvars
import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from furu.config import config

if TYPE_CHECKING:
    from furu.core import Furu


_current_executor_dir = contextvars.ContextVar[Path | None](
    "furu_current_executor_dir",
    default=None,
)


def executor_id_from_objs(objs: Sequence[Furu]) -> str:
    digest = hashlib.blake2s(digest_size=16)
    for obj in objs:
        digest.update(obj.object_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def executor_dir_for_id(executor_id: str) -> Path:
    return config.directories.executions / executor_id


def current_executor_dir() -> Path | None:
    return _current_executor_dir.get()


@contextmanager
def executor_context(executor_dir: Path) -> Iterator[None]:
    token = _current_executor_dir.set(executor_dir.resolve())
    try:
        yield
    finally:
        _current_executor_dir.reset(token)
