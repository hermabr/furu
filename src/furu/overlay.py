from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from furu import plan, record, remember
from furu.config import _set_config, get_config
from furu.dependencies import any_under_creation


@contextmanager
def overlay(path: str | os.PathLike[str] | None = None) -> Generator[Path]:
    """Run against the real Furu store read-only, writing everything to ``path``.

    Existing results in the real data directories (and custom
    ``Metadata(storage=...)`` locations) remain readable as cache hits and are
    never mutated or locked. All new writes (computed results, attempts, plans,
    executions, provenance, snapshots, remembered dependencies, and schema
    updates) go to the overlay directory, so deleting it cleanly erases the
    session.

    When ``path`` is omitted, a temporary directory is created and deleted on
    exit. When ``path`` is given, it is kept on exit so you can inspect outputs
    or reuse it across runs; pass a path on shared storage when using Slurm
    workers.
    """
    if any_under_creation():
        raise RuntimeError("furu.overlay() cannot be entered inside create()")
    previous = get_config()
    if previous.overlay is not None:
        raise RuntimeError(
            f"furu.overlay() cannot be nested inside active overlay {previous.overlay}"
        )
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="furu-overlay-")
        target = Path(temp_dir.name)
    else:
        given = Path(path)
        target = given if given.is_absolute() else Path.cwd() / given
    active = previous.model_copy(update={"overlay": target})
    active.writable_directories()
    _set_config(active)
    try:
        yield target
    finally:
        _set_config(previous)
        plan._close_under(target)
        record._close_under(target)
        remember._drop_under(target)
        if temp_dir is not None:
            temp_dir.cleanup()
