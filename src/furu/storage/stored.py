from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from furu._declared_types import declared_result_type
from furu.code_trace import versions
from furu.migration.schema import check_defaults
from furu.result.bundle import load_result_bundle
from furu.storage._layout import FIXED, data_dir_in, result_dir_in, run_dir_in

if TYPE_CHECKING:
    from furu.core import Spec


def stored_run(obj: Spec) -> Path | None:
    """The published run that holds obj's result, if its code still matches.

    A run is found by name: one file check per live code version.
    """
    # Named first: that is what finds the dataclasses the spec holds, whose
    # defaults are checked too.
    name = obj._run_name
    check_defaults(obj._class_dir, type(obj))
    code_version = obj._metadata.code_version
    if code_version == "fixed":
        run = run_dir_in(obj._class_dir / FIXED, name)
        if run.is_dir():
            return run  # the common case
    for version, why in versions(obj._class_dir, code_version):
        if why is None and (run := run_dir_in(version, name)).is_dir():
            return run
    return None


def why_rerun(obj: Spec) -> str | None:
    """Why the newest run stored for obj no longer counts, if it has one."""
    outdated = [
        (run.stat().st_mtime_ns, why)
        for version, why in versions(obj._class_dir, obj._metadata.code_version)
        if why is not None and (run := run_dir_in(version, obj._run_name)).is_dir()
    ]
    return max(outdated)[1] if outdated else None


def load_stored_result[T](obj: Spec[T], run: Path) -> T:
    (value,) = load_result_bundle(
        result_dir_in(run),
        [0],
        data_dir=data_dir_in(run),
        declared_type=declared_result_type(type(obj)),
    )
    return cast(T, value)
