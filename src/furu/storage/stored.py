from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from furu._declared_types import declared_result_type
from furu.code_trace import valid_version
from furu.migration.schema import check_defaults
from furu.result.bundle import load_result_bundle
from furu.storage._layout import data_dir_in, result_dir_in

if TYPE_CHECKING:
    from furu.core import Spec


def stored_version(obj: Spec) -> Path | None:
    """The version directory holding obj's result, if its code still matches."""
    check_defaults(obj._base_dir.parent, type(obj))
    return valid_version(
        obj._base_dir, obj._metadata.code_version, storage_root=obj._metadata.storage
    )


def load_stored_result[T](obj: Spec[T], version: Path) -> T:
    (value,) = load_result_bundle(
        result_dir_in(version),
        [0],
        data_dir=data_dir_in(version),
        declared_type=declared_result_type(type(obj)),
    )
    return cast(T, value)
