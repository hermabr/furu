from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from furu.storage._layout import data_dir_in, scratch_dir_in


@dataclass(frozen=True)
class SpecDirectory:
    """The folders a create() writes in, inside its working folder.

    They are made in the working folder and never with it: one that was
    taken from this attempt stays gone, so its files land nowhere.
    """

    _base_dir: Path

    @cached_property
    def data(self) -> Path:
        data_dir = data_dir_in(self._base_dir)
        data_dir.mkdir(exist_ok=True)
        return data_dir

    @property
    def scratch(self) -> Path:
        """Scratch space for ``create()``: kept across retries, deleted once
        codecs persist the result. Not cached, so accesses re-create it.
        """
        scratch_dir = scratch_dir_in(self._base_dir)
        scratch_dir.mkdir(exist_ok=True)
        return scratch_dir
