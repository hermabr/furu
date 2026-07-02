from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from furu.storage._layout import data_dir_in, run_log_path_in


@dataclass(frozen=True)
class SpecDirectory:
    _base_dir: Path

    @cached_property
    def data(self) -> Path:
        data_dir = data_dir_in(self._base_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @cached_property
    def run_log(self) -> Path:
        return run_log_path_in(self._base_dir)
