from dataclasses import dataclass, field
from pathlib import Path
from typing import Self


@dataclass(slots=True)
class _FuruDirectories:
    data: Path
    # TODO: make this better and more user configurable, so that it is easy for the user to define exactly which paths they want to save to and which furu objects should save where

    @classmethod
    def default(cls) -> Self:
        # TODO: make sure this location is deterministic/more predictable, such as by finding the next .git or pyproject.toml or furu directory
        base_dir = Path("furu")
        data_dir = base_dir / "data"
        return cls(data=data_dir)


@dataclass(slots=True)
class _FuruConfig:
    debug_mode: bool = False
    directories: _FuruDirectories = field(default_factory=_FuruDirectories.default)


config = _FuruConfig()
