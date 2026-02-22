from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class _FuruConfig:
    debug_mode: bool = False
    base_directory: Path = Path(
        "furu-data"
    )  # TODO: allow this to be multiple different paths too


config = _FuruConfig()
