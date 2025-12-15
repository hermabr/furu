import os
from pathlib import Path
from typing import Self


class HuldraConfig:
    """Central configuration for Huldra behavior."""

    def __init__(self: Self):
        def _get_base_root() -> Path:
            env = os.getenv("HULDRA_PATH")
            if env:
                return Path(env).expanduser().resolve()
            return Path("data-huldra").resolve()

        self.base_root = _get_base_root()
        self.poll_interval = float(os.getenv("HULDRA_POLL_INTERVAL_SECS", "10"))
        self.stale_timeout = float(os.getenv("HULDRA_STALE_AFTER_SECS", str(30 * 60)))
        self.max_requeues = int(os.getenv("HULDRA_PREEMPT_MAX", "5"))
        self.ignore_git_diff = os.getenv("HULDRA_IGNORE_DIFF", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.cancelled_is_preempted = os.getenv(
            "HULDRA_CANCELLED_IS_PREEMPTED", "false"
        ).lower() in {"1", "true", "yes"}

    def get_root(self: Self, version_controlled: bool = False) -> Path:
        """Get root directory for storage (version_controlled determines subdirectory)."""
        if version_controlled:
            return self.base_root / "git"
        return self.base_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.base_root / "raw"


HULDRA_CONFIG = HuldraConfig()


def get_huldra_root(*, version_controlled: bool = False) -> Path:
    return HULDRA_CONFIG.get_root(version_controlled=version_controlled)


def set_huldra_root(path: Path) -> None:
    HULDRA_CONFIG.base_root = path.resolve()

