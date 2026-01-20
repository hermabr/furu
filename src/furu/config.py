import os
from pathlib import Path


class FuruConfig:
    """Central configuration for Furu behavior."""

    def __init__(self):
        def _get_base_root() -> Path:
            env = os.getenv("FURU_PATH")
            if env:
                return Path(env).expanduser().resolve()
            return Path("data-furu").resolve()

        self.base_root = _get_base_root()
        self.poll_interval = float(os.getenv("FURU_POLL_INTERVAL_SECS", "10"))
        self.wait_log_every_sec = float(os.getenv("FURU_WAIT_LOG_EVERY_SECS", "10"))
        self.stale_timeout = float(os.getenv("FURU_STALE_AFTER_SECS", str(30 * 60)))
        self.lease_duration_sec = float(os.getenv("FURU_LEASE_SECS", "120"))
        hb = os.getenv("FURU_HEARTBEAT_SECS")
        self.heartbeat_interval_sec = (
            float(hb) if hb is not None else max(1.0, self.lease_duration_sec / 3.0)
        )
        self.max_requeues = int(os.getenv("FURU_PREEMPT_MAX", "5"))
        self.ignore_git_diff = os.getenv("FURU_IGNORE_DIFF", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.require_git = os.getenv("FURU_REQUIRE_GIT", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self.require_git_remote = os.getenv("FURU_REQUIRE_GIT_REMOTE", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self.force_recompute = {
            item.strip()
            for item in os.getenv("FURU_FORCE_RECOMPUTE", "").split(",")
            if item.strip()
        }
        self.cancelled_is_preempted = os.getenv(
            "FURU_CANCELLED_IS_PREEMPTED", "false"
        ).lower() in {"1", "true", "yes"}

        # Parse FURU_CACHE_METADATA: "never", "forever", or duration like "5m", "1h"
        # Default: "5m" (5 minutes) - balances performance with freshness
        self.cache_metadata_ttl_sec: float | None = self._parse_cache_duration(
            os.getenv("FURU_CACHE_METADATA", "5m")
        )

    @staticmethod
    def _parse_cache_duration(value: str) -> float | None:
        """Parse cache duration string into seconds. Returns None for 'never', float('inf') for 'forever'."""
        value = value.strip().lower()
        if value in {"never", "0", "false", "no"}:
            return None  # No caching
        if value in {"forever", "inf", "true", "yes", "1"}:
            return float("inf")  # Cache forever

        # Parse duration like "5m", "1h", "30s"
        import re

        match = re.match(r"^(\d+(?:\.\d+)?)\s*([smh]?)$", value)
        if not match:
            raise ValueError(
                f"Invalid FURU_CACHE_METADATA value: {value!r}. "
                "Use 'never', 'forever', or duration like '5m', '1h', '30s'"
            )

        num = float(match.group(1))
        unit = match.group(2) or "s"
        multipliers = {"s": 1, "m": 60, "h": 3600}
        return num * multipliers[unit]

    def get_root(self, version_controlled: bool = False) -> Path:
        """Get root directory for storage (version_controlled determines subdirectory)."""
        if version_controlled:
            return self.base_root / "git"
        return self.base_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.base_root / "raw"


FURU_CONFIG = FuruConfig()


def get_furu_root(*, version_controlled: bool = False) -> Path:
    return FURU_CONFIG.get_root(version_controlled=version_controlled)


def set_furu_root(path: Path) -> None:
    FURU_CONFIG.base_root = path.resolve()
