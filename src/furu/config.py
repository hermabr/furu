import os
from importlib import import_module
from pathlib import Path


class FuruConfig:
    """Central configuration for Furu behavior."""

    VERSION_CONTROLLED_SUBDIR = Path("furu-data") / "artifacts"

    def __init__(self):
        def _get_base_root() -> Path:
            env = os.getenv("FURU_PATH")
            if env:
                return Path(env).expanduser().resolve()
            return Path("data-furu").resolve()

        self.base_root = _get_base_root()
        self.version_controlled_root_override = self._get_version_controlled_override()
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
        always_rerun_items = {
            item.strip()
            for item in os.getenv("FURU_ALWAYS_RERUN", "").split(",")
            if item.strip()
        }
        all_entries = {item for item in always_rerun_items if item.lower() == "all"}
        if all_entries and len(always_rerun_items) > len(all_entries):
            raise ValueError(
                "FURU_ALWAYS_RERUN cannot combine 'ALL' with specific entries"
            )
        self.always_rerun_all = bool(all_entries)
        if self.always_rerun_all:
            always_rerun_items = {
                item for item in always_rerun_items if item.lower() != "all"
            }
        self._require_namespaces_exist(always_rerun_items)
        self.always_rerun = always_rerun_items
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
        """Get root directory for storage (version_controlled uses its own root)."""
        if version_controlled:
            if self.version_controlled_root_override is not None:
                return self.version_controlled_root_override
            return self._resolve_version_controlled_root()
        return self.base_root / "data"

    @classmethod
    def _get_version_controlled_override(cls) -> Path | None:
        env = os.getenv("FURU_VERSION_CONTROLLED_PATH")
        if env:
            return Path(env).expanduser().resolve()
        return None

    @classmethod
    def _resolve_version_controlled_root(cls) -> Path:
        project_root = cls._find_project_root()
        return (project_root / cls.VERSION_CONTROLLED_SUBDIR).resolve()

    @staticmethod
    def _find_project_root(start: Path | None = None) -> Path:
        base = (start or Path.cwd()).resolve()
        for path in (base, *base.parents):
            if (path / "pyproject.toml").is_file():
                return path
        raise ValueError(
            "Cannot locate pyproject.toml to determine version-controlled root. "
            "Set FURU_VERSION_CONTROLLED_PATH to override."
        )

    @staticmethod
    def _require_namespaces_exist(namespaces: set[str]) -> None:
        if not namespaces:
            return
        missing_sentinel = object()
        for namespace in namespaces:
            module_name, _, qualname = namespace.rpartition(".")
            if not module_name or not qualname:
                raise ValueError(
                    "FURU_ALWAYS_RERUN entries must be 'module.QualifiedName', "
                    f"got {namespace!r}"
                )
            target: object = import_module(module_name)
            for attr in qualname.split("."):
                value = getattr(target, attr, missing_sentinel)
                if value is missing_sentinel:
                    raise ValueError(
                        "FURU_ALWAYS_RERUN entry does not exist: "
                        f"{namespace!r}"
                    )
                target = value

    @property
    def raw_dir(self) -> Path:
        return self.base_root / "raw"


FURU_CONFIG = FuruConfig()


def get_furu_root(*, version_controlled: bool = False) -> Path:
    return FURU_CONFIG.get_root(version_controlled=version_controlled)


def set_furu_root(path: Path) -> None:
    FURU_CONFIG.base_root = path.resolve()
