import os
from importlib import import_module
from pathlib import Path
from typing import Literal, cast


RecordGitMode = Literal["ignore", "cached", "uncached"]


class FuruConfig:
    """Central configuration for Furu behavior."""

    DEFAULT_ROOT_DIR = Path("furu-data")
    VERSION_CONTROLLED_SUBDIR = DEFAULT_ROOT_DIR / "artifacts"

    def __init__(self):
        def _get_base_root() -> Path:
            env = os.getenv("FURU_PATH")
            if env:
                return Path(env).expanduser().resolve()
            project_root = self._find_project_root(fallback_to_cwd=True)
            return (project_root / self.DEFAULT_ROOT_DIR).resolve()

        self.base_root = _get_base_root()
        self.submitit_root = (
            Path(os.getenv("FURU_SUBMITIT_PATH", str(self.base_root / "submitit")))
            .expanduser()
            .resolve()
        )
        self.version_controlled_root_override = self._get_version_controlled_override()
        self.poll_interval = float(os.getenv("FURU_POLL_INTERVAL_SECS", "10"))
        self.wait_log_every_sec = float(os.getenv("FURU_WAIT_LOG_EVERY_SECS", "10"))
        self.stale_timeout = float(os.getenv("FURU_STALE_AFTER_SECS", str(30 * 60)))
        max_wait_env = os.getenv("FURU_MAX_WAIT_SECS")
        self.max_wait_time_sec = float(max_wait_env) if max_wait_env else None
        self.lease_duration_sec = float(os.getenv("FURU_LEASE_SECS", "120"))
        hb = os.getenv("FURU_HEARTBEAT_SECS")
        self.heartbeat_interval_sec = (
            float(hb) if hb is not None else max(1.0, self.lease_duration_sec / 3.0)
        )
        self.max_requeues = int(os.getenv("FURU_PREEMPT_MAX", "5"))
        self.max_compute_retries = int(os.getenv("FURU_MAX_COMPUTE_RETRIES", "3"))
        self.retry_failed = os.getenv("FURU_RETRY_FAILED", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self.record_git = self._parse_record_git(os.getenv("FURU_RECORD_GIT", "cached"))
        self.allow_no_git_origin = self._parse_bool(
            os.getenv("FURU_ALLOW_NO_GIT_ORIGIN", "0")
        )
        if self.allow_no_git_origin and self.record_git == "ignore":
            raise ValueError(
                "FURU_ALLOW_NO_GIT_ORIGIN cannot be enabled when FURU_RECORD_GIT=ignore"
            )
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

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes"}

    @classmethod
    def _parse_record_git(cls, value: str) -> RecordGitMode:
        normalized = value.strip().lower()
        allowed = {"ignore", "cached", "uncached"}
        if normalized not in allowed:
            raise ValueError(
                "FURU_RECORD_GIT must be one of 'ignore', 'cached', or 'uncached'"
            )
        return cast(RecordGitMode, normalized)

    @property
    def cache_metadata_ttl_sec(self) -> float | None:
        if self.record_git == "cached":
            return float("inf")
        return None

    def get_root(self, version_controlled: bool = False) -> Path:
        """Get root directory for storage (version_controlled uses its own root)."""
        if version_controlled:
            if self.version_controlled_root_override is not None:
                return self.version_controlled_root_override
            return self._resolve_version_controlled_root()
        return self.base_root / "data"

    def get_submitit_root(self) -> Path:
        return self.submitit_root

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
    def _find_project_root(
        start: Path | None = None, *, fallback_to_cwd: bool = False
    ) -> Path:
        base = (start or Path.cwd()).resolve()
        git_root: Path | None = None
        for path in (base, *base.parents):
            if (path / "pyproject.toml").is_file():
                return path
            if git_root is None and (path / ".git").exists():
                git_root = path
        if git_root is not None:
            return git_root
        if fallback_to_cwd:
            return base
        raise ValueError(
            "Cannot locate pyproject.toml or .git to determine version-controlled root. "
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
                        f"FURU_ALWAYS_RERUN entry does not exist: {namespace!r}"
                    )
                target = value

    @property
    def raw_dir(self) -> Path:
        return self.base_root / "raw"


FURU_CONFIG = FuruConfig()


def get_furu_root(*, version_controlled: bool = False) -> Path:
    return FURU_CONFIG.get_root(version_controlled=version_controlled)


def set_furu_root(path: Path) -> None:
    root = path.resolve()
    FURU_CONFIG.base_root = root
    if os.getenv("FURU_SUBMITIT_PATH") is None:
        FURU_CONFIG.submitit_root = (root / "submitit").resolve()
