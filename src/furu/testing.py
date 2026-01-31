from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from .config import FURU_CONFIG, RecordGitMode


@dataclass(frozen=True)
class _FuruConfigSnapshot:
    base_root: Path
    version_controlled_root_override: Path | None
    record_git: RecordGitMode
    allow_no_git_origin: bool
    poll_interval: float
    stale_timeout: float
    max_wait_time_sec: float | None
    lease_duration_sec: float
    heartbeat_interval_sec: float
    cancelled_is_preempted: bool
    retry_failed: bool
    submitit_root: Path

    @classmethod
    def capture(cls) -> "_FuruConfigSnapshot":
        return cls(
            base_root=FURU_CONFIG.base_root,
            version_controlled_root_override=FURU_CONFIG.version_controlled_root_override,
            record_git=FURU_CONFIG.record_git,
            allow_no_git_origin=FURU_CONFIG.allow_no_git_origin,
            poll_interval=FURU_CONFIG.poll_interval,
            stale_timeout=FURU_CONFIG.stale_timeout,
            max_wait_time_sec=FURU_CONFIG.max_wait_time_sec,
            lease_duration_sec=FURU_CONFIG.lease_duration_sec,
            heartbeat_interval_sec=FURU_CONFIG.heartbeat_interval_sec,
            cancelled_is_preempted=FURU_CONFIG.cancelled_is_preempted,
            retry_failed=FURU_CONFIG.retry_failed,
            submitit_root=FURU_CONFIG.submitit_root,
        )

    def restore(self) -> None:
        FURU_CONFIG.base_root = self.base_root
        FURU_CONFIG.version_controlled_root_override = (
            self.version_controlled_root_override
        )
        FURU_CONFIG.record_git = self.record_git
        FURU_CONFIG.allow_no_git_origin = self.allow_no_git_origin
        FURU_CONFIG.poll_interval = self.poll_interval
        FURU_CONFIG.stale_timeout = self.stale_timeout
        FURU_CONFIG.max_wait_time_sec = self.max_wait_time_sec
        FURU_CONFIG.lease_duration_sec = self.lease_duration_sec
        FURU_CONFIG.heartbeat_interval_sec = self.heartbeat_interval_sec
        FURU_CONFIG.cancelled_is_preempted = self.cancelled_is_preempted
        FURU_CONFIG.retry_failed = self.retry_failed
        FURU_CONFIG.submitit_root = self.submitit_root


def _apply_test_config(base_root: Path) -> Path:
    root = base_root.resolve()
    FURU_CONFIG.base_root = root
    FURU_CONFIG.version_controlled_root_override = root / "furu-data" / "artifacts"
    FURU_CONFIG.record_git = "ignore"
    FURU_CONFIG.allow_no_git_origin = False
    FURU_CONFIG.poll_interval = 0.01
    FURU_CONFIG.stale_timeout = 0.1
    FURU_CONFIG.max_wait_time_sec = None
    FURU_CONFIG.lease_duration_sec = 0.05
    FURU_CONFIG.heartbeat_interval_sec = 0.01
    FURU_CONFIG.cancelled_is_preempted = True
    FURU_CONFIG.retry_failed = True
    FURU_CONFIG.submitit_root = root / "submitit"
    return root


@contextmanager
def furu_test_env(base_root: Path) -> Generator[Path, None, None]:
    snapshot = _FuruConfigSnapshot.capture()
    root = _apply_test_config(base_root)
    try:
        yield root
    finally:
        snapshot.restore()


@pytest.fixture()
def furu_tmp_root(tmp_path: Path) -> Generator[Path, None, None]:
    """Configure furu to use a temporary root for the test."""
    with furu_test_env(tmp_path) as root:
        yield root
