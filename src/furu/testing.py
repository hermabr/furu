from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from .config import FURU_CONFIG, RecordGitMode
from .core import Furu
from .runtime.overrides import OverrideValue, override_furu_hashes

OverrideKey = Furu | str


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


@contextmanager
def override_results(
    overrides: Mapping[OverrideKey, OverrideValue],
) -> Generator[None, None, None]:
    """Override specific Furu results within the context.

    Overrides are keyed by furu_hash, so identical configs share a stub.
    Keys may be a Furu instance or a furu_hash string.
    """
    hash_overrides: dict[str, OverrideValue] = {}
    for key, value in overrides.items():
        hash_key: str
        if isinstance(key, Furu):
            hash_key = cast(str, key.furu_hash)
        elif isinstance(key, str):
            if not key:
                raise ValueError("override furu_hash must be non-empty")
            hash_key = key
        else:
            raise TypeError(
                "override_results keys must be Furu instances or furu_hash strings"
            )
        hash_overrides[hash_key] = value
    with override_furu_hashes(hash_overrides):
        yield


@pytest.fixture()
def furu_tmp_root(tmp_path: Path) -> Generator[Path, None, None]:
    """Configure furu to use a temporary root for the test."""
    with furu_test_env(tmp_path) as root:
        yield root
