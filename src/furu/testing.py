from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from .config import FURU_CONFIG, RecordGitMode
from .core import Furu
from .runtime.overrides import OverrideValue, override_furu_hashes

OverrideKey = Furu | str
_PATH_MISSING = object()


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


@contextmanager
def override_results_for(
    root: Furu,
    overrides: Mapping[str, OverrideValue],
) -> Generator[None, None, None]:
    """Override results by dotted field path relative to a root Furu object."""
    hash_overrides: dict[str, OverrideValue] = {}
    for path, value in overrides.items():
        target = _resolve_override_path(root, path)
        hash_overrides[cast(str, target.furu_hash)] = value
    with override_furu_hashes(hash_overrides):
        yield


def _resolve_override_path(root: Furu, path: str) -> Furu:
    if not path:
        raise ValueError("override path must be non-empty")
    current = root
    for segment in path.split("."):
        if not segment:
            raise ValueError(f"override path has empty segment: {path!r}")
        current = _resolve_path_segment(current, segment, path)
    if not isinstance(current, Furu):
        raise TypeError(
            f"override path {path!r} must resolve to a Furu instance; got {type(current).__name__}"
        )
    return current


def _resolve_path_segment(current: object, segment: str, path: str) -> object:
    name, keys = _parse_path_segment(segment, path)
    if name:
        value = getattr(current, name, _PATH_MISSING)
        if value is _PATH_MISSING:
            raise AttributeError(f"override path {path!r} has no attribute {name!r}")
        current = value
    for key in keys:
        index = _coerce_path_key(key)
        if isinstance(index, int):
            if not _is_indexable_sequence(current):
                raise TypeError(
                    f"override path {path!r} index {index} requires a sequence"
                )
            sequence = cast(Sequence[OverrideValue], current)
            current = sequence[index]
        else:
            if not isinstance(current, Mapping):
                raise TypeError(
                    f"override path {path!r} key {index!r} requires a mapping"
                )
            mapping = cast(Mapping[str | int, OverrideValue], current)
            current = mapping[index]
    return current


def _parse_path_segment(segment: str, path: str) -> tuple[str, list[str]]:
    name_chars: list[str] = []
    keys: list[str] = []
    index = 0
    length = len(segment)
    while index < length and segment[index] != "[":
        name_chars.append(segment[index])
        index += 1
    name = "".join(name_chars)
    while index < length:
        if segment[index] != "[":
            raise ValueError(f"override path {path!r} has invalid segment {segment!r}")
        close = segment.find("]", index + 1)
        if close == -1:
            raise ValueError(f"override path {path!r} has unterminated index")
        token = segment[index + 1 : close].strip()
        if not token:
            raise ValueError(f"override path {path!r} has empty index")
        keys.append(token)
        index = close + 1
    if not name:
        raise ValueError(f"override path {path!r} has empty segment")
    return name, keys


def _coerce_path_key(token: str) -> str | int:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    if token.startswith("-"):
        if token[1:].isdigit():
            return int(token)
        return token
    if token.isdigit():
        return int(token)
    return token


def _is_indexable_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return isinstance(value, Sequence)


@pytest.fixture()
def furu_tmp_root(tmp_path: Path) -> Generator[Path, None, None]:
    """Configure furu to use a temporary root for the test."""
    with furu_test_env(tmp_path) as root:
        yield root
