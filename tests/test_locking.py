import errno
import json
import os
import time
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path
from unittest.mock import patch

import pytest

import furu.locking as locking_module
from furu.locking import LockAcquireError, LockLostError, lock, lock_many

TEST_TIMING_SCALE = 4.0 if os.environ.get("GITHUB_ACTIONS") == "true" else 1.0
TEST_CLOCK_SLOP_S = 0.02
SHORT_SLEEP_S = 0.05 * TEST_TIMING_SCALE
SHORT_LIFETIME_S = 0.05 * TEST_TIMING_SCALE
SHORT_HEARTBEAT_INTERVAL_S = 0.02 * TEST_TIMING_SCALE
PROCESS_TIMEOUT_S = 0.5 * TEST_TIMING_SCALE


@pytest.fixture(autouse=True)
def _small_clock_slop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locking_module, "CLOCK_SLOP_S", TEST_CLOCK_SLOP_S)


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _child_acquire_batch_then_exit(lock_paths: list[Path], manifest_out: str) -> None:
    with lock_many(
        lock_paths,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        Path(manifest_out).write_text(lock_paths[0].read_text(encoding="utf-8"))
        os._exit(0)


def test_lock_many_produces_hardlinks_to_one_manifest_inode(tmp_path: Path) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]

    with lock_many(
        lock_paths,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()

        manifest = _read_manifest(lock_paths[0])
        assert manifest["version"] == 2

        claim_path = Path(str(manifest["claim_path"]))
        claim_stat = claim_path.stat()
        assert manifest["lock_paths"] == [str(path.resolve()) for path in lock_paths]

        for lock_path in lock_paths:
            assert os.path.samestat(lock_path.stat(), claim_stat)


def test_lock_wrapper_delegates_to_lock_many(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[Path], float | None]] = []

    @contextmanager
    def fake_lock_many(
        lock_paths: list[Path],
        *,
        acquire_timeout_s: float | None = None,
        **_: object,
    ):
        calls.append((lock_paths, acquire_timeout_s))
        yield lambda: True

    monkeypatch.setattr(locking_module, "lock_many", fake_lock_many)

    with lock(tmp_path / "single.lock", acquire_timeout_s=1.0) as has_lock:
        assert has_lock()

    assert calls == [([tmp_path / "single.lock"], 1.0)]


def test_has_lock_returns_false_when_any_batch_link_is_lost(tmp_path: Path) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]

    with pytest.raises(LockLostError, match="lost lock"):
        with lock_many(
            lock_paths,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as has_lock:
            lock_paths[1].unlink()
            assert not has_lock()


def test_stale_break_from_any_member_path_removes_whole_logical_lock_group(
    tmp_path: Path,
) -> None:
    lock_paths = [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"]
    manifest_out = tmp_path / "manifest.json"
    proc = Process(
        target=_child_acquire_batch_then_exit,
        args=(lock_paths, str(manifest_out)),
    )
    proc.start()
    proc.join(timeout=PROCESS_TIMEOUT_S)
    assert proc.exitcode == 0

    stale_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    stale_claim_path = Path(str(stale_manifest["claim_path"]))
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(stale_claim_path, (stale_time, stale_time))

    with lock(
        lock_paths[1],
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()
        assert not lock_paths[0].exists()
        assert not lock_paths[2].exists()
        assert not stale_claim_path.exists()


def test_cross_filesystem_lock_requests_raise_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_parent = tmp_path / "one"
    second_parent = tmp_path / "two"
    first_parent.mkdir()
    second_parent.mkdir()
    lock_paths = [first_parent / "a.lock", second_parent / "b.lock"]
    real_stat = Path.stat

    class _FakeStat:
        def __init__(self, stat_result: os.stat_result, *, st_dev: int):
            self._stat_result = stat_result
            self.st_dev = st_dev

        def __getattr__(self, name: str):
            return getattr(self._stat_result, name)

    def fake_stat(path: Path, *args, **kwargs):
        stat_result = real_stat(path, *args, **kwargs)
        if path == lock_paths[1].parent:
            return _FakeStat(stat_result, st_dev=stat_result.st_dev + 1)
        return stat_result

    monkeypatch.setattr(type(lock_paths[0].parent), "stat", fake_stat)

    with pytest.raises(LockAcquireError, match="same filesystem device"):
        with lock_many(lock_paths):
            pass


def test_partial_acquire_rollback_releases_subset_under_overlap(tmp_path: Path) -> None:
    first_lock = tmp_path / "a.lock"
    blocked_lock = tmp_path / "b.lock"

    with lock(
        blocked_lock,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        with pytest.raises(LockAcquireError):
            with lock_many(
                [first_lock, blocked_lock],
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ):
                pass

    assert not first_lock.exists()


def test_stale_break_refuses_malformed_manifest(tmp_path: Path) -> None:
    lock_path = tmp_path / "broken.lock"
    lock_path.write_text("not json", encoding="utf-8")
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale_time, stale_time))

    with pytest.raises(LockAcquireError, match="malformed lock manifest"):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            pass

    assert lock_path.read_text(encoding="utf-8") == "not json"


def test_lock_retries_once_after_benign_link_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lck"
    original_link = os.link
    call_count = 0

    def flaky_link(src, dst, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError(errno.ENOENT, "simulated benign race")
        return original_link(src, dst, *args, **kwargs)

    with patch("os.link", side_effect=flaky_link):
        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            assert lock_path.exists()

    assert call_count == 2
