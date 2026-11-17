import errno
import json
import logging
import os
import threading
import time
from contextlib import ExitStack
from multiprocessing import Process
from pathlib import Path
from unittest.mock import patch

import pytest

import furu.locking as locking_module
from furu.locking import (
    LockError,
    LockManifest,
    lock,
)

TEST_CLOCK_SLOP_S = 0.02
SHORT_SLEEP_S = 0.05
SHORT_LIFETIME_S = 0.05
SHORT_HEARTBEAT_INTERVAL_S = 0.02
# Only bounds a hung test; passing runs never wait for it, so keep it generous.
TIMEOUT_S = 10.0


@pytest.fixture(autouse=True)
def _small_clock_slop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locking_module, "CLOCK_SLOP_S", TEST_CLOCK_SLOP_S)


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_claim_path(path: Path) -> Path:
    return Path(str(_read_manifest(path)["claim_path"]))


def _child_acquire_then_exit(
    lock_path: Path, owner_path: str, *, lifetime_s: float = SHORT_LIFETIME_S
) -> None:
    with lock(
        lock_path,
        lifetime_s=lifetime_s,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ):
        Path(owner_path).write_text(
            str(_manifest_claim_path(lock_path)), encoding="utf-8"
        )
        os._exit(0)


def _drop_current_lock(lock_path: Path) -> None:
    claim_path = _manifest_claim_path(lock_path)
    lock_path.unlink()
    claim_path.unlink()


def test_lock_is_a_hardlink_to_its_claim_manifest(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with lock(lock_path, acquire_timeout_s=1.0) as has_lock:
        assert has_lock()

        manifest = _read_manifest(lock_path)
        assert manifest["lock_path"] == str(lock_path.resolve())
        claim_path = Path(str(manifest["claim_path"]))
        assert os.path.samestat(lock_path.stat(), claim_path.stat())

    assert not lock_path.exists()


def test_lock_uses_default_arguments(tmp_path: Path) -> None:
    with lock(tmp_path / "single.lock") as has_lock:
        assert has_lock()


def test_has_lock_returns_false_when_lock_is_lost(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with (
        pytest.raises(LockError, match="lost lock"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ) as has_lock,
    ):
        _drop_current_lock(lock_path)
        assert not has_lock()


def test_exit_raises_lock_lost_error_when_lock_is_lost_mid_block(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"

    with (
        pytest.raises(LockError, match="lost lock"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        _drop_current_lock(lock_path)
        time.sleep(SHORT_LIFETIME_S)


def test_refresh_extends_expiration(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    # Long enough that a starved heartbeat can't let the lock go stale.
    lifetime_s = TIMEOUT_S

    with lock(
        lock_path,
        lifetime_s=lifetime_s,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()

        initial_expiration = lock_path.stat().st_mtime
        refresh_deadline = time.monotonic() + TIMEOUT_S
        while lock_path.stat().st_mtime <= initial_expiration:
            assert time.monotonic() < refresh_deadline, "lock was not refreshed"
            time.sleep(SHORT_HEARTBEAT_INTERVAL_S / 2)

        with (
            pytest.raises(LockError, match="could not acquire lock"),
            lock(
                lock_path,
                lifetime_s=lifetime_s,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ),
        ):
            pass


def test_lock_starts_heartbeat_thread(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with lock(lock_path):
        assert [
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("lock-heartbeat:")
        ] == [f"lock-heartbeat:{lock_path.name}"]
    assert not any(
        thread.name.startswith("lock-heartbeat:") for thread in threading.enumerate()
    )


def test_timeout_when_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    # The lock is only files on disk, so a holder in this process is as real as
    # one in another; the default lifetime keeps it fresh throughout.
    with lock(lock_path):
        started_at = time.monotonic()
        with (
            pytest.raises(LockError, match="could not acquire lock"),
            lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ),
        ):
            pass
        assert time.monotonic() - started_at >= SHORT_LIFETIME_S


def test_waits_for_lock_release_before_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock_path = tmp_path / "single.lock"
    holder = ExitStack()
    holder.enter_context(lock(lock_path))
    real_sleep = time.sleep
    waits: list[float] = []

    def release_holder_then_sleep(seconds: float) -> None:
        # The waiter only sleeps after finding the lock held.
        waits.append(seconds)
        holder.close()
        real_sleep(seconds)

    monkeypatch.setattr(locking_module.time, "sleep", release_holder_then_sleep)
    with lock(
        lock_path,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        acquire_timeout_s=TIMEOUT_S,
        acquire_poll_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()
    assert waits


def test_lock_logs_when_waiting_for_lock(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
    furu_logger = logging.getLogger("furu")
    furu_logger.addHandler(caplog.handler)
    monkeypatch.setattr(
        locking_module,
        "DEFAULT_LOCK_WAIT_LOG_INTERVAL_S",
        SHORT_HEARTBEAT_INTERVAL_S / 2,
    )

    try:
        caplog.set_level(logging.INFO, logger="furu")
        with (
            lock(lock_path),
            pytest.raises(LockError, match="could not acquire lock"),
            lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=SHORT_SLEEP_S * 4,
                acquire_poll_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            ),
        ):
            pass
    finally:
        furu_logger.removeHandler(caplog.handler)

    wait_message = f"waiting for lock at {lock_path.resolve()}"
    assert caplog.messages.count(wait_message) >= 2


def test_stale_break_clears_a_lock_whose_claim_file_is_missing(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
    owner_path = tmp_path / "owner"
    proc = Process(
        daemon=True,
        target=_child_acquire_then_exit,
        args=(lock_path, str(owner_path)),
    )
    proc.start()
    proc.join(timeout=TIMEOUT_S)
    assert proc.exitcode == 0

    Path(owner_path.read_text(encoding="utf-8")).unlink()
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale_time, stale_time))

    with lock(
        lock_path,
        lifetime_s=SHORT_LIFETIME_S,
        heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
    ) as has_lock:
        assert has_lock()


def test_release_removes_the_lock_when_its_claim_file_is_missing(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"

    with (
        pytest.raises(LockError, match="lost lock"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        _manifest_claim_path(lock_path).unlink()

    assert not lock_path.exists()


def test_lock_raises_if_stale_break_would_remove_reacquired_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
    first_claim_path = tmp_path / "first.claim"
    second_claim_path = tmp_path / "second.claim"
    manifest = LockManifest(
        claim_path=first_claim_path.resolve(),
        lock_path=lock_path.resolve(),
    )

    locking_module._write_claim_file(manifest, lifetime_s=SHORT_LIFETIME_S)
    assert locking_module.try_link(
        lock_path=lock_path,
        claim_path=first_claim_path.resolve(),
    )

    stale = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale, stale))

    original_acquire_break_file = locking_module.acquire_break_file

    def reacquire_before_break(*, claim_path: Path, lifetime_s: float) -> Path | None:
        break_path = original_acquire_break_file(
            claim_path=claim_path,
            lifetime_s=lifetime_s,
        )
        if break_path is None:
            return None

        lock_path.unlink()
        second_manifest = LockManifest(
            claim_path=second_claim_path.resolve(),
            lock_path=lock_path.resolve(),
        )
        locking_module._write_claim_file(second_manifest, lifetime_s=lifetime_s)
        assert locking_module.try_link(
            lock_path=lock_path,
            claim_path=second_claim_path.resolve(),
        )
        os.utime(lock_path, (stale, stale))
        return break_path

    with (
        patch.object(
            locking_module,
            "acquire_break_file",
            side_effect=reacquire_before_break,
        ),
        pytest.raises(LockError, match="changed owners while breaking a stale lock"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        pass

    assert locking_module.owns(lock_path, claim_path=second_claim_path.resolve())


def test_does_not_break_lock_within_clock_slop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock_path = tmp_path / "single.lock"
    # A wide slop, so the lock stays "almost stale" however slow this host is.
    monkeypatch.setattr(locking_module, "CLOCK_SLOP_S", 2 * TIMEOUT_S)

    with lock(lock_path):
        almost_stale = time.time() - locking_module.CLOCK_SLOP_S + TIMEOUT_S
        os.utime(lock_path, (almost_stale, almost_stale))

        with (
            pytest.raises(LockError, match="could not acquire lock"),
            lock(
                lock_path,
                lifetime_s=SHORT_LIFETIME_S,
                heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
                acquire_timeout_s=0.0,
            ),
        ):
            pass


def test_stale_break_refuses_malformed_manifest(tmp_path: Path) -> None:
    lock_path = tmp_path / "broken.lock"
    lock_path.write_text("not json", encoding="utf-8")
    stale_time = time.time() - locking_module.CLOCK_SLOP_S - SHORT_SLEEP_S
    os.utime(lock_path, (stale_time, stale_time))

    with (
        pytest.raises(LockError, match="malformed lock manifest"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
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

    with (
        patch("os.link", side_effect=flaky_link),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        assert lock_path.exists()

    assert call_count == 2


def test_lock_raises_unexpected_link_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"

    with (
        patch("os.link", side_effect=OSError(errno.EINVAL, "bad link")),
        pytest.raises(OSError, match="bad link"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        pass


def test_release_raises_lock_lost_for_estale_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    original_unlink = os.unlink
    call_count = 0

    def flaky_unlink(path, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError(errno.ESTALE, "stale unlink")
        return original_unlink(path, *args, **kwargs)

    with (
        patch("os.unlink", side_effect=flaky_unlink),
        pytest.raises(LockError, match="lost lock"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        assert lock_path.exists()


def test_release_raises_unexpected_unlink_error(tmp_path: Path) -> None:
    lock_path = tmp_path / "single.lock"
    original_unlink = os.unlink

    def bad_unlink(path, *args, **kwargs):
        if Path(os.fsdecode(path)) == lock_path:
            raise OSError(errno.EINVAL, "bad unlink")
        return original_unlink(path, *args, **kwargs)

    with (
        patch("os.unlink", side_effect=bad_unlink),
        pytest.raises(OSError, match="bad unlink"),
        lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ),
    ):
        pass


def test_process_exit_without_cleanup_allows_reclaim_after_expiry(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "single.lock"
    owner_path = tmp_path / "owner.txt"
    proc = Process(
        daemon=True,
        target=_child_acquire_then_exit,
        args=(lock_path, str(owner_path)),
        kwargs={"lifetime_s": SHORT_LIFETIME_S},
    )
    proc.start()

    try:
        proc.join(timeout=TIMEOUT_S)
        assert proc.exitcode == 0
        first_owner = owner_path.read_text(encoding="utf-8").strip()
        # The dead owner's lock is left behind until it expires.
        assert str(_manifest_claim_path(lock_path)) == first_owner

        with lock(
            lock_path,
            lifetime_s=SHORT_LIFETIME_S,
            heartbeat_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
            acquire_timeout_s=TIMEOUT_S,
            acquire_poll_interval_s=SHORT_HEARTBEAT_INTERVAL_S,
        ):
            second_owner = str(_manifest_claim_path(lock_path))
            assert second_owner != first_owner
    finally:
        proc.join(timeout=TIMEOUT_S)
