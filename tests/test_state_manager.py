import huldra


def test_state_read_write_and_staleness(huldra_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()

    state0 = huldra.StateManager.read_state(directory)
    assert state0["status"] == "missing"

    huldra.StateManager.write_state(directory, status="queued", foo=1)
    state1 = huldra.StateManager.read_state(directory)
    assert state1["status"] == "queued"
    assert state1["foo"] == 1
    assert "updated_at" in state1

    assert huldra.StateManager.is_stale(directory, timeout=60) is False


def test_locks_are_exclusive(huldra_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()
    lock_path = directory / huldra.StateManager.COMPUTE_LOCK

    fd1 = huldra.StateManager.try_lock(lock_path)
    assert fd1 is not None
    assert huldra.StateManager.try_lock(lock_path) is None

    huldra.StateManager.release_lock(fd1, lock_path)
    assert lock_path.exists() is False


def test_leases_can_be_renewed_and_reconciled(huldra_tmp_root, tmp_path) -> None:
    directory = tmp_path / "obj"
    directory.mkdir()

    huldra.StateManager.write_state(
        directory,
        status="running",
        owner_id="a",
        **huldra.StateManager.new_lease(lease_duration_sec=0.01),
    )

    assert huldra.StateManager.renew_lease(
        directory, owner_id="b", lease_duration_sec=0.01
    ) is False
    assert huldra.StateManager.renew_lease(
        directory, owner_id="a", lease_duration_sec=0.01
    ) is True

    # Let the lease expire and reconcile.
    import time

    time.sleep(0.02)
    assert huldra.StateManager.reconcile_expired_running(directory) is True
    assert huldra.StateManager.read_state(directory).get("status") == "cancelled"
