# TODO: this is the first iteration of this file, but it has multiple errors and footguns, such as the gil being busy for more than LeaseConfig.lifetime_s or the heartbeat thread losing the lock but not notifying/killing the worker. i will either move this to zig or rewrite it at some point
# TODO: handle cases where a user submits a 8 task torch job
import os
import pickle
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from furu.locking_manager import NotLockedError, TimeOutError, lock

from furu.utils import Ok

type LeaseErrorStates = Literal["lost-lock", "missing-tmp"]


@dataclass(frozen=True, slots=True)
class LeaseConfig:
    lease_ttl_s: int = 120
    review_interval_s: int = 15


def _compute_and_stage_pickle_result[T](
    *,
    staged_result_path: Path,
    compute_fn: Callable[[], T],
) -> T:
    result = compute_fn()
    with staged_result_path.open("wb") as f:
        pickle.dump(
            result,
            f,
        )
        f.flush()  # TODO: Do i need this and the os.fsync?
        os.fsync(f.fileno())
    return result


def run_with_lease_and_pickle_result[T](
    # data_dir: Path,
    compute_fn: Callable[[], T],
    *,
    lock_path: Path,
    result_path: Path,
    lease_config: LeaseConfig | None = None,
    # TODO: add a n-retries option
) -> Ok | LeaseErrorStates:
    if lease_config is None:
        lease_config = LeaseConfig()
    # TODO: make the directory for the data and the <data-dir>/.furu
    # TODO: if the gil is busy for more than lease_config.lifetime_s we will lose the lock. move to zig or use multiprocessing. the reason we are not already using multiprocessing is that it is more painful to return the result from the self._create call when we use multiprocessing
    stop_lease_heartbeat = threading.Event()
    lease_lost = threading.Event()
    now = datetime.now()
    staged_result_path = result_path.with_suffix(
        f".{os.getpid()}-{now:%y%m%d_%H-%M-%S}.pkl.tmp"
    )  # TODO: compute this correctly

    try:
        with lock(
            lock_path,
            lifetime_s=lease_config.lease_ttl_s,
            timeout_s=0,
        ) as refresh_lock:

            def review_lease_loop() -> None:
                while not stop_lease_heartbeat.wait(lease_config.review_interval_s):
                    try:
                        refresh_lock()
                    except NotLockedError:
                        lease_lost.set()
                        stop_lease_heartbeat.set()
                        return  # TODO: How do i fail/throw correctly here?

            lease_heartbeat_thread = threading.Thread(
                target=review_lease_loop, daemon=True
            )
            lease_heartbeat_thread.start()

            try:
                result = _compute_and_stage_pickle_result(
                    staged_result_path=staged_result_path, compute_fn=compute_fn
                )
            except BaseException:
                staged_result_path.unlink(missing_ok=True)
                # TODO: maybe handle BaseException and Exception differently?
                raise

            if not staged_result_path.exists():
                return "missing-tmp"
            elif lease_lost.is_set():
                staged_result_path.unlink()
                return "lost-lock"

            try:
                refresh_lock()
            except NotLockedError:
                staged_result_path.unlink()
                return "lost-lock"

            os.replace(staged_result_path, result_path)

            return Ok(result=result)
    except TimeOutError as e:
        print(f"failed to lock at path {str(lock_path)}")
        raise NotLockedError("failed to acquire compute lock") from e

    # TODO: I used to ahve this finally block, but i no longer have that
    # finally:
    #     stop_lease_heartbeat.set()
    #     # TODO: this doesn't actually kill the heartbeat thread
    #     if staged_result_path.exists():
    #         raise NotImplementedError("TODO: i don't think this can ever happen?")
