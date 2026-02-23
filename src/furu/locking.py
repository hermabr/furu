# TODO: this is the first iteration of this file, but it has multiple errors and footguns, such as the gil being busy for more than LeaseConfig.lifetime_s or the heartbeat thread losing the lock but not notifying/killing the worker. i will either move this to zig or rewrite it at some point
import os
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from flufl.lock import Lock, NotLockedError, TimeOutError

from furu.utils import Ok

type LeaseErrorStates = Literal["lost-lock", "worker-failed", "missing-tmp"]


@dataclass(frozen=True, slots=True)
class LeaseConfig:
    lease_ttl_s: int = 120
    review_interval_s: int = 15


def _compute_and_stage_pickle_result[T](
    *,
    staged_result_path: Path,
    compute_fn: Callable[[], T],
    worker_result: Future[T],
):
    try:
        result = compute_fn()
        with staged_result_path.open("wb") as f:
            pickle.dump(
                result,
                f,
            )
            f.flush()  # TODO: Do i need this and the os.fsync?
            os.fsync(f.fileno())
    except BaseException as e:
        worker_result.set_exception(e)
    else:
        worker_result.set_result(result)


def run_with_lease_and_pickle_result[T](
    # data_dir: Path,
    compute_fn: Callable[[], T],
    *,
    lock_path: Path,
    result_path: Path,
    lease_config: LeaseConfig = LeaseConfig(),
    poll_interval_s: float = 0.5,
    # TODO: add a n-retries option
) -> Ok | LeaseErrorStates:
    # TODO: make the directory for the data and the <data-dir>/.furu
    # TODO: if the gil is busy for more than lease_config.lifetime_s we will lose the lock. move to zig or use multiprocessing. the reason we are not already using multiprocessing is that it is more painful to return the result from the self._create call when we use multiprocessing
    file_lock = Lock(str(lock_path), lifetime=lease_config.lease_ttl_s)

    stop_lease_heartbeat = threading.Event()
    lease_lost = threading.Event()

    try:
        file_lock.lock(timeout=0)  # TODO:make sure timeout 0 makes sense
    except TimeOutError:
        raise NotImplementedError(
            "TODO: wait for the object to finish and return/read a copy for some time, die if you wait for too long and check if the old lease is expired (should be automatic)"
        )

    def review_lease_loop() -> None:
        while not stop_lease_heartbeat.wait(lease_config.review_interval_s):
            try:
                file_lock.refresh()
            except NotLockedError:
                lease_lost.set()
                stop_lease_heartbeat.set()
                return  # TODO: How do i fail/throw correctly here?

    lease_heartbeat_thread = threading.Thread(target=review_lease_loop, daemon=True)
    lease_heartbeat_thread.start()

    now = datetime.now()
    staged_result_path = result_path.with_suffix(
        f".{os.getpid()}-{now:%y%m%d_%H-%M-%S}.pkl.tmp"
    )  # TODO: compute this correctly

    worker_result: Future[T] = Future()
    worker_thread = threading.Thread(
        target=_compute_and_stage_pickle_result,
        kwargs={
            "staged_result_path": staged_result_path,
            "compute_fn": compute_fn,
            "worker_result": worker_result,
        },
        name=f"task:{result_path.parent.name}",
        daemon=True,
    )
    worker_thread.start()

    try:
        while not worker_result.done():
            if lease_lost.is_set():
                worker_thread.join()  # This is sad since we don't actually kill the thread, but simply wait for it to finish
                staged_result_path.unlink(missing_ok=True)
                return "lost-lock"
            time.sleep(poll_interval_s)

        try:
            result = worker_result.result()
        except BaseException as e:
            staged_result_path.unlink(missing_ok=True)
            if isinstance(e, Exception):
                return "worker-failed"
            raise e

        if not staged_result_path.exists():
            return "missing-tmp"
        elif lease_lost.is_set():
            staged_result_path.unlink()
            return "lost-lock"

        try:
            file_lock.refresh()
        except NotLockedError:
            staged_result_path.unlink()
            return "lost-lock"

        os.replace(staged_result_path, result_path)

        return Ok(result=result)

    finally:
        stop_lease_heartbeat.set()
        lease_heartbeat_thread.join()  # TODO: this doesn't actually kill the heartbeat thread

        if staged_result_path.exists():
            raise NotImplementedError("TODO: i don't think this can ever happen?")

        file_lock.unlock(unconditionally=True)
