# TODO: this is the first iteration of this file, but it has multiple errors and footguns, such as the gil being busy for more than LeaseConfig.lifetime_s or the heartbeat thread losing the lock but not notifying/killing the worker. i will either move this to zig or rewrite it at some point
# TODO: handle cases where a user submits a 8 task torch job
import multiprocessing as mp
import os
from inspect import FrameInfo

import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Literal
import signal

from flufl.lock import AlreadyLockedError, Lock, NotLockedError, TimeOutError

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
    # stop_lease_heartbeat = threading.Event()
    # lease_lost = threading.Event()
    now = datetime.now()
    staged_result_path = result_path.with_suffix(
        f".{os.getpid()}-{now:%y%m%d_%H-%M-%S}.pkl.tmp"
    )  # TODO: compute this correctly

    try:
        file_lock = Lock(str(lock_path), lifetime=lease_config.lease_ttl_s)

        try:
            file_lock.lock(timeout=0)
        except TimeOutError as e:
            print(f"failed to lock {file_lock} at path {str(lock_path)}")
            raise NotImplementedError(
                "TODO: wait for the object to finish and return/read a copy for some time, die if you wait for too long and check if the old lease is expired (should be automatic)"
            ) from e

        if result_path.exists():
            raise NotImplementedError(
                "TODO: result exists, but got lock. maybe just read the cached value"
            )
        
    

        def review_lease_loop() -> None:
            def lease_loop_handler(signum: signal.Signals, frame: FrameInfo | None) -> None:
            
            while True:
                try:
                    file_lock.refresh()
                    time.sleep(lease_config.review_interval_s)
                except NotLockedError:
                    raise NotImplementedError("TODO: handle this")
                    # lease_lost.set()
                    # stop_lease_heartbeat.set()
                    # return  # TODO: How do i fail/throw correctly here?

        # lease_heartbeat_thread = threading.Thread(target=review_lease_loop, daemon=True)
        # lease_heartbeat_thread.start()
        ctx = mp.get_context("spawn")
        lease_heartbeat_process = ctx.Process(target=review_lease_loop, daemon=True)
        lease_heartbeat_process.start()

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
            file_lock.refresh()
        except NotLockedError:
            staged_result_path.unlink()
            return "lost-lock"

        os.replace(staged_result_path, result_path)

        return Ok(result=result)

    finally:
        stop_lease_heartbeat.set()

        # TODO: this doesn't actually kill the heartbeat thread

        if staged_result_path.exists():
            raise NotImplementedError("TODO: i don't think this can ever happen?")

        try:
            file_lock.unlock(
                unconditionally=True
            )  # TODO: this can technically fail if we didn't aquire the lock. won't fix this since this logic will be moved to zig
        except NotLockedError:  # Didn't need to unlock
            pass  # TODO: should we log that we weren't able to lock?
