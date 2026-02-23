# TODO: this is the first iteration of this file, but it has multiple errors and footguns, such as the gil being busy for more than LeaseConfig.lifetime_s or the heartbeat thread losing the lock but not notifying/killing the worker. i will either move this to zig or rewrite it at some point

import os
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from flufl.lock import Lock, NotLockedError, TimeOutError


@dataclass(frozen=True, slots=True, kw_only=True)
class Success[T]:
    result: T


type ErrorStates = Literal["lost-lock", "worker-failed", "missing-tmp"]


@dataclass(frozen=True, slots=True)
class LeaseConfig:
    lifetime_s: int = 120
    refresh_every_s: int = 15


def _run_compute_fn_and_save[T](
    *,
    tmp_path: Path,
    compute_fn: Callable[[], T],
    outcome: Future[T],
):
    try:
        result = compute_fn()
        with tmp_path.open("wb") as f:
            pickle.dump(
                result,
                f,
            )
            f.flush()  # TODO: Do i need this and the os.fsync?
            os.fsync(f.fileno())
    except BaseException as e:
        outcome.set_exception(e)
    else:
        outcome.set_result(result)


def run_safely[T](
    data_dir: Path,
    lease_config: LeaseConfig,
    compute_fn: Callable[[], T],
    recheck_compute_finished_every_s: float = 0.5,
    # TODO: add a n-retries option
) -> ErrorStates | Success:
    # TODO: make the directory for the data and the <data-dir>/.furu
    # TODO: if the gil is busy for more than lease_config.lifetime_s we will lose the lock. move to zig or use multiprocessing. the reason we are not already using multiprocessing is that it is more painful to return the result from the self._create call when we use multiprocessing
    lock = Lock(
        str(data_dir / "compute-lock"), lifetime=lease_config.lifetime_s
    )  # TODO: use the correct compute lock path: <data-dir>/.furu/compute.lock

    stop_heartbeat = threading.Event()
    lost = threading.Event()

    try:
        lock.lock(timeout=0)  # TODO:
    except TimeOutError:
        raise NotImplementedError("TODO: check if the old lease is expired")

    def heartbeat() -> None:
        while not stop_heartbeat.wait(lease_config.refresh_every_s):
            try:
                lock.refresh()
            except NotLockedError:
                lost.set()
                stop_heartbeat.set()
                return  # TODO: How do i fail/throw correctly here?

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    result_path = data_dir / "result.pkl"  # TODO: compute this correctly
    tmp_result_path = (
        data_dir / f"result.{os.getpid()}.{time.time_ns()}.pkl.tmp"
    )  # TODO: compute this correctly

    outcome: Future[T] = Future()
    worker_thread = threading.Thread(
        target=_run_compute_fn_and_save,
        kwargs={
            "tmp_path": tmp_result_path,
            "compute_fn": compute_fn,
            "outcome": outcome,
        },
        name=f"task:{data_dir.name}",
        daemon=True,
    )
    worker_thread.start()

    try:
        while not outcome.done():
            if lost.is_set():
                worker_thread.join()  # This is sad since we don't actually kill the thread, but simply wait for it to finish
                tmp_result_path.unlink(missing_ok=True)
                return "lost-lock"
            time.sleep(recheck_compute_finished_every_s)

        try:
            result = outcome.result()
        except BaseException as e:
            tmp_result_path.unlink(missing_ok=True)
            if isinstance(e, Exception):
                return "worker-failed"
            raise e

        if not tmp_result_path.exists():
            return "missing-tmp"
        elif lost.is_set():
            tmp_result_path.unlink()
            return "lost-lock"

        try:
            lock.refresh()
        except NotLockedError:
            tmp_result_path.unlink()
            return "lost-lock"

        os.replace(tmp_result_path, result_path)

        return Success(result=result)

    finally:
        stop_heartbeat.set()
        heartbeat_thread.join()  # TODO: this doesn't actually kill the heartbeat thread

        if tmp_result_path.exists():
            raise NotImplementedError("TODO: i don't think this can ever happen?")

        lock.unlock(unconditionally=True)
