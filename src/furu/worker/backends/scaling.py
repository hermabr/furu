from __future__ import annotations

import threading
from collections.abc import Callable

from furu.logging import get_logger

logger = get_logger()


class PeriodicScaler:
    def __init__(
        self,
        *,
        interval: float,
        scale_once: Callable[[], None],
        report_failure: Callable[[str], None],
        thread_name: str,
    ) -> None:
        self._interval = interval
        self._scale_once = scale_once
        self._report_failure = report_failure
        self._thread_name = thread_name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started or self._stop.is_set():
                return
            self._started = True

        if not self._run_once():
            return
        if self._interval <= 0:
            return

        thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            self._thread = thread
        thread.start()

    def is_healthy(self) -> bool:
        with self._lock:
            failure = self._failure
            thread = self._thread
            stopped = self._stop.is_set()
        return failure is None and (thread is None or stopped or thread.is_alive())

    def stop(self, *, timeout: float) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            if not self._run_once():
                return

    def _run_once(self) -> bool:
        try:
            self._scale_once()
        except Exception as exc:
            message = (
                f"worker pool scaler failed in {self._thread_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._lock:
                self._failure = exc
            try:
                self._report_failure(message)
            except Exception:
                logger.exception(
                    "worker pool scaler failed to report failure: thread=%s",
                    self._thread_name,
                )
            logger.exception("worker pool scaler failed: thread=%s", self._thread_name)
            return False
        return True
