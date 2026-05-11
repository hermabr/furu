from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from furu.core import Furu
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle
from furu.server import SchedulerClient, SubmissionStatusResponse


class SubmissionFailed(RuntimeError):
    pass


type SubmissionStatus = SubmissionStatusResponse


@dataclass(frozen=True)
class Submission[T]:
    id: str
    _client: SchedulerClient
    _roots: Sequence[Furu[Any]]
    _single_input: bool
    _poll_interval_s: float = 0.05
    _on_terminal: Callable[[], None] | None = None

    def status(self) -> SubmissionStatus:
        return self._client.get_submission_status(self.id)

    def result(self, *, timeout_s: float | None = None) -> T:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        try:
            while True:
                status = self.status()
                if status.status == "failed":
                    raise SubmissionFailed(
                        status.failure_message or "submission failed"
                    )
                if status.status == "done":
                    break
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("submission did not finish before timeout")
                time.sleep(self._poll_interval_s)

            values = []
            for root in self._roots:
                result_dir = result_dir_for_loading(root)
                if result_dir is None:
                    raise RuntimeError(
                        "Submission marked done but root result is missing: "
                        f"{root.data_dir}"
                    )
                values.append(load_result_bundle(result_dir))
            return cast(T, values[0] if self._single_input else values)
        finally:
            if self._on_terminal is not None:
                self._on_terminal()
