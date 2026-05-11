from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from furu.core import Furu
    from furu.server import SchedulerClient


T = TypeVar("T")


class SubmissionFailed(RuntimeError):
    pass


class SubmissionStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    submission_id: str
    status: Literal["running", "done", "failed"]
    failure_message: str | None = None
    total_nodes: int
    done_nodes: int
    queued_nodes: int
    running_nodes: int


class Submission(Generic[T]):
    id: str

    def __init__(
        self,
        *,
        submission_id: str,
        roots: tuple[Furu[T], ...],
        single_input: bool,
        client: SchedulerClient,
        on_done: Callable[[], None] | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.id = submission_id
        self._roots = roots
        self._single_input = single_input
        self._client = client
        self._on_done = on_done
        self._poll_interval_s = poll_interval_s
        self._terminal = False

    def status(self) -> SubmissionStatus:
        return self._client.get_submission_status(self.id)

    def result(self, *, timeout_s: float | None = None) -> T:
        from typing import cast

        from furu.migration import result_dir_for_loading
        from furu.result import load_result_bundle

        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        while True:
            status = self._client.get_submission_status(self.id)

            if status.status == "failed":
                self._finalize()
                raise SubmissionFailed(status.failure_message or "submission failed")

            if status.status == "done":
                self._finalize()
                break

            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("submission did not finish before timeout")

            time.sleep(self._poll_interval_s)

        values: list[object] = []
        for root in self._roots:
            result_dir = result_dir_for_loading(root)
            if result_dir is None:
                raise RuntimeError(
                    f"Submission marked done but root result is missing: {root.data_dir}"
                )
            values.append(load_result_bundle(result_dir))

        if self._single_input:
            return cast(T, values[0])

        return cast(T, values)

    def _finalize(self) -> None:
        if self._terminal:
            return
        self._terminal = True
        if self._on_done is not None:
            self._on_done()
