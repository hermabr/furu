from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict

from furu.core import Furu
from furu.migration import result_dir_for_loading
from furu.result import load_result_bundle


class SubmissionStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    submission_id: str
    status: Literal["running", "done", "failed"]
    failure_message: str | None = None
    total_nodes: int
    done_nodes: int
    queued_nodes: int
    running_nodes: int


class SubmissionFailed(RuntimeError):
    pass


class _SubmissionClient(Protocol):
    def get_submission_status(self, submission_id: str) -> SubmissionStatus: ...


@dataclass
class Submission[T]:
    id: str
    _client: _SubmissionClient
    _roots: tuple[Furu[Any], ...]
    _single_input: bool
    _poll_interval_s: float = 0.05
    _on_terminal: Callable[[], None] | None = None
    _terminal_callback_ran: bool = field(default=False, init=False)

    def status(self) -> SubmissionStatus:
        return self._client.get_submission_status(self.id)

    def result(self, *, timeout_s: float | None = None) -> T:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        while True:
            status = self.status()

            if status.status == "failed":
                self._run_terminal_callback()
                raise SubmissionFailed(status.failure_message or "submission failed")

            if status.status == "done":
                self._run_terminal_callback()
                break

            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("submission did not finish before timeout")

            time.sleep(self._poll_interval_s)

        values: list[Any] = []
        for root in self._roots:
            result_dir = result_dir_for_loading(root)
            if result_dir is None:
                raise RuntimeError(
                    "Submission marked done but root result is missing: "
                    f"{root.data_dir}"
                )
            values.append(load_result_bundle(result_dir))

        if self._single_input:
            return cast(T, values[0])
        return cast(T, values)

    def _run_terminal_callback(self) -> None:
        if self._terminal_callback_ran or self._on_terminal is None:
            return
        self._terminal_callback_ran = True
        self._on_terminal()
