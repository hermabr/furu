from typing import Literal

from pydantic import BaseModel, ConfigDict


class State(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)


class SuccessState(State):
    status: Literal["success"] = "success"


class FailedState(State):
    status: Literal["failed"] = "failed"


class RunningState(State):
    status: Literal["running"] = "running"
