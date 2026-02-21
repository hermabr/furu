from typing import Literal

from pydantic import BaseModel


class State(BaseModel):
    pass


class SuccessState(State):
    status: Literal["success"] = "success"


class FailedState(State):
    status: Literal["failed"] = "failed"
