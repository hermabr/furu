from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from furu.core import Furu


class NodeKey(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    object_id: str
    data_path: str

    @classmethod
    def from_furu(cls, obj: Furu[Any]) -> NodeKey:
        return cls(
            object_id=obj.object_id,
            data_path=str(obj.data_dir.resolve(strict=False)),
        )
