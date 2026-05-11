from __future__ import annotations

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
    def from_furu[TFuru: Furu](cls, obj: TFuru) -> NodeKey:
        return cls(
            object_id=obj.object_id,
            data_path=str(obj.data_dir.resolve(strict=False)),
        )
