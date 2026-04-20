from __future__ import annotations

import pickle
from pathlib import Path

from furu.utils import JsonValue


class PickleCodec:
    codec_id = "python.pickle.v1"

    def dump(
        self,
        value: object,
        artifact_dir: Path,
        ctx: object,
    ) -> dict[str, JsonValue] | None:
        del ctx
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "data.pkl"
        with path.open("wb") as f:
            pickle.dump(value, f)
        return None

    def load(
        self,
        artifact_dir: Path,
        meta: dict[str, JsonValue] | None,
        ctx: object,
    ) -> object:
        del meta, ctx
        path = artifact_dir / "data.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing pickle artifact at {path}")
        with path.open("rb") as f:
            return pickle.load(f)
