from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

type PathToken = str | int

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")


@dataclass(frozen=True, slots=True)
class LogicalPath:
    parts: tuple[PathToken, ...] = ()

    def child(self, token: PathToken) -> "LogicalPath":
        return LogicalPath(parts=(*self.parts, token))

    def display(self) -> str:
        pieces = ["result"]
        for token in self.parts:
            if isinstance(token, str):
                pieces.append(f"[{json.dumps(token)}]")
            else:
                pieces.append(f"[{token}]")
        return "".join(pieces)


def encode_artifact_dir(path: LogicalPath) -> str:
    segments = [_encode_token(token) for token in path.parts]
    if not segments:
        segments = ["root"]
    return Path("artifacts", *segments).as_posix()


def _encode_token(token: PathToken) -> str:
    if isinstance(token, int):
        if token >= 0:
            return f"{token:06d}"
        return f"neg{abs(token):06d}"

    if _SAFE_SEGMENT.fullmatch(token):
        return token

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", token).strip("-.")
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=6).hexdigest()
    if normalized:
        return f"{normalized[:24]}-{digest}"
    return f"key-{digest}"
