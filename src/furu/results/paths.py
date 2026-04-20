from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

type PathToken = str | int

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class LogicalPath:
    tokens: tuple[PathToken, ...] = ()

    def child(self, token: PathToken) -> "LogicalPath":
        return LogicalPath(self.tokens + (token,))

    def display(self) -> str:
        parts = ["result"]
        for token in self.tokens:
            if isinstance(token, str):
                parts.append(f"[{json.dumps(token)}]")
            else:
                parts.append(f"[{token}]")
        return "".join(parts)


def artifact_relpath_for(path: LogicalPath) -> str:
    if not path.tokens:
        return "artifacts/root"
    parts = ["artifacts"]
    parts.extend(_artifact_segment_for_token(token) for token in path.tokens)
    return "/".join(parts)


def _artifact_segment_for_token(token: PathToken) -> str:
    if isinstance(token, int):
        if token >= 0:
            return f"{token:06d}"
        return f"neg{abs(token):06d}"

    if _is_safe_segment(token):
        return token

    slug = re.sub(r"[^A-Za-z0-9]+", "-", token).strip("-").lower()[:24] or "item"
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=5).hexdigest()
    return f"{slug}-{digest}"


def _is_safe_segment(token: str) -> bool:
    if token in {".", ".."}:
        return False
    if "/" in token or "\\" in token:
        return False
    return _SAFE_SEGMENT_RE.fullmatch(token) is not None
