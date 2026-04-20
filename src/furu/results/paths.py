from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

type PathToken = str | int

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_SEGMENT_LENGTH = 64
_MAX_SLUG_LENGTH = 40


def _stable_hash(text: str) -> str:
    return hashlib.blake2s(text.encode("utf-8"), digest_size=4).hexdigest()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    if not slug:
        return "key"
    return slug[:_MAX_SLUG_LENGTH]


def encode_path_segment(token: PathToken) -> str:
    if isinstance(token, int):
        if token < 0:
            raise ValueError("Logical path indexes must be non-negative")
        return f"{token:06d}"

    if (
        token
        and len(token) <= _MAX_SEGMENT_LENGTH
        and token not in {".", ".."}
        and "/" not in token
        and "\\" not in token
        and _SAFE_SEGMENT_RE.fullmatch(token)
    ):
        return token

    slug = _slugify(token)
    return f"{slug}--{_stable_hash(token)}"


def _format_token(token: PathToken) -> str:
    if isinstance(token, int):
        return f"[{token}]"
    return f"[{json.dumps(token, ensure_ascii=False)}]"


@dataclass(frozen=True, slots=True)
class LogicalPath:
    segments: tuple[PathToken, ...] = ()

    def child_field(self, name: str) -> "LogicalPath":
        return LogicalPath((*self.segments, name))

    def child_key(self, key: str) -> "LogicalPath":
        return LogicalPath((*self.segments, key))

    def child_index(self, index: int) -> "LogicalPath":
        return LogicalPath((*self.segments, index))

    def format(self) -> str:
        return "result" + "".join(_format_token(token) for token in self.segments)

    def artifact_relative_dir(self) -> Path:
        path = Path("artifacts")
        for token in self.segments:
            path /= encode_path_segment(token)
        return path

    def __str__(self) -> str:
        return self.format()
