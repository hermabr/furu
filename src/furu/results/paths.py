from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Final

type LogicalToken = str | int
type LogicalPath = tuple[LogicalToken, ...]

_SAFE_SEGMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$"
)
_MAX_SLUG_LENGTH: Final[int] = 32


def format_logical_path(path: LogicalPath) -> str:
    rendered = "result"
    for token in path:
        if isinstance(token, int):
            rendered += f"[{token}]"
        else:
            rendered += f".{token}"
    return rendered


def artifact_relpath_for_logical_path(path: LogicalPath) -> PurePosixPath:
    if not path:
        return PurePosixPath("artifacts", "__root__")
    return PurePosixPath("artifacts", *[_encode_segment(token) for token in path])


def artifact_path_for_logical_path(bundle_dir: Path, path: LogicalPath) -> Path:
    return bundle_dir / artifact_relpath_for_logical_path(path)


def artifact_relpath_str(path: LogicalPath) -> str:
    return artifact_relpath_for_logical_path(path).as_posix()


def _encode_segment(token: LogicalToken) -> str:
    if isinstance(token, int):
        if token < 0:
            raise ValueError(f"Negative logical index {token} is not supported")
        return f"@{token:06d}"

    if _SAFE_SEGMENT_RE.fullmatch(token):
        return token

    slug = _slugify(token)
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).hexdigest()
    return f"@key-{slug}-{digest}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    if not slug:
        slug = "key"
    return slug[:_MAX_SLUG_LENGTH]
