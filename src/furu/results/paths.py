from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SEGMENT_LENGTH = 48
_ROOT_SEGMENT = "__root__"
_KEY_SUFFIX = "__key__"


@dataclass(frozen=True, slots=True)
class MappingKeyToken:
    display: str
    fingerprint: str


type PathToken = str | int | MappingKeyToken
type LogicalPath = tuple[PathToken, ...]


def child_path(path: LogicalPath, token: PathToken) -> LogicalPath:
    return (*path, token)


def key_path(path: LogicalPath, token: MappingKeyToken) -> LogicalPath:
    return (*path, token, _KEY_SUFFIX)


def mapping_key_token_for(key: object) -> MappingKeyToken:
    display = _mapping_display(key)
    return MappingKeyToken(display=display, fingerprint=_mapping_fingerprint(key))


def artifact_dir_for(path: LogicalPath, artifacts_root: Path) -> Path:
    if not path:
        return artifacts_root / _ROOT_SEGMENT
    return artifacts_root.joinpath(*(_segment_for_token(token) for token in path))


def artifact_path_for(path: LogicalPath) -> str:
    return artifact_dir_for(path, Path("artifacts")).as_posix()


def format_logical_path(path: LogicalPath) -> str:
    if not path:
        return "<root>"
    parts: list[str] = []
    for token in path:
        if isinstance(token, str):
            if token == _KEY_SUFFIX:
                parts.append("<key>")
            elif parts:
                parts.append(f".{token}")
            else:
                parts.append(token)
        elif isinstance(token, int):
            parts.append(f"[{token}]")
        else:
            parts.append(f"[{token.display!r}]")
    return "".join(parts)


def _segment_for_token(token: PathToken) -> str:
    if isinstance(token, str):
        return _safe_segment(token)
    if isinstance(token, int):
        if token < 0:
            raise ValueError(f"sequence indices must be non-negative, got {token}")
        return f"{token:06d}"
    slug = _safe_segment(token.display)
    digest = _short_hash(token.fingerprint)
    return f"{slug}--{digest}"


def _mapping_display(key: object) -> str:
    if isinstance(key, str):
        return key
    if key is None or isinstance(key, bool | int | float):
        return repr(key)
    return type(key).__name__


def _mapping_fingerprint(key: object) -> str:
    if key is None or isinstance(key, bool | int | float | str):
        return repr(key)
    if isinstance(key, tuple):
        return "(" + ",".join(_mapping_fingerprint(item) for item in key) + ")"
    if isinstance(key, frozenset):
        parts = sorted(_mapping_fingerprint(item) for item in key)
        return "frozenset(" + ",".join(parts) + ")"
    return f"{type(key).__module__}.{type(key).__qualname__}:{key!r}"


def _safe_segment(text: str) -> str:
    slug = _SEGMENT_RE.sub("-", text).strip("._-")
    if not slug:
        slug = "item"
    if len(slug) > _MAX_SEGMENT_LENGTH:
        slug = slug[:_MAX_SEGMENT_LENGTH].rstrip("._-")
    if slug in {".", ".."}:
        slug = f"item-{_short_hash(text)}"
    return slug


def _short_hash(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=5).hexdigest()
