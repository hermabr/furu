from __future__ import annotations

import hashlib
import inspect
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any

from furu.utils import JsonValue, _hash_dict_deterministically


@dataclass(frozen=True, kw_only=True)
class CodeFingerprint:
    create_hook_hash: str

    def to_json(self) -> JsonValue:
        return {
            "create_hook_hash": self.create_hook_hash,
        }


@dataclass(frozen=True, kw_only=True)
class DependencyFingerprint:
    python_version: str
    platform: str
    pyproject_hash: str | None
    uv_lock_hash: str | None

    def to_json(self) -> JsonValue:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "pyproject_hash": self.pyproject_hash,
            "uv_lock_hash": self.uv_lock_hash,
        }


@dataclass(frozen=True, kw_only=True)
class IdentitySpec:
    version: int
    class_name: str
    artifact_schema_hash: str
    artifact_hash: str
    code: CodeFingerprint
    dependencies: DependencyFingerprint

    def to_json(self) -> JsonValue:
        return {
            "version": self.version,
            "class_name": self.class_name,
            "artifact_schema_hash": self.artifact_schema_hash,
            "artifact_hash": self.artifact_hash,
            "code": self.code.to_json(),
            "dependencies": self.dependencies.to_json(),
        }

    @property
    def hash(self) -> str:
        return _hash_dict_deterministically(self.to_json())


def code_fingerprint(cls: type[Any]) -> CodeFingerprint:
    hook = _create_hook(cls)
    return CodeFingerprint(create_hook_hash=_hash_bytes(_callable_bytes(hook)))


def dependency_fingerprint(project_root: Path | None = None) -> DependencyFingerprint:
    root = project_root or Path.cwd()
    return DependencyFingerprint(
        python_version=sys.version,
        platform=platform.platform(),
        pyproject_hash=_file_hash(root / "pyproject.toml"),
        uv_lock_hash=_file_hash(root / "uv.lock"),
    )


def _create_hook(cls: type[Any]) -> Any:
    mode = getattr(cls, "_furu_create_mode", None)
    if mode == "batched":
        return getattr(cls, "_create_batched")
    return getattr(cls, "_create")


def _callable_bytes(value: Any) -> bytes:
    try:
        return inspect.getsource(value).encode()
    except (OSError, TypeError):
        function = _unwrap_function(value)
        if function is not None:
            code = function.__code__
            parts = [
                code.co_code,
                repr(code.co_consts).encode(),
                repr(code.co_names).encode(),
            ]
            return b"\0".join(parts)
        return repr(value).encode()


def _unwrap_function(value: Any) -> FunctionType | None:
    if isinstance(value, FunctionType):
        return value
    function = getattr(value, "__func__", None)
    if isinstance(function, FunctionType):
        return function
    return None


def _file_hash(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return _hash_bytes(data)


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2s(data, digest_size=10).hexdigest()
