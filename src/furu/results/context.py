from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from furu.results.nodes import ManifestValue
from furu.results.paths import LogicalPath, PathToken, encode_artifact_dir

if TYPE_CHECKING:
    from furu.config import ResultConfig


type DumpValueFn = Callable[[object, "DumpContext", object | None, bool], ManifestValue]
type LoadValueFn = Callable[[ManifestValue, "LoadContext"], object]


@dataclass(frozen=True, slots=True)
class DumpContext:
    result_dir: Path
    config: "ResultConfig"
    path: LogicalPath
    _dump_value: DumpValueFn

    def child(self, token: PathToken) -> "DumpContext":
        return replace(self, path=self.path.child(token))

    def dump(
        self,
        value: object,
        *,
        token: PathToken | None = None,
        annotation: object | None = None,
        inline: bool = False,
    ) -> ManifestValue:
        child = self.child(token) if token is not None else self
        return self._dump_value(value, child, annotation, inline)

    @property
    def artifact_dir(self) -> Path:
        return self.result_dir / encode_artifact_dir(self.path)

    @property
    def artifact_dir_rel(self) -> str:
        return encode_artifact_dir(self.path)


@dataclass(frozen=True, slots=True)
class LoadContext:
    result_dir: Path
    config: "ResultConfig"
    path: LogicalPath
    _load_value: LoadValueFn

    def child(self, token: PathToken) -> "LoadContext":
        return replace(self, path=self.path.child(token))

    def load(self, node: ManifestValue, *, token: PathToken | None = None) -> object:
        child = self.child(token) if token is not None else self
        return self._load_value(node, child)

    def resolve_artifact_dir(self, artifact_dir: str) -> Path:
        return self.result_dir / artifact_dir
