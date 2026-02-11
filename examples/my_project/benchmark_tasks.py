from __future__ import annotations

import furu
from furu import furu_dep
from furu.core import Furu


class TinyLeaf(Furu[int]):
    node_id: int = furu.chz.field()

    def _create(self) -> int:
        value = self.node_id % 97
        (self.furu_dir / "value.txt").write_text(str(value))
        return value

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())


class TinyMerge(Furu[int]):
    node_id: int = furu.chz.field()
    deps: list[Furu[int]] = furu.chz.field(default_factory=list)

    @furu_dep
    def task_dependencies(self) -> furu.DependencySpec:
        return self.deps

    def _create(self) -> int:
        total = self.node_id
        for dep in self.deps:
            total += dep.get()
        (self.furu_dir / "value.txt").write_text(str(total))
        return total

    def _load(self) -> int:
        return int((self.furu_dir / "value.txt").read_text())
