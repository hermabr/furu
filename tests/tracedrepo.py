"""A scratch package whose code furu traces, for tests that edit it."""

import importlib
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from furu import code_trace, remember

# Source files get distinct mtimes in the past, so furu trusts that a module
# imported before tracing started matches the file on disk, and every rewrite
# invalidates bytecode and parse caches.
_MTIMES = itertools.count(1_000_000_000)


def fresh_process() -> None:
    """Forget what this process decided once and has not saved.

    That is which code versions it found live, and what it read and kept of
    what dependency methods returned.
    """
    code_trace._CHECKED.clear()
    remember._LINES.clear()
    remember._UNSAVED.clear()


@dataclass
class Repo:
    root: Path
    package: str

    def write(self, name: str, text: str) -> None:
        path = self.root / self.package / name
        path.write_text(text.replace("PKG", self.package), encoding="utf-8")
        mtime = next(_MTIMES) * 1_000_000_000
        os.utime(path, ns=(mtime, mtime))
        fresh_process()

    def read(self, name: str) -> str:
        text = (self.root / self.package / name).read_text(encoding="utf-8")
        return text.replace(self.package, "PKG")

    def edit(self, name: str, old: str, new: str) -> None:
        text = self.read(name)
        assert old in text, old
        self.write(name, text.replace(old, new))

    def load(self, name: str) -> ModuleType:
        """Import the package's current code, as a fresh process would."""
        for module in [m for m in sys.modules if m.split(".")[0] == self.package]:
            del sys.modules[module]
        code_trace._IMPORTED.clear()
        importlib.invalidate_caches()
        return importlib.import_module(f"{self.package}.{name.removesuffix('.py')}")
