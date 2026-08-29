"""Spec module with an import-time side effect.

Importing this module writes a marker file named after the importing pid to
``$FURU_TEST_IMPORT_MARKER_DIR``; the subprocess tests use it to prove that
only the worker's child process imports spec modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from furu import Spec

if _marker_dir := os.environ.get("FURU_TEST_IMPORT_MARKER_DIR"):
    (Path(_marker_dir) / str(os.getpid())).touch()


class ImportSideEffectLeaf(Spec[int]):
    def create(self) -> int:
        return os.getpid()
