import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _child_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Workers run create() in a child process that imports spec classes by
    # fully qualified name; the test modules defining them live here.
    tests_directory = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        tests_directory if not existing else f"{tests_directory}{os.pathsep}{existing}",
    )
