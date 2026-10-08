import os
from pathlib import Path

import pytest
from canned_probes import CANNED_PROBES


@pytest.fixture(autouse=True)
def _child_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Workers run create() in a child process that imports spec classes by
    # fully qualified name; the test modules defining them live here. This also
    # puts tests/sitecustomize.py in front of those children.
    tests_directory = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        tests_directory if not existing else f"{tests_directory}{os.pathsep}{existing}",
    )


@pytest.fixture(autouse=True)
def _canned_probes(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("real_probes"):
        return
    for (owner, name), fake in CANNED_PROBES.items():
        monkeypatch.setattr(owner, name, fake)
