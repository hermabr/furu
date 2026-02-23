import hashlib
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from furu.config import _FuruDirectories
from furu.config import config as furu_config


@dataclass(slots=True)
class _FuruPytestState:  # TODO: maybe make this auto snapshot the config
    original_directories: _FuruDirectories
    run_directories: _FuruDirectories


_STATE_KEY = pytest.StashKey[_FuruPytestState]()


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("FURU_PYTEST_MODE", "on").strip().lower() == "off":
        return

    run_base_directory = (
        Path(tempfile.gettempdir())
        / f"furu-data-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    )
    run_base_directory.mkdir(parents=True, exist_ok=True)

    state = _FuruPytestState(
        original_directories=furu_config.directories,
        run_directories=_FuruDirectories(
            data=run_base_directory / "data",
        ),
    )

    furu_config.directories = state.run_directories
    config.stash[_STATE_KEY] = state


def pytest_unconfigure(config: pytest.Config) -> None:
    state = config.stash[_STATE_KEY]
    furu_config.directories = state.original_directories
    if os.environ.get("FURU_PYTEST_KEEP", "clean") == "keep":
        return
    shutil.rmtree(
        state.run_directories.data, ignore_errors=True
    )  # TODO: decide how we want to handle the raw data


@pytest.fixture(scope="session")
def furu_tmp_root(pytestconfig: pytest.Config) -> _FuruDirectories:
    return pytestconfig.stash[_STATE_KEY].run_directories


@pytest.fixture(autouse=True)
def _furu_per_test_base_directory(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
):
    state = pytestconfig.stash[_STATE_KEY]
    previous_base_directory = furu_config.directories

    (
        test_base_directory := (
            state.run_directories.data
            / hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
        )
    ).mkdir(parents=True, exist_ok=True)

    furu_config.directories = _FuruDirectories(data=test_base_directory)
    try:
        yield
    finally:
        furu_config.directories = previous_base_directory
