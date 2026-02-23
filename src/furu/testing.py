import hashlib
import os
import secrets
import shutil
import tempfile
import types
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


def _is_furu_pytest_mode_enabled() -> bool:
    return os.environ.get("FURU_PYTEST_MODE", "on").strip().lower() != "off"


def pytest_configure(config: pytest.Config) -> None:
    if not _is_furu_pytest_mode_enabled():
        return

    run_base_directory = (
        Path(tempfile.gettempdir())
        / f"furu-data-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    )
    # run_base_directory.mkdir(parents=True, exist_ok=True)

    state = _FuruPytestState(
        original_directories=furu_config.directories,
        run_directories=_FuruDirectories(
            data=run_base_directory,
        ),
    )

    furu_config.directories = state.run_directories
    config.stash[_STATE_KEY] = state


def pytest_unconfigure(config: pytest.Config) -> None:
    if not _is_furu_pytest_mode_enabled():
        return

    state = config.stash[_STATE_KEY]

    furu_config.directories = state.original_directories

    if os.environ.get("FURU_PYTEST_KEEP", "clean").strip().lower() != "keep":
        # TODO: how do we make sure we delete everything?
        shutil.rmtree(state.run_directories.data)


@pytest.fixture(autouse=True)
def _furu_per_test_base_directory(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
) -> types.GeneratorType:
    if not _is_furu_pytest_mode_enabled():
        yield
        return

    state = pytestconfig.stash[_STATE_KEY]
    previous_base_directory = furu_config.directories

    test_base_directory = (
        state.run_directories.data
        / hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    )

    furu_config.directories = _FuruDirectories(data=test_base_directory)
    try:
        yield
    finally:
        furu_config.directories = previous_base_directory
