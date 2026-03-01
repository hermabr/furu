import hashlib
import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, fields
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


def _keep_furu_data() -> bool:
    return os.environ.get("FURU_PYTEST_KEEP", "clean").strip().lower() == "keep"


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

    if not _keep_furu_data():
        for field in fields(state.run_directories):
            path = getattr(state.run_directories, field.name)
            if not isinstance(path, Path):
                raise TypeError(
                    f"Expected Path for run directory '{field.name}', got {type(path).__name__}"
                )
            shutil.rmtree(path, ignore_errors=True)
    else:
        print(f"kept furu data at {state.run_directories.data}")


@pytest.fixture(autouse=True)
def _furu_per_test_base_directory(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
) -> Iterator[None]:
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
