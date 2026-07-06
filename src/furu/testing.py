import contextlib
import hashlib
import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from furu import provenance
from furu.config import _Config, _FuruDirectories, _set_config, get_config


@contextmanager
def override_config(config: _Config) -> Iterator[None]:
    previous = get_config()
    _set_config(config)
    try:
        yield
    finally:
        _set_config(previous)


@dataclass(slots=True)
class _FuruPytestState:
    original_config: _Config
    run_config: _Config


_STATE_KEY = pytest.StashKey[_FuruPytestState]()


def _is_furu_pytest_mode_enabled() -> bool:
    return os.environ.get("FURU_PYTEST_MODE", "on").strip().lower() != "off"


def _keep_furu_data() -> bool:
    return os.environ.get("FURU_PYTEST_KEEP", "clean").strip().lower() == "keep"


def _replace_config_directories(
    config: _Config,
    directories: _FuruDirectories,
) -> _Config:
    data = config.model_dump()
    data["directories"] = directories
    return _Config.model_validate(data)


def pytest_configure(config: pytest.Config) -> None:
    if not _is_furu_pytest_mode_enabled():
        return

    # Tests chdir freely, so prime the per-process environment capture while
    # cwd is still the project root. Recording is never skipped; a missing
    # project surfaces at the first create() with the real error.
    with contextlib.suppress(RuntimeError):
        provenance.EnvironmentIdentity.capture()

    run_base_directory = (
        Path(tempfile.gettempdir())
        / f"furu-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    )

    run_config = _replace_config_directories(
        get_config(),
        _FuruDirectories(
            objects=run_base_directory / "objects",
            executions=run_base_directory / "executions",
            debug=run_base_directory / "debug",
        ),
    )

    state = _FuruPytestState(
        original_config=get_config(),
        run_config=run_config,
    )
    _set_config(run_config)
    config.stash[_STATE_KEY] = state


def pytest_unconfigure(config: pytest.Config) -> None:
    if not _is_furu_pytest_mode_enabled():
        return

    state = config.stash[_STATE_KEY]

    _set_config(state.original_config)

    if not _keep_furu_data():
        for field_name in type(state.run_config.directories).model_fields:
            path = getattr(state.run_config.directories, field_name)
            shutil.rmtree(path, ignore_errors=True)
    else:
        print(f"kept furu objects at {state.run_config.directories.objects}")
        print(f"kept furu executions at {state.run_config.directories.executions}")
        print(f"kept furu debug at {state.run_config.directories.debug}")


@pytest.fixture(autouse=True)
def _furu_per_test_base_directory(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
) -> Iterator[None]:
    if not _is_furu_pytest_mode_enabled():
        yield
        return

    state = pytestconfig.stash[_STATE_KEY]
    previous_config = get_config()

    test_id = hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    test_objects_directory = state.run_config.directories.objects / test_id
    test_executions_directory = state.run_config.directories.executions / test_id

    _set_config(
        _replace_config_directories(
            previous_config,
            _FuruDirectories(
                objects=test_objects_directory,
                executions=test_executions_directory,
                debug=state.run_config.directories.debug / test_id,
            ),
        )
    )
    try:
        yield
    finally:
        _set_config(previous_config)
