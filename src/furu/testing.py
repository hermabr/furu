import hashlib
import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

import furu.config as furu_config


@dataclass(slots=True)
class _FuruPytestState:  # TODO: maybe make this auto snapshot the config
    original_data_dir: Path
    run_data_dir: Path


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
        / f"furu-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    )
    # run_base_directory.mkdir(parents=True, exist_ok=True)

    state = _FuruPytestState(
        original_data_dir=furu_config.settings.data_dir,
        run_data_dir=run_base_directory,
    )

    furu_config.configure(data_dir=state.run_data_dir)
    config.stash[_STATE_KEY] = state


def pytest_unconfigure(config: pytest.Config) -> None:
    if not _is_furu_pytest_mode_enabled():
        return

    state = config.stash[_STATE_KEY]

    furu_config.configure(data_dir=state.original_data_dir)

    if not _keep_furu_data():
        shutil.rmtree(state.run_data_dir, ignore_errors=True)
    else:
        print(f"kept furu data at {state.run_data_dir}")


@pytest.fixture(autouse=True)
def _furu_per_test_base_directory(
    request: pytest.FixtureRequest,
    pytestconfig: pytest.Config,
) -> Iterator[None]:
    if not _is_furu_pytest_mode_enabled():
        yield
        return

    state = pytestconfig.stash[_STATE_KEY]
    previous_data_dir = furu_config.settings.data_dir

    test_base_directory = (
        state.run_data_dir
        / hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    )

    furu_config.configure(data_dir=test_base_directory)
    try:
        yield
    finally:
        furu_config.configure(data_dir=previous_data_dir)
