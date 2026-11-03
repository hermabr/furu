import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from gitrepo import init_repo

from furu import provenance
from furu.config import _project_anchor

# Set once here so xdist workers and every child process they start inherit it.
os.environ.update(
    {
        # Worker children import spec classes by fully qualified name from the
        # test modules in this directory.
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")])
        ),
        # Scratch repos ignore the user's git config (hooks, signing, excludes).
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t.t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t.t",
        # Process start-up dominates the suite: numpy starts an OpenBLAS thread
        # per core at import (about 1s of CPU), and pydantic scans every
        # installed package for plugins (about 80ms per worker child).
        "OPENBLAS_NUM_THREADS": "1",
        "PYDANTIC_DISABLE_PLUGINS": "__all__",
    }
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A repo on main with a tracked file, committed and pushed to origin."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.txt").write_text("content\n")
    return init_repo(repo)


@pytest.fixture
def fresh_caches(pytestconfig: pytest.Config) -> Iterator[None]:
    """Empty furu's per-process caches of the project and machine, for tests
    that chdir into or fake them, then re-prime them from this repo afterwards."""
    caches = (
        provenance.EnvironmentIdentity.capture,
        provenance._require_uv,
        provenance._probe_accelerators,
        _project_anchor,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()
    with contextlib.chdir(pytestconfig.rootpath):
        provenance.EnvironmentIdentity.capture()
        _project_anchor()
